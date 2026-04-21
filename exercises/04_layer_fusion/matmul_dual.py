# matmul_dual.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  GIVEN — Dual-tile parallel vectorized matrix multiplication        │
# │                                                                     │
# │  Two AIE cores each compute half of the output matrix C = A × B.   │
# │  A single shim-to-mem FIFO is split at the mem tile to distribute  │
# │  A tile-rows to two cores.  B is broadcast from mem tile to both   │
# │  cores.  Each core outputs C tiles via its own mem-to-shim path.   │
# │  DMA layout transforms (same as exercise 03) rearrange data for    │
# │  the vectorized kernel.                                             │
# └──────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   python3 matmul_dual.py > build/matmul_dual/aie.mlir
#
# =========================================================================
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def matmul_dual():
    # ------------------------------------------------------------------
    # 1. Dimensions
    # ------------------------------------------------------------------
    M, K, N = 256, 256, 256
    m, k, n = 64, 64, 64
    dtype = np.int16

    r, s, t = 4, 4, 8

    M_div_m = M // m                    # 4
    K_div_k = K // k                    # 4
    N_div_n = N // n                    # 4

    n_cores = 2
    tiles_per_core = (M_div_m * N_div_n) // n_cores  # 8

    # ------------------------------------------------------------------
    # 2. Types
    # ------------------------------------------------------------------
    A_ty = np.ndarray[(M * K,), np.dtype[dtype]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype]]
    # Each inA element carries 2 A tiles (one per core) concatenated
    a_pair_ty = np.ndarray[(2 * m * k,), np.dtype[dtype]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype]]

    # ------------------------------------------------------------------
    # 3. Kernels
    # ------------------------------------------------------------------
    zero_fn = Kernel("zero_i16", "mm.cc.o", [c_ty])
    matmul_fn = Kernel("matmul_i16_i16", "mm.cc.o", [a_ty, b_ty, c_ty])

    # ------------------------------------------------------------------
    # 4. DMA layout transforms (same as exercise 03)
    # ------------------------------------------------------------------
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]

    # ------------------------------------------------------------------
    # 5. ObjectFIFOs
    # ------------------------------------------------------------------
    # A: one shim FIFO carrying 2 tiles, split at mem tile to 2 cores
    # (NPU2Col1 shim has only 2 input + 2 output DMA channels)
    inA = ObjectFifo(a_pair_ty, name="inA")
    memA_fifos = inA.cons().split(
        offsets=[0, m * k],
        obj_types=[a_ty, a_ty],
        names=["memA0", "memA1"],
        dims_to_stream=[a_dims, a_dims],
    )

    # B: one shim FIFO, forwarded through mem tile, broadcast to both cores
    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    # C: separate per-core paths (2 shim input channels)
    memC0 = ObjectFifo(c_ty, name="memC0")
    outC0 = memC0.cons().forward(name="outC0", dims_to_stream=c_dims)

    memC1 = ObjectFifo(c_ty, name="memC1")
    outC1 = memC1.cons().forward(name="outC1", dims_to_stream=c_dims)

    # ------------------------------------------------------------------
    # 6. Core function (same for both cores)
    # ------------------------------------------------------------------
    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles_per_core):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    # ------------------------------------------------------------------
    # 7. Workers
    # ------------------------------------------------------------------
    worker0 = Worker(
        core_fn,
        [memA_fifos[0].cons(), memB.cons(), memC0.prod(), zero_fn, matmul_fn],
    )
    worker1 = Worker(
        core_fn,
        [memA_fifos[1].cons(), memB.cons(), memC1.prod(), zero_fn, matmul_fn],
    )

    # ------------------------------------------------------------------
    # 8. Runtime sequence
    # ------------------------------------------------------------------
    # A pair taps: each sends 2 consecutive tile-rows of A (128×K),
    # repeated N_div_n times for each output column
    a_pair_taps = TensorTiler2D.group_tiler(
        (M, K), (n_cores * m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    # B tap: all K×N tiles in column-major tile-group order
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
    )[0]
    # C taps: each covers 1 tile-row of C
    c_taps = TensorTiler2D.group_tiler(
        (M, N), (m, n), (1, N_div_n)
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker0, worker1)

        # Process in pairs: pair 0 → A rows 0-1, pair 1 → A rows 2-3
        rows_per_core = M_div_m // n_cores  # 2
        for pair in range(rows_per_core):
            tr0 = 2 * pair          # core 0 tile-row index
            tr1 = 2 * pair + 1      # core 1 tile-row index

            tg = rt.task_group()
            rt.fill(inA.prod(), A, tap=a_pair_taps[pair], task_group=tg)
            rt.fill(inB.prod(), B, tap=b_tap, task_group=tg)
            rt.drain(outC0.cons(), C, tap=c_taps[tr0], task_group=tg, wait=True)
            rt.drain(outC1.cons(), C, tap=c_taps[tr1], task_group=tg, wait=True)
            rt.finish_task_group(tg)

    # ------------------------------------------------------------------
    # 9. Compile
    # ------------------------------------------------------------------
    return Program(NPU2Col1(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    print(matmul_dual())
