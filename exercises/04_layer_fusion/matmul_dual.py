# matmul_dual.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  GIVEN — Dual-tile parallel vectorized matrix multiplication        │
# │                                                                     │
# │  Two AIE cores each compute half of the output matrix C = A × B.   │
# │  Core 0 handles tile-rows 0–1, Core 1 handles tile-rows 2–3.       │
# │  Each core has its own A, B and C ObjectFIFOs forwarded through     │
# │  the mem tile with DMA layout transforms (same as exercise 03).     │
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
from aie.iron.device import NPU2
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
    # 5. ObjectFIFOs — one set per core, forwarded through mem tile
    # ------------------------------------------------------------------
    inA0 = ObjectFifo(a_ty, name="inA0")
    memA0 = inA0.cons().forward(name="memA0", dims_to_stream=a_dims)

    inB0 = ObjectFifo(b_ty, name="inB0")
    memB0 = inB0.cons().forward(name="memB0", dims_to_stream=b_dims)

    memC0 = ObjectFifo(c_ty, name="memC0")
    outC0 = memC0.cons().forward(name="outC0", dims_to_stream=c_dims)

    inA1 = ObjectFifo(a_ty, name="inA1")
    memA1 = inA1.cons().forward(name="memA1", dims_to_stream=a_dims)

    inB1 = ObjectFifo(b_ty, name="inB1")
    memB1 = inB1.cons().forward(name="memB1", dims_to_stream=b_dims)

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
        [memA0.cons(), memB0.cons(), memC0.prod(), zero_fn, matmul_fn],
    )
    worker1 = Worker(
        core_fn,
        [memA1.cons(), memB1.cons(), memC1.prod(), zero_fn, matmul_fn],
    )

    # ------------------------------------------------------------------
    # 8. Runtime sequence
    # ------------------------------------------------------------------
    # A taps: each tap covers 1 tile-row of A, repeated N_div_n times
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    # B tap: all K×N tiles in column-major tile-group order
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
    )[0]
    # C taps: each tap covers 1 tile-row of C
    c_taps = TensorTiler2D.group_tiler(
        (M, N), (m, n), (1, N_div_n)
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker0, worker1)

        # Process tile-rows in pairs: (0,2), (1,3)
        rows_per_core = M_div_m // n_cores  # 2
        for pair in range(rows_per_core):
            tr0 = pair                       # core 0: tile-rows 0, 1
            tr1 = pair + rows_per_core       # core 1: tile-rows 2, 3

            tg = rt.task_group()
            rt.fill(inA0.prod(), A, tap=a_taps[tr0], task_group=tg)
            rt.fill(inA1.prod(), A, tap=a_taps[tr1], task_group=tg)
            rt.fill(inB0.prod(), B, tap=b_tap, task_group=tg)
            rt.fill(inB1.prod(), B, tap=b_tap, task_group=tg)
            rt.drain(outC0.cons(), C, tap=c_taps[tr0], task_group=tg, wait=True)
            rt.drain(outC1.cons(), C, tap=c_taps[tr1], task_group=tg, wait=True)
            rt.finish_task_group(tg)

    # ------------------------------------------------------------------
    # 9. Compile
    # ------------------------------------------------------------------
    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    print(matmul_dual())
