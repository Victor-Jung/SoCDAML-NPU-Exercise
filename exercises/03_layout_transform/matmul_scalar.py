# matmul_scalar.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  GIVEN — Scalar single-core matrix multiplication                   │
# │                                                                     │
# │  This design performs C = A × B using a naive (non-vectorized)      │
# │  kernel.  Data stays in row-major order — no DMA layout transform.  │
# │  Use this as the baseline for comparison with the vectorized        │
# │  design that you will complete in matmul_vectorized.py.             │
# └──────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   python3 matmul_scalar.py > build/scalar/aie.mlir
#
# =========================================================================
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def matmul_scalar():
    # ------------------------------------------------------------------
    # 1. Dimensions
    # ------------------------------------------------------------------
    M, K, N = 128, 128, 128       # outer matrix size
    m, k, n = 64, 64, 64          # tile size (fits in AIE L1)
    dtype = np.int16

    M_div_m = M // m               # 2
    K_div_k = K // k               # 2
    N_div_n = N // n               # 2
    tiles = M_div_m * N_div_n      # 4 output tiles total

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
    # 3. Kernels (scalar — row-major data, no DMA transform needed)
    # ------------------------------------------------------------------
    zero_fn = Kernel("zero_i16", "mm.cc.o", [c_ty])
    matmul_fn = Kernel("matmul_scalar_i16_i16", "mm.cc.o", [a_ty, b_ty, c_ty])

    # ------------------------------------------------------------------
    # 4. ObjectFIFOs — forwarded through Mem tile, NO transforms
    # ------------------------------------------------------------------
    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA")          # row-major in, row-major out

    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB")          # row-major in, row-major out

    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC")          # row-major in, row-major out

    # ------------------------------------------------------------------
    # 5. Core function
    # ------------------------------------------------------------------
    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_fn, matmul_fn],
    )

    # ------------------------------------------------------------------
    # 6. Runtime sequence — tile the outer matrices
    # ------------------------------------------------------------------
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
    )[0]
    c_taps = TensorTiler2D.group_tiler(
        (M, N), (m, n), (1, N_div_n)
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        for tile_row in range(M_div_m):
            tg = rt.task_group()
            rt.fill(inA.prod(), A, tap=a_taps[tile_row], task_group=tg)
            rt.fill(inB.prod(), B, tap=b_tap, task_group=tg)
            rt.drain(outC.cons(), C, tap=c_taps[tile_row], task_group=tg, wait=True)
            rt.finish_task_group(tg)

    # ------------------------------------------------------------------
    # 7. Compile
    # ------------------------------------------------------------------
    return Program(NPU2Col1(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    print(matmul_scalar())
