# matmul_vectorized_solution.py
#
# SOLUTION — Vectorized single-core matrix multiplication
#            with DMA layout transforms.
#
# Usage:
#   python3 matmul_vectorized_solution.py > build/vectorized/aie.mlir
#
# =========================================================================
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def matmul_vectorized():
    # ------------------------------------------------------------------
    # 1. Dimensions
    # ------------------------------------------------------------------
    M, K, N = 128, 128, 128       # outer matrix size
    m, k, n = 64, 64, 64          # tile size (fits in AIE L1)
    dtype = np.int16

    # AIE microkernel intrinsic dimensions (npu2, int16)
    r, s, t = 4, 4, 8

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
    # 3. Kernels (vectorized — expects tiled data from DMA transforms)
    # ------------------------------------------------------------------
    zero_fn = Kernel("zero_i16", "mm.cc.o", [c_ty])
    matmul_fn = Kernel("matmul_i16_i16", "mm.cc.o", [a_ty, b_ty, c_ty])

    # ------------------------------------------------------------------
    # 4. DMA layout transforms
    # ------------------------------------------------------------------
    # The vectorized kernel expects data in sub-tile order:
    #   A: (m/r)×(k/s) sub-tiles of r×s     (row-major → tiled)
    #   B: (k/s)×(n/t) sub-tiles of s×t     (row-major → tiled)
    #   C: (m/r)×(n/t) sub-tiles of r×t     (tiled → row-major)
    #
    # Each transform is a list of 4 (size, stride) tuples describing
    # how the Mem-tile DMA reads from its buffer (outermost to innermost).

    # A: row-major m×k → (r×s)-tiled
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]

    # B: row-major k×n → (s×t)-tiled
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]

    # C: (r×t)-tiled → row-major m×n
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]

    # ------------------------------------------------------------------
    # 5. ObjectFIFOs — forwarded through Mem tile WITH transforms
    # ------------------------------------------------------------------
    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    # ------------------------------------------------------------------
    # 6. Core function (identical to scalar — only data layout differs)
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
    # 7. Runtime sequence — tile the outer matrices
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
    # 8. Compile
    # ------------------------------------------------------------------
    return Program(NPU2Col1(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    print(matmul_vectorized())
