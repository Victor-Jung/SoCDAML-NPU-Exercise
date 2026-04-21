# matmul_vectorized.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  EXERCISE                                                           │
# │                                                                     │
# │  Vectorized single-core matrix multiplication with DMA layout       │
# │  transforms.  The AIE vector unit needs data in sub-tile order      │
# │  (r×s for A, s×t for B, r×t for C).  The Mem-tile DMA re-arranges  │
# │  data on-the-fly using (size, stride) descriptors.                  │
# │                                                                     │
# │  Your task: compute the three DMA transform descriptors             │
# │  (a_dims, b_dims, c_dims) so that data arrives in the layout        │
# │  expected by the vectorized kernel.                                 │
# └──────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   python3 matmul_vectorized.py > build/vectorized/aie.mlir
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

    # AIE microkernel intrinsic dimensions (npu2, int16).
    # The vectorized kernel uses aie::mmul<r, s, t> which multiplies
    # an (r×s) sub-tile of A with an (s×t) sub-tile of B to produce
    # an (r×t) sub-tile of C.
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
    # The Mem-tile DMA can re-arrange data on the fly using up to 4
    # nested (size, stride) pairs — equivalent to 4 nested for loops:
    #
    #   for i0 in range(size_0):
    #     for i1 in range(size_1):
    #       for i2 in range(size_2):
    #         for i3 in range(size_3):
    #           access buffer[i0*stride_0 + i1*stride_1
    #                       + i2*stride_2 + i3*stride_3]
    #
    # Your task: derive (size, stride) tuples that re-tile the row-major
    # m×k / k×n buffers into the sub-tile order expected by the
    # vectorized kernel, and that un-tile the m×n output back to
    # row-major.
    #
    # The sub-tile sizes are:
    #   A sub-tile: r × s = 4 × 4   (16 elements)
    #   B sub-tile: s × t = 4 × 8   (32 elements)
    #   C sub-tile: r × t = 4 × 8   (32 elements)

    # ── TODO: A transform (row-major m×k → tiled) ────────────────────
    #
    # The DMA must read an m×k row-major buffer and stream it out as
    # (m/r)×(k/s) sub-tiles, each containing r×s elements in row-major
    # order within the sub-tile.
    #
    # Hint: the 4 loop levels iterate over:
    #   - sub-tile rows    (m // r groups)
    #   - sub-tile columns (k // s groups)
    #   - rows within sub-tile (r rows)
    #   - columns within sub-tile (s columns)
    #
    a_dims = ???

    # ── TODO: B transform (row-major k×n → tiled) ────────────────────
    #
    # Same idea but the sub-tile is s×t instead of r×s.
    #
    b_dims = ???

    # ── TODO: C transform (tiled → row-major m×n) ────────────────────
    #
    # The kernel writes C in sub-tile order.  The DMA must read the
    # tiled buffer and stream it out in row-major order.
    #
    # Hint: the 4 loop levels now iterate over:
    #   - sub-tile rows    (m // r groups)
    #   - rows within sub-tile (r rows)
    #   - sub-tile columns (n // t groups)
    #   - columns within sub-tile (t columns)
    #
    c_dims = ???

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
