# matmul_relu_pipeline.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  TEMPLATE — 2-stage pipeline: matmul → relu                        │
# │                                                                     │
# │  Core 0: computes C tiles (matmul) and sends them to Core 1.       │
# │  Core 1: receives C tiles, applies relu, and sends D tiles out.    │
# │                                                                     │
# │  The C tiles flow directly from Core 0 to Core 1 via a             │
# │  core-to-core ObjectFifo (no DRAM round-trip).                     │
# │  The DMA un-tiles the output D via the mem tile c_dims transform.  │
# │                                                                     │
# │  Fill in the ??? sections to complete the design.                   │
# └──────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   python3 matmul_relu_pipeline.py > build/pipeline/aie.mlir
#
# =========================================================================
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def matmul_relu_pipeline():
    # ------------------------------------------------------------------
    # 1. Dimensions
    # ------------------------------------------------------------------
    M, K, N = 4096, 256, 4096
    m, k, n = 64, 64, 64
    dtype = np.int16

    r, s, t = 4, 4, 8

    M_div_m = M // m                    # 64
    K_div_k = K // k                    # 4
    N_div_n = N // n                    # 64

    total_tiles = M_div_m * N_div_n     # 4096

    # ------------------------------------------------------------------
    # 2. Types
    # ------------------------------------------------------------------
    A_ty = np.ndarray[(M * K,), np.dtype[dtype]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype]]
    D_ty = np.ndarray[(M * N,), np.dtype[dtype]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype]]

    # ------------------------------------------------------------------
    # 3. Kernels
    # ------------------------------------------------------------------
    zero_fn = Kernel("zero_i16", "mm.cc.o", [c_ty])
    matmul_fn = Kernel("matmul_i16_i16", "mm.cc.o", [a_ty, b_ty, c_ty])
    relu_fn = Kernel("relu_fused_i16", "mm.cc.o", [c_ty, c_ty])

    # ------------------------------------------------------------------
    # 4. DMA layout transforms
    # ------------------------------------------------------------------
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]

    # ------------------------------------------------------------------
    # 5. ObjectFIFOs
    # ------------------------------------------------------------------
    # A: shim → mem tile (with tiling transform) → Core 0
    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # B: shim → mem tile (with tiling transform) → Core 0
    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    # ┌──────────────────────────────────────────────────────────────┐
    # │  ???: Create a core-to-core ObjectFifo named "coreC" that   │
    # │  carries c_ty tiles from Core 0 (producer) to Core 1        │
    # │  (consumer).  This is just a regular ObjectFifo — the       │
    # │  placer will put producer and consumer on different tiles.   │
    # │  Use depth 2 (double-buffered) for pipelining.              │
    # └──────────────────────────────────────────────────────────────┘
    coreC = ???

    # D: Core 1 → mem tile (with un-tiling transform) → shim
    # ┌──────────────────────────────────────────────────────────────┐
    # │  ???: Create an ObjectFifo for the relu output, forwarded   │
    # │  through the mem tile with c_dims transform to un-tile      │
    # │  back to row-major layout.  Name the inner FIFO "memD" and  │
    # │  the outer FIFO "outD".                                     │
    # └──────────────────────────────────────────────────────────────┘
    memD = ???
    outD = ???

    # ------------------------------------------------------------------
    # 6. Core functions
    # ------------------------------------------------------------------
    # Core 0: matmul producer — computes C tiles and sends to Core 1
    # ┌──────────────────────────────────────────────────────────────┐
    # │  ???: Write the matmul core function.                       │
    # │  For each of total_tiles output tiles:                      │
    # │    1. Acquire a coreC element (producer side)               │
    # │    2. Zero it                                               │
    # │    3. For each of K_div_k A/B tile pairs:                   │
    # │       acquire A and B, call matmul, release A and B         │
    # │    4. Release the coreC element                             │
    # └──────────────────────────────────────────────────────────────┘
    def matmul_core_fn(of_a, of_b, of_c, zero, matmul):
        ???

    # Core 1: relu consumer — reads C tiles from Core 0, writes D
    # ┌──────────────────────────────────────────────────────────────┐
    # │  ???: Write the relu core function.                         │
    # │  For each of total_tiles tiles:                             │
    # │    1. Acquire a coreC element (consumer side)               │
    # │    2. Acquire a memD element (producer side)                │
    # │    3. Call relu(coreC_elem, memD_elem)                      │
    # │    4. Release both                                          │
    # └──────────────────────────────────────────────────────────────┘
    def relu_core_fn(of_c, of_d, relu):
        ???

    # ------------------------------------------------------------------
    # 7. Workers
    # ------------------------------------------------------------------
    # ┌──────────────────────────────────────────────────────────────┐
    # │  ???: Create two workers:                                   │
    # │  - worker_mm using matmul_core_fn with:                     │
    # │      memA.cons(), memB.cons(), coreC.prod(),                │
    # │      zero_fn, matmul_fn                                     │
    # │  - worker_relu using relu_core_fn with:                     │
    # │      coreC.cons(), memD.prod(), relu_fn                     │
    # └──────────────────────────────────────────────────────────────┘
    worker_mm = ???
    worker_relu = ???

    # ------------------------------------------------------------------
    # 8. Runtime sequence
    # ------------------------------------------------------------------
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
    )[0]
    d_taps = TensorTiler2D.group_tiler(
        (M, N), (m, n), (1, N_div_n)
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, D_ty) as (A, B, D):
        rt.start(worker_mm, worker_relu)

        for tr in range(M_div_m):
            tg = rt.task_group()
            rt.fill(inA.prod(), A, tap=a_taps[tr], task_group=tg)
            rt.fill(inB.prod(), B, tap=b_tap, task_group=tg)
            rt.drain(outD.cons(), D, tap=d_taps[tr], task_group=tg, wait=True)
            rt.finish_task_group(tg)

    # ------------------------------------------------------------------
    # 9. Compile
    # ------------------------------------------------------------------
    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    print(matmul_relu_pipeline())
