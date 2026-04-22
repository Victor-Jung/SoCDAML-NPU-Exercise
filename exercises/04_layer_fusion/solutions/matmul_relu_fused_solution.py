# matmul_relu_fused.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  TEMPLATE — Dual-tile fused matmul + relu  (Task 2)                 │
# │                                                                     │
# │  Same structure as matmul_dual.py, but the core function must also  │
# │  apply relu in-place after the matmul K-loop.  Fill in the ???     │
# │  sections: the core function and the Worker instantiation.          │
# └──────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   python3 matmul_relu_fused.py > build/fused/aie.mlir
#
# =========================================================================
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def matmul_relu_fused():
    # ------------------------------------------------------------------
    # 1. Dimensions
    # ------------------------------------------------------------------
    M, K, N = 4096, 256, 4096
    m, k, n = 64, 64, 64
    dtype = np.int16

    r, s, t = 4, 4, 8

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n

    n_cores = 2
    tiles_per_core = (M_div_m * N_div_n) // n_cores  # 2048

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
    # 6. Core function — matmul + fused relu
    # ------------------------------------------------------------------
    def core_fn(of_a, of_b, of_c, zero, matmul, relu):
        for _ in range_(tiles_per_core):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            relu(elem_out, elem_out)
            of_c.release(1)

    # ------------------------------------------------------------------
    # 7. Workers
    # ------------------------------------------------------------------
    worker0 = Worker(
        core_fn,
        [memA0.cons(), memB0.cons(), memC0.prod(),
         zero_fn, matmul_fn, relu_fn],
    )
    worker1 = Worker(
        core_fn,
        [memA1.cons(), memB1.cons(), memC1.prod(),
         zero_fn, matmul_fn, relu_fn],
    )

    # ------------------------------------------------------------------
    # 8. Runtime sequence (same data movement as matmul_dual)
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
    with rt.sequence(A_ty, B_ty, D_ty) as (A, B, D):
        rt.start(worker0, worker1)

        rows_per_core = M_div_m // n_cores
        for pair in range(rows_per_core):
            tr0 = pair
            tr1 = pair + rows_per_core

            tg = rt.task_group()
            rt.fill(inA0.prod(), A, tap=a_taps[tr0], task_group=tg)
            rt.fill(inA1.prod(), A, tap=a_taps[tr1], task_group=tg)
            rt.fill(inB0.prod(), B, tap=b_tap, task_group=tg)
            rt.fill(inB1.prod(), B, tap=b_tap, task_group=tg)
            rt.drain(outC0.cons(), D, tap=c_taps[tr0], task_group=tg, wait=True)
            rt.drain(outC1.cons(), D, tap=c_taps[tr1], task_group=tg, wait=True)
            rt.finish_task_group(tg)

    # ------------------------------------------------------------------
    # 9. Compile
    # ------------------------------------------------------------------
    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    print(matmul_relu_fused())
