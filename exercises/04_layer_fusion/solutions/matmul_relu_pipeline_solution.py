# matmul_relu_pipeline_solution.py
#
# Solution for Task 3: 2-stage pipeline (matmul → relu)
#
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def matmul_relu_pipeline():
    M, K, N = 4096, 256, 4096
    m, k, n = 64, 64, 64
    dtype = np.int16

    r, s, t = 4, 4, 8

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n

    total_tiles = M_div_m * N_div_n

    A_ty = np.ndarray[(M * K,), np.dtype[dtype]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype]]
    D_ty = np.ndarray[(M * N,), np.dtype[dtype]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype]]

    zero_fn = Kernel("zero_i16", "mm.cc.o", [c_ty])
    matmul_fn = Kernel("matmul_i16_i16", "mm.cc.o", [a_ty, b_ty, c_ty])
    relu_fn = Kernel("relu_fused_i16", "mm.cc.o", [c_ty, c_ty])

    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]

    # A: shim → mem tile → Core 0
    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # B: shim → mem tile → Core 0
    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    # C: Core 0 → Core 1 (direct core-to-core, double-buffered)
    coreC = ObjectFifo(c_ty, name="coreC", depth=2)

    # D: Core 1 → mem tile (un-tile via c_dims) → shim
    memD = ObjectFifo(c_ty, name="memD")
    outD = memD.cons().forward(name="outD", dims_to_stream=c_dims)

    # Core 0: matmul producer
    def matmul_core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(total_tiles):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    # Core 1: relu consumer
    def relu_core_fn(of_c, of_d, relu):
        for _ in range_(total_tiles):
            elem_c = of_c.acquire(1)
            elem_d = of_d.acquire(1)
            relu(elem_c, elem_d)
            of_c.release(1)
            of_d.release(1)

    worker_mm = Worker(
        matmul_core_fn,
        [memA.cons(), memB.cons(), coreC.prod(), zero_fn, matmul_fn],
    )
    worker_relu = Worker(
        relu_core_fn,
        [coreC.cons(), memD.prod(), relu_fn],
    )

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

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    print(matmul_relu_pipeline())
