# relu_dual.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  GIVEN — Dual-tile parallel vectorized ReLU                         │
# │                                                                     │
# │  Two AIE cores each apply element-wise ReLU (max(x, 0)) to half    │
# │  of a 4096×4096 int16 matrix.  No DMA layout transform needed —     │
# │  ReLU is element-wise so data order doesn't matter.                 │
# └──────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   python3 relu_dual.py > build/relu_dual/aie.mlir
#
# =========================================================================
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.device.tile import Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def relu_dual():
    # ------------------------------------------------------------------
    # 1. Dimensions
    # ------------------------------------------------------------------
    M, N = 4096, 4096
    m, n = 64, 64
    dtype = np.int16

    M_div_m = M // m                    # 64
    N_div_n = N // n                    # 64
    total_tiles = M_div_m * N_div_n     # 4096

    n_cores = 2
    tiles_per_core = total_tiles // n_cores  # 2048

    # ------------------------------------------------------------------
    # 2. Types
    # ------------------------------------------------------------------
    D_ty = np.ndarray[(M * N,), np.dtype[dtype]]
    chunk_ty = np.ndarray[(m, n), np.dtype[dtype]]

    # ------------------------------------------------------------------
    # 3. Kernel
    # ------------------------------------------------------------------
    relu_fn = Kernel("relu_i16", "mm.cc.o", [chunk_ty, chunk_ty])

    # ------------------------------------------------------------------
    # Trace support: pass trace_size as first CLI argument
    # ------------------------------------------------------------------
    trace_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    enable_trace = 1 if trace_size > 0 else 0

    # ------------------------------------------------------------------
    # 4. ObjectFIFOs — one set per core, pinned to separate columns
    # ------------------------------------------------------------------
    # Core 0 — column 0
    inD0 = ObjectFifo(chunk_ty, name="inD0")
    memD0_in = inD0.cons().forward(name="memD0_in", placement=Tile(0, 1))
    outD0 = ObjectFifo(chunk_ty, name="outD0")
    memD0_out = outD0.cons().forward(name="memD0_out", placement=Tile(0, 1))

    # Core 1 — column 1
    inD1 = ObjectFifo(chunk_ty, name="inD1")
    memD1_in = inD1.cons().forward(name="memD1_in", placement=Tile(1, 1))
    outD1 = ObjectFifo(chunk_ty, name="outD1")
    memD1_out = outD1.cons().forward(name="memD1_out", placement=Tile(1, 1))

    # ------------------------------------------------------------------
    # 5. Core function
    # ------------------------------------------------------------------
    def core_fn(of_in, of_out, relu):
        for _ in range_(tiles_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            relu(elem_in, elem_out)
            of_in.release(1)
            of_out.release(1)

    # ------------------------------------------------------------------
    # 6. Workers — pinned to separate columns
    # ------------------------------------------------------------------
    worker0 = Worker(
        core_fn,
        [memD0_in.cons(), outD0.prod(), relu_fn],
        placement=Tile(0, 2),
        trace=enable_trace,
    )
    worker1 = Worker(
        core_fn,
        [memD1_in.cons(), outD1.prod(), relu_fn],
        placement=Tile(1, 2),
        trace=enable_trace,
    )

    # ------------------------------------------------------------------
    # 7. Runtime sequence
    # ------------------------------------------------------------------
    d_taps = TensorTiler2D.group_tiler(
        (M, N), (m, n), (1, N_div_n)
    )

    rt = Runtime()
    with rt.sequence(D_ty, D_ty) as (D_in, D_out):
        if trace_size > 0:
            rt.enable_trace(trace_size)
        rt.start(worker0, worker1)

        rows_per_core = M_div_m // n_cores  # 2
        for pair in range(rows_per_core):
            tr0 = pair
            tr1 = pair + rows_per_core

            tg = rt.task_group()
            rt.fill(inD0.prod(), D_in, tap=d_taps[tr0], task_group=tg,
                    placement=Tile(0, 0))
            rt.fill(inD1.prod(), D_in, tap=d_taps[tr1], task_group=tg,
                    placement=Tile(1, 0))
            rt.drain(memD0_out.cons(), D_out, tap=d_taps[tr0], task_group=tg,
                     wait=True, placement=Tile(0, 0))
            rt.drain(memD1_out.cons(), D_out, tap=d_taps[tr1], task_group=tg,
                     wait=True, placement=Tile(1, 0))
            rt.finish_task_group(tg)

    # ------------------------------------------------------------------
    # 8. Compile
    # ------------------------------------------------------------------
    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    print(relu_dual())
