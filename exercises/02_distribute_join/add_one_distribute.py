# add_one_distribute.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  EXERCISE                                                           │
# │                                                                     │
# │  Distribute a 64×128 int32 matrix across 4 AIE cores using a       │
# │  non-uniform row partition [8, 24, 24, 8], add 1 to every          │
# │  element in parallel, and join the results back.                    │
# │                                                                     │
# │  Your task:                                                         │
# │    1. Compute per-core element counts and cumulative offsets.       │
# │    2. Build the per-core tile types.                                │
# │    3. Write the split() and join() calls.                           │
# └──────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   python3 add_one_distribute.py -i1s 32768 -os 32768
#
# =========================================================================
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1


def add_one_distribute(in1_size, out_size):
    assert out_size == in1_size, "Output size must equal input size."

    # ------------------------------------------------------------------
    # 1. Data types & dimensions
    # ------------------------------------------------------------------
    dtype = np.int32
    n_cores = 4

    M = 64                        # matrix rows
    K = 128                       # matrix columns
    rows_per_core = [8, 24, 24, 8]  # non-uniform row partition

    # Full matrix type (what the host sends / receives)
    matrix_ty = np.ndarray[(M * K,), np.dtype[dtype]]

    # ── TODO: Compute partition parameters ────────────────────────────
    #
    # Each core processes a different number of rows.  In row-major
    # layout, core i owns a contiguous block of rows_per_core[i] × K
    # elements.  Compute:
    #
    #   elems_per_core : list[int] — element count per core
    #   of_offsets     : list[int] — cumulative element offsets
    #                    (where each core's block starts in the flat buffer)
    #   tile_types     : list      — one np.ndarray type per core,
    #                    each sized to its elems_per_core[i]
    #
    elems_per_core = ???
    of_offsets = ???
    tile_types = ???

    # ------------------------------------------------------------------
    # 2. ObjectFIFOs + Distribute / Join
    # ------------------------------------------------------------------
    of_in = ObjectFifo(matrix_ty, name="in")
    of_out = ObjectFifo(matrix_ty, name="out")

    # ── TODO: Distribute input matrix to 4 cores ─────────────────────
    #
    # Use of_in.cons().split() with of_offsets, tile_types, and names.
    #
    of_ins = ???

    # ── TODO: Join output from 4 cores back to host ──────────────────
    #
    # Use of_out.prod().join() with of_offsets, tile_types, and names.
    #
    of_outs = ???

    # ------------------------------------------------------------------
    # 3. Kernels
    # ------------------------------------------------------------------
    # Different tile types require different kernel declarations
    # (MLIR needs unique function signatures per symbol name).
    # The C file exports addOneSmall and addOneLarge.
    # One Kernel instance per unique element count (shared across cores
    # with the same size to avoid duplicate MLIR declarations).
    unique_kerns = {}
    for n in sorted(set(elems_per_core)):
        name = "addOneSmall" if n == min(elems_per_core) else "addOneLarge"
        ty = np.ndarray[(n,), np.dtype[dtype]]
        unique_kerns[n] = Kernel(name, "addOne.cc.o", [ty, ty, np.int32])

    # ------------------------------------------------------------------
    # 4. Workers
    # ------------------------------------------------------------------
    workers = []
    for i in range(n_cores):
        kern = unique_kerns[elems_per_core[i]]
        n_elems = elems_per_core[i]

        def core_fn(of_in, of_out, addOne, *, size=n_elems):
            elemIn = of_in.acquire(1)
            elemOut = of_out.acquire(1)
            addOne(elemIn, elemOut, size)
            of_in.release(1)
            of_out.release(1)

        workers.append(
            Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), kern])
        )

    # ------------------------------------------------------------------
    # 5. Runtime sequence
    # ------------------------------------------------------------------
    rt = Runtime()
    with rt.sequence(matrix_ty, matrix_ty, matrix_ty) as (a_in, b_out, _):
        rt.start(*workers)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # ------------------------------------------------------------------
    # 6. Compile
    # ------------------------------------------------------------------
    dev = NPU2Col1()
    return Program(dev, rt).resolve_program(SequentialPlacer())


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Distribute + join add-one for XDNA2 NPU"
    )
    p.add_argument(
        "-i1s", "--in1_size", type=int, default=32768,
        help="Input buffer size in bytes (default: 32768)",
    )
    p.add_argument(
        "-os", "--out_size", type=int, default=32768,
        help="Output buffer size in bytes (default: 32768)",
    )
    opts = p.parse_args(sys.argv[1:])
    mlir = add_one_distribute(opts.in1_size, opts.out_size)
    print(mlir)
