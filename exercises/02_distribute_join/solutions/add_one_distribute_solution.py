# add_one_distribute_solution.py
#
# SOLUTION -- Distribute + join add-one across 4 AIE cores
#             with non-uniform row partition [8, 24, 24, 8].
#
# Usage:
#   python3 add_one_distribute_solution.py -i1s 32768 -os 32768
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
    assert sum(rows_per_core) == M

    # Per-core element counts and cumulative offsets
    elems_per_core = [r * K for r in rows_per_core]
    # = [1024, 3072, 3072, 1024]

    of_offsets = []
    acc = 0
    for n in elems_per_core:
        of_offsets.append(acc)
        acc += n
    # = [0, 1024, 4096, 7168]

    # Full matrix type (what the host sends / receives)
    matrix_ty = np.ndarray[(M * K,), np.dtype[dtype]]

    # Per-core tile types (different sizes for non-uniform partition)
    tile_types = [np.ndarray[(n,), np.dtype[dtype]] for n in elems_per_core]

    # ------------------------------------------------------------------
    # 2. ObjectFIFOs + Distribute / Join
    # ------------------------------------------------------------------
    of_in = ObjectFifo(matrix_ty, name="in")
    of_out = ObjectFifo(matrix_ty, name="out")

    of_ins = of_in.cons().split(
        of_offsets,
        obj_types=tile_types,
        names=[f"in{i}" for i in range(n_cores)],
    )

    of_outs = of_out.prod().join(
        of_offsets,
        obj_types=tile_types,
        names=[f"out{i}" for i in range(n_cores)],
    )

    # ------------------------------------------------------------------
    # 3. Kernels
    # ------------------------------------------------------------------
    # Different tile types require different kernel declarations
    # (MLIR needs unique function signatures per symbol name).
    # The C file exports addOneSmall and addOneLarge.
    # Create one Kernel instance per unique element count (shared
    # across cores with the same size to avoid duplicate func.func).
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
        description="Distribute + join add-one for XDNA2 NPU (SOLUTION)"
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
