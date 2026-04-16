# add_one_double.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  EXERCISE                                                           │
# │                                                                     │
# │  This design is functionally identical to add_one_single.py.        │
# │  Your task: enable double buffering so that data transfer and       │
# │  compute can overlap.                                               │
# └──────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   python3 add_one_double.py -i1s 65536 -os 65536 [-t TRACE_SIZE]
#
# =========================================================================
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_


def add_one_double(in1_size, out_size, trace_size):
    assert out_size == in1_size, "Output size must equal input size."

    # ------------------------------------------------------------------
    # 1. Data types
    # ------------------------------------------------------------------
    dtype = np.int32

    element_bytes = np.dtype(dtype).itemsize              # 4
    total_elements = in1_size // element_bytes             # 16384 for 65536 B

    num_chunks = 8
    chunk_w = 16
    chunk_elements = total_elements // num_chunks          # 2048
    chunk_h = chunk_elements // chunk_w                    # 128

    chunk_type = np.ndarray[(chunk_h, chunk_w), np.dtype[dtype]]
    vector_type = np.ndarray[(num_chunks * chunk_h, chunk_w), np.dtype[dtype]]

    # ------------------------------------------------------------------
    # 2. ObjectFIFOs
    # ------------------------------------------------------------------
    of_in = ObjectFifo(chunk_type, name="in", depth=1)
    of_out = ObjectFifo(chunk_type, name="out", depth=1)

    # ------------------------------------------------------------------
    # 3. External kernel
    # ------------------------------------------------------------------
    add_one_fn = Kernel(
        "addOneLine",
        "addOne.cc.o",
        [chunk_type, chunk_type, np.int32],
    )

    # ------------------------------------------------------------------
    # 4. Core task
    # ------------------------------------------------------------------
    def core_fn(of_in, of_out, addOneLine):
        for _ in range_(num_chunks):
            elemIn = of_in.acquire(1)
            elemOut = of_out.acquire(1)
            addOneLine(elemIn, elemOut, chunk_elements)
            of_in.release(1)
            of_out.release(1)

    # ------------------------------------------------------------------
    # 5. Worker
    # ------------------------------------------------------------------
    enable_trace = 1 if trace_size > 0 else 0
    my_worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), add_one_fn],
        trace=enable_trace,
    )

    # ------------------------------------------------------------------
    # 6. Runtime sequence
    # ------------------------------------------------------------------
    rt = Runtime()
    with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
        rt.enable_trace(trace_size)
        rt.start(my_worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # ------------------------------------------------------------------
    # 7. Compile
    # ------------------------------------------------------------------
    dev = NPU2Col1()
    return Program(dev, rt).resolve_program(SequentialPlacer())


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Double-buffer add-one for XDNA2 NPU"
    )
    p.add_argument(
        "-i1s", "--in1_size", type=int, default=65536,
        help="Input buffer size in bytes (default: 65536)",
    )
    p.add_argument(
        "-os", "--out_size", type=int, default=65536,
        help="Output buffer size in bytes (default: 65536)",
    )
    p.add_argument(
        "-t", "--trace_size", type=int, default=0,
        help="Trace buffer size in bytes (0 = no trace, default: 0)",
    )
    opts = p.parse_args(sys.argv[1:])
    mlir = add_one_double(opts.in1_size, opts.out_size, opts.trace_size)
    print(mlir)
