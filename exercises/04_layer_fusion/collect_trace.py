# collect_trace.py
#
# Minimal script to run a relu xclbin with tracing enabled and write trace.txt.
# Used by the Makefile to collect hardware traces for Perfetto visualization.
#
# Usage:
#   python3 collect_trace.py -x <xclbin> -i <insts.bin> \
#       -k MLIR_AIE -i1s <bytes> -os <bytes> -t <trace_size>
#
import numpy as np
import sys

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

M, N = 4096, 4096


def main(opts):
    dtype = np.int16
    num_elems = int(opts.in1_size) // np.dtype(dtype).itemsize

    in_data = np.random.randint(-100, 100, num_elems, dtype=dtype)
    ref = np.maximum(in_data, np.int16(0))

    in_buf = iron.tensor(in_data.copy(), dtype=dtype)
    out_buf = iron.zeros([num_elems], dtype=dtype)

    npu_opts = test_utils.create_npu_kernel(opts)
    DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in_buf, out_buf],
        {1: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
