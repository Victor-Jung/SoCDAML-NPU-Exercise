# test.py
#
# Python test harness for the distribute + join add-one exercise.
# Verifies that the output tensor equals input + 1.
#
# Usage:
#   python3 test.py -x build/final.xclbin -i build/insts.bin \
#                   -k MLIR_AIE -i1s 65536 -os 65536
#
import numpy as np
import sys

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime


def main(opts):
    in1_size = int(opts.in1_size)
    out_size = int(opts.out_size)

    dtype = np.int32

    volume = in1_size // np.dtype(dtype).itemsize
    out_volume = out_size // np.dtype(dtype).itemsize
    assert out_size == in1_size, "Output size must equal input size."

    # Reference: for an add-one kernel, output == input + 1
    input_data = np.arange(0, volume, dtype=dtype)
    ref = input_data + np.int32(1)
    in1 = iron.tensor(input_data.copy(), dtype=dtype)
    out = iron.zeros([out_volume], dtype=dtype)

    print("Running...\n")
    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, out],
        {1: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
