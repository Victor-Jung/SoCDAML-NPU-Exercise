# test.py
#
# Python test harness for the distribute + join add-one exercise.
# Verifies that the output tensor equals input + 1, then measures latency.
#
# Usage:
#   python3 test.py -x build/final.xclbin -i build/insts.bin \
#                   -k MLIR_AIE -i1s 32768 -os 32768
#
import numpy as np
import sys

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

WARMUP_ITERS = 5
BENCH_ITERS = 20


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

    # ── Correctness check ────────────────────────────────────────────
    in1 = iron.tensor(input_data.copy(), dtype=dtype)
    out = iron.zeros([out_volume], dtype=dtype)

    print("Running correctness check...\n")
    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, out],
        {1: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res != 0:
        print("\nFAIL!\n")
        sys.exit(res)
    print("\nPASS!\n")

    # ── Latency benchmark ────────────────────────────────────────────
    kernel_handle = DefaultNPURuntime.load(npu_opts.npu_kernel)
    total_iters = WARMUP_ITERS + BENCH_ITERS
    npu_time_total = 0
    npu_time_min = float("inf")
    npu_time_max = 0

    for i in range(total_iters):
        in1 = iron.tensor(input_data.copy(), dtype=dtype)
        out = iron.zeros([out_volume], dtype=dtype)
        ret = DefaultNPURuntime.run(kernel_handle, [in1, out])
        if i < WARMUP_ITERS:
            continue
        t_us = ret.npu_time / 1000.0  # ns → µs
        npu_time_total += t_us
        npu_time_min = min(npu_time_min, t_us)
        npu_time_max = max(npu_time_max, t_us)

    avg = npu_time_total / BENCH_ITERS
    print(f"Latency ({BENCH_ITERS} iters, {WARMUP_ITERS} warmup):")
    print(f"  avg = {avg:.1f} µs   min = {npu_time_min:.1f} µs   max = {npu_time_max:.1f} µs")


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
