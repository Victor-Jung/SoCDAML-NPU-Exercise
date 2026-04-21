# test.py
#
# Test harness for the layer-fusion exercise (single-xclbin designs).
# Verifies D = relu(A × B) for int16, then measures latency.
#
# Used by:  make run_fused_scalar, make run_fused, make run_pipeline
#
# Usage:
#   python3 test.py -x build/<variant>/final.xclbin \
#                   -i build/<variant>/insts.bin \
#                   -k MLIR_AIE -i1s 131072 -i2s 131072 -os 131072
#
import numpy as np
import sys

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

M, K, N = 256, 256, 256
WARMUP_ITERS = 5
BENCH_ITERS = 20


def main(opts):
    dtype = np.int16

    in1_size = int(opts.in1_size)    # A bytes
    in2_size = int(opts.in2_size)    # B bytes
    out_size = int(opts.out_size)    # D bytes

    volume_a = in1_size // np.dtype(dtype).itemsize
    volume_b = in2_size // np.dtype(dtype).itemsize
    volume_d = out_size // np.dtype(dtype).itemsize

    assert volume_a == M * K, f"Expected A = {M}×{K} = {M*K} elements"
    assert volume_b == K * N, f"Expected B = {K}×{N} = {K*N} elements"
    assert volume_d == M * N, f"Expected D = {M}×{N} = {M*N} elements"

    # Random inputs (small values to avoid int16 overflow)
    np.random.seed(42)
    A_data = np.random.randint(-2, 3, volume_a, dtype=dtype)
    B_data = np.random.randint(-2, 3, volume_b, dtype=dtype)

    # Reference: D = relu(A @ B)
    ref = np.maximum(
        np.matmul(
            A_data.reshape(M, K).astype(np.int32),
            B_data.reshape(K, N).astype(np.int32),
        ).astype(dtype),
        np.int16(0),
    ).flatten()

    # ── Correctness check ────────────────────────────────────────────
    in1 = iron.tensor(A_data.copy(), dtype=dtype)
    in2 = iron.tensor(B_data.copy(), dtype=dtype)
    out = iron.zeros([volume_d], dtype=dtype)

    print("Running correctness check...\n")
    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, in2, out],
        {2: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res != 0:
        out.to("cpu")
        actual = out.numpy()
        n_err = int(np.sum(actual != ref))
        print(f"\nFAIL!  ({n_err} / {len(ref)} elements wrong)\n")
        mismatches = np.where(actual != ref)[0]
        for i in mismatches[:10]:
            row, col = divmod(int(i), N)
            print(f"  D[{row},{col}] (flat {i}): got {actual[i]}, expected {ref[i]}")
        if n_err > 10:
            print(f"  ... and {n_err - 10} more")
        if np.all(actual == 0):
            print("  NOTE: output is ALL ZEROS — design may not have executed.")
        sys.exit(res)
    print("\nPASS!\n")

    # ── Latency benchmark ────────────────────────────────────────────
    kernel_handle = DefaultNPURuntime.load(npu_opts.npu_kernel)
    total_iters = WARMUP_ITERS + BENCH_ITERS
    npu_time_total = 0
    npu_time_min = float("inf")
    npu_time_max = 0

    for i in range(total_iters):
        in1 = iron.tensor(A_data.copy(), dtype=dtype)
        in2 = iron.tensor(B_data.copy(), dtype=dtype)
        out = iron.zeros([volume_d], dtype=dtype)
        ret = DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
        if i < WARMUP_ITERS:
            continue
        t_us = ret.npu_time / 1000.0
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
