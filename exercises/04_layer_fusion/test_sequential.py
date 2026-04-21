# test_sequential.py
#
# Runs matmul and relu as TWO separate xclbin loads (sequential execution).
# Demonstrates the overhead of:
#   1. DRAM round-trip for intermediate result C
#   2. xclbin reload between matmul and relu
#
# Usage:
#   python3 test_sequential.py \
#       --mm-xclbin  build/matmul_dual/final.xclbin  \
#       --mm-insts   build/matmul_dual/insts.bin      \
#       --relu-xclbin build/relu_dual/final.xclbin    \
#       --relu-insts  build/relu_dual/insts.bin
#
import argparse
import time
import numpy as np
import sys

import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils.npukernel import NPUKernel

M, K, N = 256, 256, 256
WARMUP_ITERS = 5
BENCH_ITERS = 20


def main():
    p = argparse.ArgumentParser(description="Sequential matmul + relu test")
    p.add_argument("--mm-xclbin", required=True, help="Path to matmul xclbin")
    p.add_argument("--mm-insts", required=True, help="Path to matmul insts.bin")
    p.add_argument("--relu-xclbin", required=True, help="Path to relu xclbin")
    p.add_argument("--relu-insts", required=True, help="Path to relu insts.bin")
    opts = p.parse_args()

    dtype = np.int16

    # Random inputs (small values, include negatives to exercise ReLU)
    np.random.seed(42)
    A_data = np.random.randint(-2, 3, M * K, dtype=dtype)
    B_data = np.random.randint(-2, 3, K * N, dtype=dtype)

    # Reference: D = relu(A @ B)
    ref = np.maximum(
        np.matmul(
            A_data.reshape(M, K).astype(np.int32),
            B_data.reshape(K, N).astype(np.int32),
        ).astype(dtype),
        np.int16(0),
    ).flatten()

    # ── Correctness check ────────────────────────────────────────────
    print("Running sequential matmul + relu correctness check...\n")

    # Step 1: Matmul — load xclbin, run A×B → C
    mm_kernel = NPUKernel(
        xclbin_path=opts.mm_xclbin,
        insts_path=opts.mm_insts,
        kernel_name="MLIR_AIE",
    )
    mm_handle = DefaultNPURuntime.load(mm_kernel)

    in1 = iron.tensor(A_data.copy(), dtype=dtype)
    in2 = iron.tensor(B_data.copy(), dtype=dtype)
    C_buf = iron.zeros([M * N], dtype=dtype)
    DefaultNPURuntime.run(mm_handle, [in1, in2, C_buf])

    # Step 2: ReLU — load new xclbin, run relu(C) → D
    relu_kernel = NPUKernel(
        xclbin_path=opts.relu_xclbin,
        insts_path=opts.relu_insts,
        kernel_name="MLIR_AIE",
    )
    relu_handle = DefaultNPURuntime.load(relu_kernel)

    D_buf = iron.zeros([M * N], dtype=dtype)
    DefaultNPURuntime.run(relu_handle, [C_buf, D_buf])

    # Verify
    D_buf.to("cpu")
    actual = D_buf.numpy()
    n_err = int(np.sum(actual != ref))

    if n_err > 0:
        print(f"FAIL!  ({n_err} / {len(ref)} elements wrong)\n")
        mismatches = np.where(actual != ref)[0]
        for i in mismatches[:10]:
            row, col = divmod(int(i), N)
            print(f"  D[{row},{col}] (flat {i}): got {actual[i]}, expected {ref[i]}")
        if n_err > 10:
            print(f"  ... and {n_err - 10} more")
        if np.all(actual == 0):
            print("  NOTE: output is ALL ZEROS — design may not have executed.")
        sys.exit(1)
    print("PASS!\n")

    # ── Latency benchmark ────────────────────────────────────────────
    total_iters = WARMUP_ITERS + BENCH_ITERS
    mm_total = 0.0
    relu_total = 0.0
    reload_total = 0.0
    combined_min = float("inf")
    combined_max = 0.0

    for i in range(total_iters):
        # Matmul
        in1 = iron.tensor(A_data.copy(), dtype=dtype)
        in2 = iron.tensor(B_data.copy(), dtype=dtype)
        C_buf = iron.zeros([M * N], dtype=dtype)

        mm_handle = DefaultNPURuntime.load(mm_kernel)
        ret_mm = DefaultNPURuntime.run(mm_handle, [in1, in2, C_buf])

        # Reload
        t_reload_start = time.perf_counter_ns()
        relu_handle = DefaultNPURuntime.load(relu_kernel)
        t_reload_end = time.perf_counter_ns()

        # ReLU
        D_buf = iron.zeros([M * N], dtype=dtype)
        ret_relu = DefaultNPURuntime.run(relu_handle, [C_buf, D_buf])

        if i < WARMUP_ITERS:
            continue

        mm_us = ret_mm.npu_time / 1000.0
        relu_us = ret_relu.npu_time / 1000.0
        reload_us = (t_reload_end - t_reload_start) / 1000.0

        mm_total += mm_us
        relu_total += relu_us
        reload_total += reload_us
        combined = mm_us + reload_us + relu_us
        combined_min = min(combined_min, combined)
        combined_max = max(combined_max, combined)

    mm_avg = mm_total / BENCH_ITERS
    relu_avg = relu_total / BENCH_ITERS
    reload_avg = reload_total / BENCH_ITERS
    total_avg = mm_avg + reload_avg + relu_avg

    print(f"Sequential latency ({BENCH_ITERS} iters, {WARMUP_ITERS} warmup):")
    print(f"  matmul:  {mm_avg:.1f} µs")
    print(f"  reload:  {reload_avg:.1f} µs")
    print(f"  relu:    {relu_avg:.1f} µs")
    print(f"  total:   {total_avg:.1f} µs  (min={combined_min:.1f}  max={combined_max:.1f})")


if __name__ == "__main__":
    main()
