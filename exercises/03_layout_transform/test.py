# test.py
#
# Python test harness for the layout-transform matmul exercise.
# Verifies C = A × B (int16), then measures latency.
#
# Usage:
#   python3 test.py -x build/scalar/final.xclbin -i build/scalar/insts.bin \
#                   -k MLIR_AIE -i1s 32768 -i2s 32768 -os 32768
#
import numpy as np
import sys

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

M, K, N = 128, 128, 128
WARMUP_ITERS = 5
BENCH_ITERS = 20


def main(opts):
    dtype = np.int16

    in1_size = int(opts.in1_size)    # A bytes
    in2_size = int(opts.in2_size)    # B bytes
    out_size = int(opts.out_size)    # C bytes

    volume_a = in1_size // np.dtype(dtype).itemsize
    volume_b = in2_size // np.dtype(dtype).itemsize
    volume_c = out_size // np.dtype(dtype).itemsize

    assert volume_a == M * K, f"Expected A = {M}×{K} = {M*K} elements"
    assert volume_b == K * N, f"Expected B = {K}×{N} = {K*N} elements"
    assert volume_c == M * N, f"Expected C = {M}×{N} = {M*N} elements"

    # Random inputs (small values to avoid int16 overflow:
    # max dot-product = 9 × 9 × 128 = 10368 < 32767)
    np.random.seed(42)
    A_data = np.random.randint(0, 3, volume_a, dtype=dtype)
    B_data = np.random.randint(0, 3, volume_b, dtype=dtype)

    # Reference (compute in int32 then truncate to int16)
    ref = np.matmul(
        A_data.reshape(M, K).astype(np.int32),
        B_data.reshape(K, N).astype(np.int32),
    ).astype(dtype).flatten()

    # ── Correctness check ────────────────────────────────────────────
    in1 = iron.tensor(A_data.copy(), dtype=dtype)
    in2 = iron.tensor(B_data.copy(), dtype=dtype)
    out = iron.zeros([volume_c], dtype=dtype)

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
        act2d = actual.reshape(M, N)
        ref2d = ref.reshape(M, N)
        diff2d = (act2d != ref2d)
        n_err = int(np.sum(diff2d))
        print(f"\nFAIL!  ({n_err} / {M*N} elements wrong)\n")

        # ── Per-tile breakdown ──
        m, n = 64, 64
        for tr in range(M // m):
            for tc in range(N // n):
                tile = diff2d[tr*m:(tr+1)*m, tc*n:(tc+1)*n]
                te = int(np.sum(tile))
                print(f"  Tile C({tr},{tc})  rows [{tr*m}:{(tr+1)*m}) cols [{tc*n}:{(tc+1)*n})  errors: {te}/{m*n}")

        # ── Row / column error histograms ──
        row_errs = np.sum(diff2d, axis=1)
        err_rows = np.where(row_errs > 0)[0]
        print(f"\n  Rows with errors: {err_rows[0]}..{err_rows[-1]} ({len(err_rows)} rows)"
              if len(err_rows) else "\n  No row errors")
        if len(err_rows):
            print(f"  Errors per error-row (min/max): {int(row_errs[err_rows].min())} / {int(row_errs[err_rows].max())}")

        col_errs = np.sum(diff2d, axis=0)
        err_cols = np.where(col_errs > 0)[0]
        print(f"  Cols with errors: {list(err_cols[:30])}{'...' if len(err_cols)>30 else ''}")
        if len(err_cols):
            col_mod = [int(c) % 4 for c in err_cols]
            print(f"  Error col mod-4 pattern: {sorted(set(col_mod))}")
            col_mod8 = [int(c) % 8 for c in err_cols]
            print(f"  Error col mod-8 pattern: {sorted(set(col_mod8))}")

        # ── Value analysis ──
        mismatches = np.argwhere(diff2d)
        diffs = np.array([int(act2d[r, c]) - int(ref2d[r, c]) for r, c in mismatches])
        ratios = np.array([float(act2d[r, c]) / float(ref2d[r, c])
                           if ref2d[r, c] != 0 else float('inf')
                           for r, c in mismatches])
        print(f"\n  Value diffs (got-exp):  min={diffs.min()}  max={diffs.max()}  "
              f"mean={diffs.mean():.1f}  median={np.median(diffs):.1f}")
        finite = ratios[np.isfinite(ratios)]
        if len(finite):
            print(f"  Value ratios (got/exp): min={finite.min():.3f}  max={finite.max():.3f}  "
                  f"mean={finite.mean():.3f}")

        # ── Sample mismatches: first 20 ──
        print(f"\n  First 20 mismatches:")
        for idx, (r, c) in enumerate(mismatches[:20]):
            r, c = int(r), int(c)
            print(f"    C[{r:3d},{c:3d}]: got {act2d[r,c]:6d}  exp {ref2d[r,c]:6d}  "
                  f"diff {int(act2d[r,c])-int(ref2d[r,c]):+6d}")

        # ── Dump a small 8×8 patch around first error ──
        r0, c0 = int(mismatches[0][0]), int(mismatches[0][1])
        r0 = max(0, r0 - 2)
        c0 = max(0, (c0 // 8) * 8)  # align to 8-col boundary
        print(f"\n  Actual  C[{r0}:{r0+8}, {c0}:{c0+8}]:")
        for rr in range(r0, min(r0+8, M)):
            vals = " ".join(f"{act2d[rr,cc]:5d}" for cc in range(c0, min(c0+8, N)))
            print(f"    row {rr:3d}: {vals}")
        print(f"  Expected C[{r0}:{r0+8}, {c0}:{c0+8}]:")
        for rr in range(r0, min(r0+8, M)):
            vals = " ".join(f"{ref2d[rr,cc]:5d}" for cc in range(c0, min(c0+8, N)))
            print(f"    row {rr:3d}: {vals}")

        # ── Check if output is all zeros ──
        if np.all(actual == 0):
            print("\n  NOTE: output is ALL ZEROS — design may not have executed.")

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
        out = iron.zeros([volume_c], dtype=dtype)
        ret = DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
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
