// test_cpp.cpp — C++ XRT test harness for single-xclbin designs
//
// Verifies D = relu(A × B) for int16, then measures latency.
// Used for: fused, pipeline designs (3 data buffers: A, B, D).
//
// Build:
//   g++ -O2 -o test_cpp test_cpp.cpp \
//       -I${XRT_DIR}/include -L${XRT_DIR}/lib -lxrt_coreutil
//
// Usage:
//   ./test_cpp <xclbin> <insts.bin> <M> <K> <N> [warmup=5] [iters=20]

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

static std::vector<uint32_t> load_instr_binary(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);
    auto size = f.tellg();
    f.seekg(0);
    if (size % 4 != 0)
        throw std::runtime_error("File size not multiple of 4");
    std::vector<uint32_t> v(size / 4);
    f.read(reinterpret_cast<char *>(v.data()), size);
    return v;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <xclbin> <insts.bin> <M> <K> <N> [warmup=5] [iters=20]\n";
        return 1;
    }

    std::string xclbin_path = argv[1];
    std::string insts_path = argv[2];
    int M = std::stoi(argv[3]);
    int K = std::stoi(argv[4]);
    int N = std::stoi(argv[5]);
    int warmup = argc > 6 ? std::stoi(argv[6]) : 5;
    int iters = argc > 7 ? std::stoi(argv[7]) : 20;

    int a_elems = M * K;
    int b_elems = K * N;
    int d_elems = M * N;
    int a_bytes = a_elems * sizeof(int16_t);
    int b_bytes = b_elems * sizeof(int16_t);
    int d_bytes = d_elems * sizeof(int16_t);

    std::cout << "test_cpp: D = relu(A × B)  M=" << M << " K=" << K
              << " N=" << N << "\n";
    std::cout << "  A: " << a_elems << " elems (" << a_bytes << " B)"
              << "  B: " << b_elems << " elems (" << b_bytes << " B)"
              << "  D: " << d_elems << " elems (" << d_bytes << " B)\n\n";

    // ── Reference computation ───────────────────────────────────────
    std::vector<int16_t> A_data(a_elems), B_data(b_elems);
    std::vector<int32_t> C_ref(d_elems, 0);
    std::vector<int16_t> D_ref(d_elems);

    // Deterministic pseudo-random input (small values to avoid overflow)
    srand(42);
    for (int i = 0; i < a_elems; i++)
        A_data[i] = (int16_t)((rand() % 5) - 2);  // -2..2
    for (int i = 0; i < b_elems; i++)
        B_data[i] = (int16_t)((rand() % 5) - 2);

    // C_ref = A @ B  (int32 accumulation, K-inner for cache locality)
    std::cout << "Computing host reference (may take a moment)... " << std::flush;
    for (int i = 0; i < M; i++)
        for (int p = 0; p < K; p++) {
            int32_t a_val = A_data[i * K + p];
            for (int j = 0; j < N; j++)
                C_ref[i * N + j] += a_val * (int32_t)B_data[p * N + j];
        }

    // D_ref = relu(C_ref), truncated to int16
    for (int i = 0; i < d_elems; i++)
        D_ref[i] = std::max((int16_t)C_ref[i], (int16_t)0);
    std::cout << "done.\n";

    // ── Load instructions ───────────────────────────────────────────
    auto instr_v = load_instr_binary(insts_path);

    // ── Init XRT ────────────────────────────────────────────────────
    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(xclbin_path);
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());

    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
        [](xrt::xclbin::kernel &k) {
            return k.get_name().rfind("MLIR_AIE", 0) == 0;
        });
    auto kernel = xrt::kernel(context, xkernel.get_name());

    // Instruction buffer
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    memcpy(bo_instr.map<void *>(), instr_v.data(),
           instr_v.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Data buffers: A→group(3), B→group(4), D→group(5)
    auto bo_a = xrt::bo(device, a_bytes, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
    auto bo_b = xrt::bo(device, b_bytes, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
    auto bo_d = xrt::bo(device, d_bytes, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

    // Fill input buffers
    memcpy(bo_a.map<void *>(), A_data.data(), a_bytes);
    memcpy(bo_b.map<void *>(), B_data.data(), b_bytes);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ── Correctness check ───────────────────────────────────────────
    std::cout << "Running correctness check... " << std::flush;

    memset(bo_d.map<void *>(), 0, d_bytes);
    bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(3, bo_instr, instr_v.size(), bo_a, bo_b, bo_d);
    run.wait();

    bo_d.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto *d_ptr = bo_d.map<int16_t *>();

    int n_err = 0;
    for (int i = 0; i < d_elems; i++) {
        if (d_ptr[i] != D_ref[i]) n_err++;
    }

    if (n_err > 0) {
        std::cout << "\nFAIL!  (" << n_err << " / " << d_elems
                  << " elements wrong)\n";
        int shown = 0;
        for (int i = 0; i < d_elems && shown < 10; i++) {
            if (d_ptr[i] != D_ref[i]) {
                int row = i / N, col = i % N;
                std::cout << "  D[" << row << "," << col << "] (flat " << i
                          << "): got " << d_ptr[i] << ", expected "
                          << D_ref[i] << "\n";
                shown++;
            }
        }
        if (n_err > 10)
            std::cout << "  ... and " << (n_err - 10) << " more\n";
        bool all_zero = true;
        for (int i = 0; i < d_elems; i++) {
            if (d_ptr[i] != 0) { all_zero = false; break; }
        }
        if (all_zero)
            std::cout << "  NOTE: output is ALL ZEROS — "
                         "design may not have executed.\n";
        return 1;
    }
    std::cout << "PASS!\n\n";

    // ── Latency benchmark ───────────────────────────────────────────
    std::vector<double> times;
    times.reserve(iters);

    for (int i = 0; i < warmup + iters; i++) {
        // Re-sync input each iteration (like the Python test)
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto run = kernel(3, bo_instr, instr_v.size(), bo_a, bo_b, bo_d);
        run.wait();
        auto t1 = std::chrono::high_resolution_clock::now();

        if (i >= warmup) {
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t1 - t0).count() / 1000.0;
            times.push_back(us);
        }
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / times.size();
    double mn = *std::min_element(times.begin(), times.end());
    double mx = *std::max_element(times.begin(), times.end());
    double sq_sum = 0;
    for (auto t : times) sq_sum += (t - avg) * (t - avg);
    double std_dev = std::sqrt(sq_sum / times.size());

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Latency (" << iters << " iters, " << warmup << " warmup):\n";
    std::cout << "  avg=" << avg << "  min=" << mn
              << "  max=" << mx << "  std=" << std_dev << " µs\n";

    return 0;
}
