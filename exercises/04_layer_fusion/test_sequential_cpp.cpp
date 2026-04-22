// test_sequential_cpp.cpp — C++ XRT test for sequential matmul → relu
//
// Loads the matmul xclbin, runs A×B → C, then loads the relu xclbin
// and runs relu(C) → D.  Reports the total time including the xclbin
// reload overhead.
//
// Build:
//   g++ -O2 -o test_sequential_cpp test_sequential_cpp.cpp \
//       -I${XRT_DIR}/include -L${XRT_DIR}/lib -lxrt_coreutil
//
// Usage:
//   ./test_sequential_cpp <mm_xclbin> <mm_insts> <relu_xclbin> <relu_insts> \
//       <M> <K> <N> [warmup=5] [iters=20]

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

static xrt::kernel find_mlir_kernel(xrt::hw_context &ctx,
                                     xrt::xclbin &xclbin) {
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
        [](xrt::xclbin::kernel &k) {
            return k.get_name().rfind("MLIR_AIE", 0) == 0;
        });
    return xrt::kernel(ctx, xkernel.get_name());
}

int main(int argc, char *argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <mm_xclbin> <mm_insts> <relu_xclbin> <relu_insts>"
                     " <M> <K> <N> [warmup=5] [iters=20]\n";
        return 1;
    }

    std::string mm_xclbin_path = argv[1];
    std::string mm_insts_path = argv[2];
    std::string relu_xclbin_path = argv[3];
    std::string relu_insts_path = argv[4];
    int M = std::stoi(argv[5]);
    int K = std::stoi(argv[6]);
    int N = std::stoi(argv[7]);
    int warmup = argc > 8 ? std::stoi(argv[8]) : 5;
    int iters = argc > 9 ? std::stoi(argv[9]) : 20;

    int a_elems = M * K;
    int b_elems = K * N;
    int c_elems = M * N;   // matmul output / relu input
    int d_elems = M * N;   // relu output
    int a_bytes = a_elems * sizeof(int16_t);
    int b_bytes = b_elems * sizeof(int16_t);
    int c_bytes = c_elems * sizeof(int16_t);
    int d_bytes = d_elems * sizeof(int16_t);

    std::cout << "test_sequential_cpp: D = relu(A × B)\n"
              << "  M=" << M << " K=" << K << " N=" << N << "\n"
              << "  A: " << a_bytes << " B  B: " << b_bytes
              << " B  C/D: " << c_bytes << " B\n\n";

    // ── Reference computation ───────────────────────────────────────
    std::vector<int16_t> A_data(a_elems), B_data(b_elems);
    std::vector<int32_t> C_ref(c_elems, 0);
    std::vector<int16_t> D_ref(d_elems);

    srand(42);
    for (int i = 0; i < a_elems; i++)
        A_data[i] = (int16_t)((rand() % 5) - 2);
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
    for (int i = 0; i < d_elems; i++)
        D_ref[i] = std::max((int16_t)C_ref[i], (int16_t)0);
    std::cout << "done.\n\n";

    // ── Load instruction files ──────────────────────────────────────
    auto mm_instr = load_instr_binary(mm_insts_path);
    auto relu_instr = load_instr_binary(relu_insts_path);

    // ── Open device ─────────────────────────────────────────────────
    auto device = xrt::device(0);

    // ── Helper: run matmul then relu ────────────────────────────────
    // Each call loads both xclbins sequentially (like the real flow).

    auto run_sequential = [&](int16_t *d_out,
                              double &mm_us, double &reload_us,
                              double &relu_us) {
        using clk = std::chrono::high_resolution_clock;
        namespace ch = std::chrono;

        // Step 1: Matmul  ─ load xclbin, allocate BOs, run
        auto mm_xcl = xrt::xclbin(mm_xclbin_path);
        device.register_xclbin(mm_xcl);
        xrt::hw_context mm_ctx(device, mm_xcl.get_uuid());
        auto mm_kernel = find_mlir_kernel(mm_ctx, mm_xcl);

        auto mm_bo_instr = xrt::bo(device,
            mm_instr.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, mm_kernel.group_id(1));
        memcpy(mm_bo_instr.map<void *>(), mm_instr.data(),
               mm_instr.size() * sizeof(uint32_t));
        mm_bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Matmul: A→group(3), B→group(4), C→group(5)
        auto bo_a = xrt::bo(device, a_bytes, XRT_BO_FLAGS_HOST_ONLY,
                            mm_kernel.group_id(3));
        auto bo_b = xrt::bo(device, b_bytes, XRT_BO_FLAGS_HOST_ONLY,
                            mm_kernel.group_id(4));
        auto bo_c = xrt::bo(device, c_bytes, XRT_BO_FLAGS_HOST_ONLY,
                            mm_kernel.group_id(5));

        memcpy(bo_a.map<void *>(), A_data.data(), a_bytes);
        memcpy(bo_b.map<void *>(), B_data.data(), b_bytes);
        memset(bo_c.map<void *>(), 0, c_bytes);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto t_mm0 = clk::now();
        auto mm_run = mm_kernel(3, mm_bo_instr, mm_instr.size(),
                                bo_a, bo_b, bo_c);
        mm_run.wait();
        auto t_mm1 = clk::now();
        mm_us = ch::duration_cast<ch::nanoseconds>(t_mm1 - t_mm0).count()
                / 1000.0;

        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Step 2: Reload relu xclbin
        auto t_rl0 = clk::now();
        auto relu_xcl = xrt::xclbin(relu_xclbin_path);
        device.register_xclbin(relu_xcl);
        xrt::hw_context relu_ctx(device, relu_xcl.get_uuid());
        auto relu_kernel = find_mlir_kernel(relu_ctx, relu_xcl);

        auto relu_bo_instr = xrt::bo(device,
            relu_instr.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, relu_kernel.group_id(1));
        memcpy(relu_bo_instr.map<void *>(), relu_instr.data(),
               relu_instr.size() * sizeof(uint32_t));
        relu_bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t_rl1 = clk::now();
        reload_us = ch::duration_cast<ch::nanoseconds>(t_rl1 - t_rl0).count()
                    / 1000.0;

        // ReLU: in→group(3), out→group(4)
        auto bo_relu_in = xrt::bo(device, c_bytes, XRT_BO_FLAGS_HOST_ONLY,
                                  relu_kernel.group_id(3));
        auto bo_relu_out = xrt::bo(device, d_bytes, XRT_BO_FLAGS_HOST_ONLY,
                                   relu_kernel.group_id(4));

        // Copy matmul output → relu input
        memcpy(bo_relu_in.map<void *>(), bo_c.map<void *>(), c_bytes);
        memset(bo_relu_out.map<void *>(), 0, d_bytes);
        bo_relu_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_relu_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto t_relu0 = clk::now();
        auto relu_run = relu_kernel(3, relu_bo_instr, relu_instr.size(),
                                    bo_relu_in, bo_relu_out);
        relu_run.wait();
        auto t_relu1 = clk::now();
        relu_us = ch::duration_cast<ch::nanoseconds>(t_relu1 - t_relu0).count()
                  / 1000.0;

        bo_relu_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (d_out)
            memcpy(d_out, bo_relu_out.map<void *>(), d_bytes);
    };

    // ── Correctness check ───────────────────────────────────────────
    std::cout << "Running correctness check... " << std::flush;

    std::vector<int16_t> D_actual(d_elems);
    double mm_t, rl_t, relu_t;
    run_sequential(D_actual.data(), mm_t, rl_t, relu_t);

    int n_err = 0;
    for (int i = 0; i < d_elems; i++) {
        if (D_actual[i] != D_ref[i]) n_err++;
    }

    if (n_err > 0) {
        std::cout << "\nFAIL!  (" << n_err << " / " << d_elems
                  << " elements wrong)\n";
        int shown = 0;
        for (int i = 0; i < d_elems && shown < 10; i++) {
            if (D_actual[i] != D_ref[i]) {
                int row = i / N, col = i % N;
                std::cout << "  D[" << row << "," << col << "] (flat " << i
                          << "): got " << D_actual[i] << ", expected "
                          << D_ref[i] << "\n";
                shown++;
            }
        }
        if (n_err > 10)
            std::cout << "  ... and " << (n_err - 10) << " more\n";
        return 1;
    }
    std::cout << "PASS!\n\n";

    // ── Latency benchmark ───────────────────────────────────────────
    std::vector<double> mm_times, reload_times, relu_times, total_times;
    mm_times.reserve(iters);
    reload_times.reserve(iters);
    relu_times.reserve(iters);
    total_times.reserve(iters);

    for (int i = 0; i < warmup + iters; i++) {
        double mm_us, reload_us, relu_us;
        run_sequential(nullptr, mm_us, reload_us, relu_us);
        if (i >= warmup) {
            mm_times.push_back(mm_us);
            reload_times.push_back(reload_us);
            relu_times.push_back(relu_us);
            total_times.push_back(mm_us + reload_us + relu_us);
        }
    }

    auto stats = [](const std::vector<double> &v) {
        double s = std::accumulate(v.begin(), v.end(), 0.0);
        double avg = s / v.size();
        double mn = *std::min_element(v.begin(), v.end());
        double mx = *std::max_element(v.begin(), v.end());
        return std::make_tuple(avg, mn, mx);
    };

    auto [mm_avg, mm_min, mm_max] = stats(mm_times);
    auto [rl_avg, rl_min, rl_max] = stats(reload_times);
    auto [re_avg, re_min, re_max] = stats(relu_times);
    auto [tot_avg, tot_min, tot_max] = stats(total_times);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Latency (" << iters << " iters, " << warmup << " warmup):\n";
    std::cout << "  matmul:  avg=" << mm_avg << "  min=" << mm_min
              << "  max=" << mm_max << " µs\n";
    std::cout << "  reload:  avg=" << rl_avg << "  min=" << rl_min
              << "  max=" << rl_max << " µs\n";
    std::cout << "  relu:    avg=" << re_avg << "  min=" << re_min
              << "  max=" << re_max << " µs\n";
    std::cout << "  total:   avg=" << tot_avg << "  min=" << tot_min
              << "  max=" << tot_max << " µs\n";

    return 0;
}
