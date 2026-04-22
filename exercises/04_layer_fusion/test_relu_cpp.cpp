// test_relu_cpp.cpp — C++ XRT test harness for relu-only xclbin
//
// Verifies out = relu(in), then measures latency.
// Used for: relu_dual_scalar, relu_dual designs (2 data buffers: in, out).
//
// Build:
//   g++ -O2 -o test_relu_cpp test_relu_cpp.cpp \
//       -I${XRT_DIR}/include -L${XRT_DIR}/lib -lxrt_coreutil
//
// Usage:
//   ./test_relu_cpp <xclbin> <insts.bin> <num_elements> [warmup=5] [iters=20]

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
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <xclbin> <insts.bin> <num_elements> [warmup=5] [iters=20]\n";
        return 1;
    }

    std::string xclbin_path = argv[1];
    std::string insts_path = argv[2];
    int num_elements = std::stoi(argv[3]);
    int warmup = argc > 4 ? std::stoi(argv[4]) : 5;
    int iters = argc > 5 ? std::stoi(argv[5]) : 20;

    int buf_size = num_elements * sizeof(int16_t);

    std::cout << "test_relu_cpp: out = relu(in)  elements=" << num_elements
              << "  bytes=" << buf_size << "\n\n";

    // ── Reference data ──────────────────────────────────────────────
    std::vector<int16_t> in_data(num_elements);
    std::vector<int16_t> ref(num_elements);

    srand(42);
    for (int i = 0; i < num_elements; i++) {
        in_data[i] = (int16_t)((rand() % 200) - 100);  // -100..99
        ref[i] = std::max(in_data[i], (int16_t)0);
    }

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

    // Data buffers: in→group(3), out→group(4)
    auto bo_in = xrt::bo(device, buf_size, XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(3));
    auto bo_out = xrt::bo(device, buf_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(4));

    // Fill input
    memcpy(bo_in.map<void *>(), in_data.data(), buf_size);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ── Correctness check ───────────────────────────────────────────
    std::cout << "Running correctness check... " << std::flush;

    memset(bo_out.map<void *>(), 0, buf_size);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(3, bo_instr, instr_v.size(), bo_in, bo_out);
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto *out_ptr = bo_out.map<int16_t *>();

    int n_err = 0;
    for (int i = 0; i < num_elements; i++) {
        if (out_ptr[i] != ref[i]) n_err++;
    }

    if (n_err > 0) {
        std::cout << "\nFAIL!  (" << n_err << " / " << num_elements
                  << " elements wrong)\n";
        int shown = 0;
        for (int i = 0; i < num_elements && shown < 10; i++) {
            if (out_ptr[i] != ref[i]) {
                std::cout << "  [" << i << "]: got " << out_ptr[i]
                          << ", expected " << ref[i] << "\n";
                shown++;
            }
        }
        if (n_err > 10)
            std::cout << "  ... and " << (n_err - 10) << " more\n";
        bool all_zero = true;
        for (int i = 0; i < num_elements; i++) {
            if (out_ptr[i] != 0) { all_zero = false; break; }
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
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto run = kernel(3, bo_instr, instr_v.size(), bo_in, bo_out);
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
