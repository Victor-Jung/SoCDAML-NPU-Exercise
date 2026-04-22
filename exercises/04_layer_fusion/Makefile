##===- Makefile -- Layer-fusion exercise ------------------------------------===##
#
# Task 1 — Implement ReLU kernels:
#   build_matmul_dual         Compile the given 2-core matmul design
#   build_relu_dual_scalar    Compile relu design using relu_scalar_i16 (trace-enabled)
#   build_relu_dual           Compile relu design using relu_i16 (trace-enabled)
#   run_sequential_scalar     Run matmul → scalar relu + verify + timing
#   run_sequential            Run matmul → vectorized relu + verify + timing
#
# Task 2 — Kernel fusion:
#   build_fused               Compile fused matmul+relu design
#   run_fused                 Run + verify fused design
#
# Task 3 — Pipeline:
#   build_pipeline            Compile 2-stage pipeline design
#   run_pipeline              Run + verify pipeline design
#
#   clean                     Remove all build artifacts
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# ========================= Tool paths ====================================
MLIR_AIE_DIR ?= /scratch/jungvi/mlir-aie
include ${MLIR_AIE_DIR}/programming_examples/makefile-common

XRT_DIR ?= ${MLIR_AIE_DIR}/xdna-driver/xrt/build/Release/opt/xilinx/xrt
export PATH := ${XRT_DIR}/bin:${PATH}

PEANO_INSTALL_DIR ?= $(shell python3 -c "import aie.utils.config; print(aie.utils.config.peano_install_dir())" 2>/dev/null)
ifeq ($(PEANO_INSTALL_DIR),)
$(error PEANO_INSTALL_DIR not set and auto-detection failed. Set it or install llvm-aie.)
endif

MLIR_AIE_INSTALL_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())" 2>/dev/null)
AIE_INCLUDE_FLAGS = -I ${MLIR_AIE_INSTALL_DIR}/include

# ========================= Configuration =================================
devicename = npu2
# Matrix dimensions: M=4096, K=256, N=4096
# A = M×K int16, B = K×N int16, C/D = M×N int16
M = 4096
K = 256
N = 4096
i1_size = 2097152
i2_size = 2097152
o_size  = 33554432
relu_elems = 16777216
trace_size = 16384

# C++ compiler and XRT
CXX = g++
CXXFLAGS = -O2 -I${XRT_DIR}/include
LDFLAGS = -L${XRT_DIR}/lib -lxrt_coreutil

# Trace utilities
TRACE_PARSE   = python3 ${MLIR_AIE_DIR}/python/utils/trace/parse.py
TRACE_SUMMARY = python3 ${MLIR_AIE_DIR}/python/utils/trace/get_trace_summary.py

# Override to use solution kernel file
KERNEL_SRC ?= ${srcdir}/mm.cc

# Override to build fused/pipeline with the solution
FUSED_DESIGN ?= ${srcdir}/matmul_relu_fused.py
PIPELINE_DESIGN ?= ${srcdir}/matmul_relu_pipeline.py

.PHONY: all clean \
        build_matmul_dual build_relu_dual_scalar build_relu_dual \
        run_sequential_scalar run_sequential \
        build_fused run_fused \
        build_pipeline run_pipeline

all: build_matmul_dual build_relu_dual

# ========================= C++ test harness compilation ==================
build/test_relu_cpp: ${srcdir}/test_relu_cpp.cpp
	mkdir -p ${@D}
	${CXX} ${CXXFLAGS} -o $@ $< ${LDFLAGS}

build/test_cpp: ${srcdir}/test_cpp.cpp
	mkdir -p ${@D}
	${CXX} ${CXXFLAGS} -o $@ $< ${LDFLAGS}

build/test_sequential_cpp: ${srcdir}/test_sequential_cpp.cpp
	mkdir -p ${@D}
	${CXX} ${CXXFLAGS} -o $@ $< ${LDFLAGS}

# ========================= Kernel compilation ============================
build/mm.cc.o: $(KERNEL_SRC)
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${AIE_INCLUDE_FLAGS} \
		-c $(abspath $<) -o ${@F}

# ========================= Matmul dual design (given) ====================
build/matmul_dual/aie.mlir: ${srcdir}/matmul_dual.py
	mkdir -p ${@D}
	python3 $< > $@

build/matmul_dual/final.xclbin: build/matmul_dual/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/matmul_dual/mm.cc.o
	cd build/matmul_dual && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

build_matmul_dual: build/matmul_dual/final.xclbin

# ========================= ReLU dual scalar design (Task 1a) =============
# Built with tracing enabled so run_sequential_scalar can collect traces.
build/relu_dual_scalar/aie.mlir: ${srcdir}/relu_dual_scalar.py
	mkdir -p ${@D}
	python3 $< ${trace_size} > $@

build/relu_dual_scalar/final.xclbin: build/relu_dual_scalar/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/relu_dual_scalar/mm.cc.o
	cd build/relu_dual_scalar && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

build_relu_dual_scalar: build/relu_dual_scalar/final.xclbin

# ========================= ReLU dual vectorized design (Task 1b) =========
build/relu_dual/aie.mlir: ${srcdir}/relu_dual.py
	mkdir -p ${@D}
	python3 $< ${trace_size} > $@

build/relu_dual/final.xclbin: build/relu_dual/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/relu_dual/mm.cc.o
	cd build/relu_dual && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

build_relu_dual: build/relu_dual/final.xclbin

# ========================= Sequential runs (Task 1) ======================
# Each target: runs the C++ test harness for correctness + timing, then
# collects a relu trace and parses it for Perfetto visualization.

run_sequential_scalar: build_matmul_dual build_relu_dual_scalar build/test_sequential_cpp build/test_relu_cpp
	LD_LIBRARY_PATH=${XRT_DIR}/lib:$$LD_LIBRARY_PATH \
	build/test_sequential_cpp \
		build/matmul_dual/final.xclbin  build/matmul_dual/insts.bin \
		build/relu_dual_scalar/final.xclbin build/relu_dual_scalar/insts.bin \
		${M} ${K} ${N}
	LD_LIBRARY_PATH=${XRT_DIR}/lib:$$LD_LIBRARY_PATH \
	build/test_relu_cpp \
		build/relu_dual_scalar/final.xclbin build/relu_dual_scalar/insts.bin \
		${relu_elems}
	python3 ${srcdir}/collect_trace.py \
		-x build/relu_dual_scalar/final.xclbin -i build/relu_dual_scalar/insts.bin \
		-k MLIR_AIE -i1s ${o_size} -os ${o_size} -t ${trace_size}
	${TRACE_PARSE} --input trace.txt \
		--mlir build/relu_dual_scalar/aie.mlir.prj/input_with_addresses.mlir \
		--output trace_relu_scalar.json
	${TRACE_SUMMARY} --input trace_relu_scalar.json

run_sequential: build_matmul_dual build_relu_dual build/test_sequential_cpp build/test_relu_cpp
	LD_LIBRARY_PATH=${XRT_DIR}/lib:$$LD_LIBRARY_PATH \
	build/test_sequential_cpp \
		build/matmul_dual/final.xclbin  build/matmul_dual/insts.bin \
		build/relu_dual/final.xclbin    build/relu_dual/insts.bin \
		${M} ${K} ${N}
	LD_LIBRARY_PATH=${XRT_DIR}/lib:$$LD_LIBRARY_PATH \
	build/test_relu_cpp \
		build/relu_dual/final.xclbin build/relu_dual/insts.bin \
		${relu_elems}
	python3 ${srcdir}/collect_trace.py \
		-x build/relu_dual/final.xclbin -i build/relu_dual/insts.bin \
		-k MLIR_AIE -i1s ${o_size} -os ${o_size} -t ${trace_size}
	${TRACE_PARSE} --input trace.txt \
		--mlir build/relu_dual/aie.mlir.prj/input_with_addresses.mlir \
		--output trace_relu.json
	${TRACE_SUMMARY} --input trace_relu.json

# ========================= Fused design (Task 2) =========================
build/fused/aie.mlir: $(FUSED_DESIGN)
	mkdir -p ${@D}
	python3 $< > $@

build/fused/final.xclbin: build/fused/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/fused/mm.cc.o
	cd build/fused && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

build_fused: build/fused/final.xclbin

run_fused: build_fused build/test_cpp
	LD_LIBRARY_PATH=${XRT_DIR}/lib:$$LD_LIBRARY_PATH \
	build/test_cpp \
		build/fused/final.xclbin build/fused/insts.bin \
		${M} ${K} ${N}

# ========================= Pipeline design (Task 3) ======================
build/pipeline/aie.mlir: $(PIPELINE_DESIGN)
	mkdir -p ${@D}
	python3 $< > $@

build/pipeline/final.xclbin: build/pipeline/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/pipeline/mm.cc.o
	cd build/pipeline && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

build_pipeline: build/pipeline/final.xclbin

run_pipeline: build_pipeline build/test_cpp
	LD_LIBRARY_PATH=${XRT_DIR}/lib:$$LD_LIBRARY_PATH \
	build/test_cpp \
		build/pipeline/final.xclbin build/pipeline/insts.bin \
		${M} ${K} ${N}

# ========================= Clean =========================================
clean:
	rm -rf build
