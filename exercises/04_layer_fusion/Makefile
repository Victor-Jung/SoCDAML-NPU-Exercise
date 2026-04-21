##===- Makefile -- Layer-fusion exercise ------------------------------------===##
#
# Phase 1 targets (baseline):
#   build_matmul_dual    Compile the 2-core parallel matmul design
#   build_relu_dual      Compile the 2-core parallel relu design
#   run_sequential       Run matmul then relu (2 xclbin loads) + verify
#
# Phase 2 targets (fused — added later):
#   build_fused_scalar   Compile fused design with scalar relu
#   build_fused          Compile fused design with vectorized relu
#   run_fused_scalar     Run + verify fused scalar design
#   run_fused            Run + verify fused vectorized design
#
# Phase 3 targets (pipeline — added later):
#   build_pipeline       Compile 2-stage pipeline design
#   run_pipeline         Run + verify pipeline design
#
#   clean                Remove all build artifacts
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
# A = M×K int16, B = K×N int16, C/D = M×N int16  (M=K=N=256)
i1_size = 131072
i2_size = 131072
o_size  = 131072

# Override to build pipeline with the solution
PIPELINE_DESIGN ?= ${srcdir}/matmul_relu_pipeline.py

# Override to use solution kernel file
KERNEL_SRC ?= ${srcdir}/mm.cc

.PHONY: all build build_matmul_dual build_relu_dual \
        build_fused_scalar build_fused build_pipeline \
        run_sequential run_fused_scalar run_fused run_pipeline clean

all: build_matmul_dual build_relu_dual

# ========================= Kernel compilation ============================
build/mm.cc.o: $(KERNEL_SRC)
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${AIE_INCLUDE_FLAGS} \
		-c $< -o ${@F}

# ========================= Matmul dual design ============================
build/matmul_dual/aie.mlir: ${srcdir}/matmul_dual.py
	mkdir -p ${@D}
	python3 $< > $@

build/matmul_dual/final.xclbin: build/matmul_dual/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/matmul_dual/mm.cc.o
	cd build/matmul_dual && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

build_matmul_dual: build/matmul_dual/final.xclbin

# ========================= ReLU dual design ==============================
build/relu_dual/aie.mlir: ${srcdir}/relu_dual.py
	mkdir -p ${@D}
	python3 $< > $@

build/relu_dual/final.xclbin: build/relu_dual/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/relu_dual/mm.cc.o
	cd build/relu_dual && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

build_relu_dual: build/relu_dual/final.xclbin

# ========================= Sequential run (Phase 1) ======================
run_sequential: build_matmul_dual build_relu_dual
	python3 ${srcdir}/test_sequential.py \
		--mm-xclbin  build/matmul_dual/final.xclbin  \
		--mm-insts   build/matmul_dual/insts.bin      \
		--relu-xclbin build/relu_dual/final.xclbin    \
		--relu-insts  build/relu_dual/insts.bin

# ========================= Fused designs (Phase 2) =======================
# build_fused_scalar and build_fused will be added when matmul_relu_fused.py
# is created in Phase 2.

# ========================= Pipeline design (Phase 3) =====================
# build_pipeline and run_pipeline will be added when
# matmul_relu_pipeline.py is created in Phase 3.

# ========================= Clean =========================================
clean:
	rm -rf build
