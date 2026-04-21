##===- Makefile -- Layout-transform matmul exercise -------------------------===##
#
# Targets
# -------
#   build_scalar       Compile the scalar (no-transform) design
#   build_vectorized   Compile the vectorized (DMA-transform) design
#   build              Build both
#   run_scalar         Run + verify scalar design
#   run_vectorized     Run + verify vectorized design
#   clean              Remove all build artifacts
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
# A = M×K int16, B = K×N int16, C = M×N int16  (M=K=N=128)
i1_size = 32768
i2_size = 32768
o_size  = 32768

# Override to build with the solution instead of the student template
VECTORIZED_DESIGN ?= ${srcdir}/matmul_vectorized.py

.PHONY: all build build_scalar build_vectorized run_scalar run_vectorized clean

all: build

# ========================= Kernel compilation ============================
# Single object file used by both scalar and vectorized designs.

build/mm.cc.o: ${srcdir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${AIE_INCLUDE_FLAGS} \
		-c $< -o ${@F}

# ========================= Scalar design =================================

build/scalar/aie.mlir: ${srcdir}/matmul_scalar.py
	mkdir -p ${@D}
	python3 $< > $@

build/scalar/final.xclbin: build/scalar/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/scalar/mm.cc.o
	cd build/scalar && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

# ========================= Vectorized design ==============================

build/vectorized/aie.mlir: $(VECTORIZED_DESIGN)
	mkdir -p ${@D}
	python3 $< > $@

build/vectorized/final.xclbin: build/vectorized/aie.mlir build/mm.cc.o
	ln -sf ../mm.cc.o build/vectorized/mm.cc.o
	cd build/vectorized && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

# ========================= Build / Run ===================================

build_scalar: build/scalar/final.xclbin
build_vectorized: build/vectorized/final.xclbin
build: build_scalar build_vectorized

run_scalar: build_scalar
	python3 ${srcdir}/test.py \
		-x build/scalar/final.xclbin -i build/scalar/insts.bin \
		-k MLIR_AIE -i1s ${i1_size} -i2s ${i2_size} -os ${o_size}

run_vectorized: build_vectorized
	python3 ${srcdir}/test.py \
		-x build/vectorized/final.xclbin -i build/vectorized/insts.bin \
		-k MLIR_AIE -i1s ${i1_size} -i2s ${i2_size} -os ${o_size}

# ========================= Clean =========================================

clean:
	rm -rf build
