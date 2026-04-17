##===- Makefile -- Distribute + join add-one exercise ---------------------===##
#
# Targets
# -------
#   build       Compile the distribute + join design
#   run         Run + verify (Python)
#   clean       Remove all build artifacts
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# ========================= Tool paths ====================================
MLIR_AIE_DIR ?= /scratch/jungvi/mlir-aie
include ${MLIR_AIE_DIR}/programming_examples/makefile-common

PEANO_INSTALL_DIR ?= $(shell python3 -c "import aie.utils.config; print(aie.utils.config.peano_install_dir())" 2>/dev/null)
ifeq ($(PEANO_INSTALL_DIR),)
$(error PEANO_INSTALL_DIR not set and auto-detection failed. Set it or install llvm-aie.)
endif

MLIR_AIE_INSTALL_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())" 2>/dev/null)
AIE_INCLUDE_FLAGS = -I ${MLIR_AIE_INSTALL_DIR}/include

# ========================= Configuration =================================
devicename = npu2
in1_size   = 32768
out_size   = 32768

.PHONY: all build run clean

all: build

# ========================= Kernel compilation ============================

build/addOne.cc.o: ${srcdir}/addOne.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${AIE_INCLUDE_FLAGS} \
		-c $< -o ${@F}

# ========================= MLIR generation ===============================

build/aie.mlir: ${srcdir}/solutions/add_one_distribute_solution.py
	mkdir -p ${@D}
	python3 $< -i1s ${in1_size} -os ${out_size} > $@

# ========================= XCLBIN compilation ============================

build/final.xclbin: build/aie.mlir build/addOne.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

# ========================= Build / Run ===================================

build: build/final.xclbin

run: build
	python3 ${srcdir}/test.py \
		-x build/final.xclbin -i build/insts.bin \
		-k MLIR_AIE -i1s ${in1_size} -os ${out_size}

# ========================= Clean =========================================

clean:
	rm -rf build
