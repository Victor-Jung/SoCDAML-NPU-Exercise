##===- Makefile -- Single / Double buffer add-one exercise ----------------===##
#
# Targets
# -------
#   build_single    Compile the single-buffer (depth=1) design
#   build_double    Compile the double-buffer design
#   run_single      Run + verify the single-buffer design  (Python)
#   run_double      Run + verify the double-buffer design  (Python)
#   trace_single    Run + parse trace -> trace_single.json
#   trace_double    Run + parse trace -> trace_double.json
#   clean           Remove all build artifacts
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# ========================= Tool paths ====================================
MLIR_AIE_DIR ?= /scratch/jungvi/mlir-aie
include ${MLIR_AIE_DIR}/programming_examples/makefile-common

# Peano (llvm-aie) install dir -- try env var first, then auto-detect from Python
PEANO_INSTALL_DIR ?= $(shell python3 -c "import aie.utils.config; print(aie.utils.config.peano_install_dir())" 2>/dev/null)
ifeq ($(PEANO_INSTALL_DIR),)
$(error PEANO_INSTALL_DIR not set and auto-detection failed. Set it or install llvm-aie.)
endif

# Installed mlir-aie include dir (contains aie_api/ headers)
MLIR_AIE_INSTALL_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())" 2>/dev/null)
AIE_INCLUDE_FLAGS = -I ${MLIR_AIE_INSTALL_DIR}/include

TRACE_PARSE = python3 ${MLIR_AIE_DIR}/python/utils/trace/parse.py
TRACE_SUMMARY = python3 ${MLIR_AIE_DIR}/python/utils/trace/get_trace_summary.py

# ========================= Configuration =================================
devicename = npu2
in1_size   = 65536
out_size   = 65536
trace_size = 8192

.PHONY: all build_single build_double \
        run_single run_double \
        trace_single trace_double clean

all: build_single build_double

# ========================= Kernel compilation ============================
# Same kernel object for both variants (int32 only).

build/single/addOne.cc.o: ${srcdir}/addOne.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${AIE_INCLUDE_FLAGS} \
		-c $< -o ${@F}

build/double/addOne.cc.o: ${srcdir}/addOne.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${AIE_INCLUDE_FLAGS} \
		-c $< -o ${@F}

# ========================= MLIR generation ===============================
# Tracing is always enabled (trace_size > 0).

build/single/aie.mlir: ${srcdir}/add_one_single.py
	mkdir -p ${@D}
	python3 $< -i1s ${in1_size} -os ${out_size} -t ${trace_size} > $@

build/double/aie.mlir: ${srcdir}/add_one_double.py
	mkdir -p ${@D}
	python3 $< -i1s ${in1_size} -os ${out_size} -t ${trace_size} > $@

# ========================= XCLBIN compilation ============================

build/single/final.xclbin: build/single/aie.mlir build/single/addOne.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

build/double/final.xclbin: build/double/aie.mlir build/double/addOne.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir

# ========================= Build aliases =================================

build_single: build/single/final.xclbin
build_double: build/double/final.xclbin

# ========================= Run (Python) ==================================

run_single: build_single
	python3 ${srcdir}/test.py \
		-x build/single/final.xclbin -i build/single/insts.bin \
		-k MLIR_AIE -i1s ${in1_size} -os ${out_size} -t ${trace_size}

run_double: build_double
	python3 ${srcdir}/test.py \
		-x build/double/final.xclbin -i build/double/insts.bin \
		-k MLIR_AIE -i1s ${in1_size} -os ${out_size} -t ${trace_size}

# ========================= Trace =========================================

trace_single: build_single
	python3 ${srcdir}/test.py \
		-x build/single/final.xclbin -i build/single/insts.bin \
		-k MLIR_AIE -i1s ${in1_size} -os ${out_size} -t ${trace_size}
	${TRACE_PARSE} --input trace.txt \
		--mlir build/single/aie.mlir.prj/input_with_addresses.mlir --output trace_single.json
	${TRACE_SUMMARY} --input trace_single.json

trace_double: build_double
	python3 ${srcdir}/test.py \
		-x build/double/final.xclbin -i build/double/insts.bin \
		-k MLIR_AIE -i1s ${in1_size} -os ${out_size} -t ${trace_size}
	${TRACE_PARSE} --input trace.txt \
		--mlir build/double/aie.mlir.prj/input_with_addresses.mlir --output trace_double.json
	${TRACE_SUMMARY} --input trace_double.json

# ========================= Clean =========================================

clean:
	rm -rf build _build *.exe trace*.txt trace*.json
