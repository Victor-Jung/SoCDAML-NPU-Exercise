######################################################################
## 				Make Targets for SoftHier Software	 				##
######################################################################

app_path = $(abspath app)
sw_cmake_arg = "-DSRC_DIR=$(app_path)"
arch_cmake_arg := "-DRISCV_ARCH=rv32imafd_zfh"

sw:
	rm -rf build && mkdir build
	cd build && cmake $(sw_cmake_arg) $(arch_cmake_arg) ${SDK_HOME}/soft_hier/flex_cluster_sdk/ && make
	@! grep -q "ebreak" build/softhier.dump || (echo "Error: 'ebreak' found in build/softhier.dump" && exit 1)

clean:
	rm -rf build


######################################################################
## 				Make Targets for Run Simulator		 				##
######################################################################
run: sw
	cd build && gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary softhier.elf run --trace=debug_mem


######################################################################
## 				Make Targets for Trace Analyzer		 				##
######################################################################
trace: sw
	cd build && gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary softhier.elf run --trace=redmule --trace=idma --trace=spatz --trace=cluster_registers | tee analyze_trace.txt
	python ${SDK_HOME}/soft_hier/flex_cluster_utilities/trace_perfetto/parse.py build/analyze_trace.txt build/roi.json
	python ${SDK_HOME}/soft_hier/flex_cluster_utilities/trace_perfetto/visualize.py build/roi.json -o build/trace.pfto
