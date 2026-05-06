eval "$("/scratch/SoCDAML_exe_7b/miniconda/bin/conda" shell.bash hook)"
conda activate py311
export SDK_HOME=/scratch/SoCDAML_exe_7b/soft_hier_release
export PATH=$SDK_HOME/install/bin:$PATH
export PYTHONPATH=$SDK_HOME/install/python:$PYTHONPATH
export PYTHONPATH=$SDK_HOME/soft_hier/flex_cluster_utilities:$PYTHONPATH
export SYSTEMC_HOME=$SDK_HOME/third_party/systemc_install
export LD_LIBRARY_PATH=${SYSTEMC_HOME}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SDK_HOME/third_party/DRAMSys:$LD_LIBRARY_PATH
export PATH=$SDK_HOME/third_party/toolchain/install/bin:$PATH
export PATH=$SDK_HOME/soft_hier/build_dynlib_from_github_dramsys5/cmake_bin/bin:$PATH
