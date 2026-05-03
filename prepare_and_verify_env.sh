#!/bin/bash
set -e

# Call this script to prepare the environment of the exercise on the AUP NPU Cloud
# Run it from the root of the project

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

# ========================= Environment setup =============================
deactivate 2>/dev/null || true
python3.12 -m venv .socdaml-venv
source .socdaml-venv/bin/activate
pip install -r requirements.txt
export MLIR_AIE_DIR=/notebooks/mlir-aie/
export PYTHONPATH=/opt/xilinx/xrt/python/

echo ""
echo "========================================="
echo " Environment ready. Running exercises..."
echo "========================================="
echo ""

FAILED=0

run_exercise() {
    local name="$1"
    local dir="$2"
    shift 2
    echo "--- [$name] Building and running..."
    if (cd "$dir" && "$@"); then
        echo "--- [$name] PASSED"
    else
        echo "--- [$name] FAILED"
        FAILED=1
    fi
    echo ""
}

# ========================= Exercise 01: Single / Double Buffer ===========
EX01=exercises/01_single_double_buffer

# 01a: Single buffer (already complete, no solution needed)
run_exercise "01 single buffer" "$EX01" make -j run_single

# 01b: Double buffer (use solution)
cp "$EX01/add_one_double.py" "$EX01/add_one_double.py.bak"
cp "$EX01/solutions/add_one_double_solution.py" "$EX01/add_one_double.py"
run_exercise "01 double buffer" "$EX01" make -j run_double
mv "$EX01/add_one_double.py.bak" "$EX01/add_one_double.py"

rm -rf "$EX01/build"

# ========================= Exercise 02: Distribute + Join ================
EX02=exercises/02_distribute_join

cp "$EX02/add_one_distribute.py" "$EX02/add_one_distribute.py.bak"
cp "$EX02/solutions/add_one_distribute_solution.py" "$EX02/add_one_distribute.py"
run_exercise "02 distribute join" "$EX02" make -j run
mv "$EX02/add_one_distribute.py.bak" "$EX02/add_one_distribute.py"

rm -rf "$EX02/build"

# ========================= Exercise 03: Layout Transform =================
EX03=exercises/03_layout_transform

# 03a: Scalar matmul (already complete)
run_exercise "03 scalar matmul" "$EX03" make -j run_scalar

# 03b: Vectorized matmul (use solution)
run_exercise "03 vectorized matmul" "$EX03" \
    make -j run_vectorized VECTORIZED_DESIGN="$ROOT_DIR/$EX03/solutions/matmul_vectorized_solution.py"

rm -rf "$EX03/build"

# ========================= Exercise 04: Layer Fusion =====================
EX04=exercises/04_layer_fusion

# 04: Fused design (use solution kernel + solution design)
run_exercise "04 fused" "$EX04" \
    make -j run_fused \
    KERNEL_SRC="$ROOT_DIR/$EX04/solutions/mm_solution.cc" \
    FUSED_DESIGN="$ROOT_DIR/$EX04/solutions/matmul_relu_fused_solution.py"

# 04: Pipeline design (use solution kernel + solution design)
run_exercise "04 pipeline" "$EX04" \
    make -j run_pipeline \
    KERNEL_SRC="$ROOT_DIR/$EX04/solutions/mm_solution.cc" \
    PIPELINE_DESIGN="$ROOT_DIR/$EX04/solutions/matmul_relu_pipeline_solution.py"

rm -rf "$EX04/build"

# ========================= Summary =======================================
echo "========================================="
if [ "$FAILED" -eq 0 ]; then
    echo " All exercises PASSED"
else
    echo " Some exercises FAILED"
fi
echo "========================================="

exit $FAILED
