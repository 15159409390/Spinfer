#!/bin/bash

# V100专用综合测试脚本
# 运行所有算法的基准测试

echo "=========================================="
echo "SpInfer Benchmark Suite for V100"
echo "=========================================="
echo "Matrix sizes adjusted for V100 memory constraints"
echo "Original sizes reduced by ~50% to fit in 32GB memory"
echo ""

# 检查可执行文件是否存在
check_executable() {
    if [ ! -f "$1" ]; then
        echo "Error: Executable $1 not found. Please compile first."
        echo "Run 'make' to compile all executables."
        exit 1
    fi
}

# 检查所有可执行文件
echo "Checking executables..."
check_executable "./spmm_test_sparta"
check_executable "./spmm_test_sputnik" 
check_executable "./spmm_test_cusparse"
check_executable "./spmm_test"
echo "All executables found. Starting benchmarks..."
echo ""

# 运行SparTA测试
echo "=========================================="
echo "Running SparTA benchmark..."
echo "=========================================="
./run_sparta_v100.sh

# 运行Sputnik测试
echo ""
echo "=========================================="
echo "Running Sputnik benchmark..."
echo "=========================================="
./run_sputnik_v100.sh

# 运行cuSPARSE测试
echo ""
echo "=========================================="
echo "Running cuSPARSE benchmark..."
echo "=========================================="
./run_cusparse_v100.sh

# 运行SpInfer测试
echo ""
echo "=========================================="
echo "Running SpInfer benchmark..."
echo "=========================================="
./run_all_main.sh

echo ""
echo "=========================================="
echo "All benchmarks completed for V100!"
echo "=========================================="
echo "Results saved in CSV files:"
echo "- SparTA_performance_results.csv"
echo "- sputnik_performance_results.csv" 
echo "- cusparse_performance_results.csv"
echo "- main_res.csv"
echo ""
echo "Note: Performance will be lower than RTX 4090 due to V100's older architecture." 