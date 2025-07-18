#!/bin/bash

# V100专用SparTA测试脚本
echo "Compiling for V100 architecture..."

# 重新编译V100版本
make clean
make spmm_test_sparta_v100

if [ $? -ne 0 ]; then
    echo "Compilation failed. Please check CUDA installation."
    exit 1
fi

echo "Compilation successful. Starting benchmark..."

# V100专用矩阵大小
M=(4096 2048 16000 16000 14336 2560 2560 1792 2048 6912 4096 9472 7168 2048 4096 5504 16000 10240 1792 10752 3584 14336 3584 13824 4608 18432 4608 18432 6144 24576 6144)
K=(14784 2048 2560 4096 4096 2560 6912 10240 5504 2560 4096 1792 2048 7168 14336 2048 2048 1792 9472 3584 3584 3584 14336 4608 4608 4608 18432 6144 6144 6144 24576)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)

echo "Starting SparTA benchmark on V100..."
echo "Matrix sizes adjusted for V100 memory constraints"
echo "Using V100-compatible version (Sputnik only, no cuSPARSELt)"

for ((i=0; i<${#M[@]}; i++)); do
    m=${M[i]}
    k=${K[i]}
    for n in "${N[@]}"; do
        for s in "${SPARSITY[@]}"; do
            for sk in "${SPLIT_K[@]}"; do
                echo "Running SparTA V100 test case: M=$m, K=$k, N=$n, S=$s, SK=$sk"
                ./spmm_test_sparta_v100 $m $k $n $s $sk
                
                if [ $? -ne 0 ]; then
                    echo "Error: SparTA V100 test failed for M=$m, K=$k, N=$n, S=$s, SK=$sk"
                    continue
                fi
            done
        done
    done
done

echo "SparTA V100 benchmark completed" 