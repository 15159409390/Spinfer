#!/bin/bash

# V100专用Sputnik测试脚本
# 调整矩阵大小以适应V100的32GB内存限制

# 减小矩阵大小以适应V100内存限制
M=(4096 2048 16000 16000 14336 2560 2560 1792 2048 6912 4096 9472 7168 2048 4096 5504 16000 10240 1792 10752 3584 14336 3584 13824 4608 18432 4608 18432 6144 24576 6144)
K=(14784 2048 2560 4096 4096 2560 6912 10240 5504 2560 4096 1792 2048 7168 14336 2048 2048 1792 9472 3584 3584 3584 14336 4608 4608 4608 18432 6144 6144 6144 24576)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)

echo "Starting Sputnik benchmark on V100..."
echo "Matrix sizes adjusted for V100 memory constraints"

for ((i=0; i<${#M[@]}; i++)); do
    m=${M[i]}
    k=${K[i]}
    for n in "${N[@]}"; do
        for s in "${SPARSITY[@]}"; do
            for sk in "${SPLIT_K[@]}"; do
                echo "Running Sputnik test case: M=$m, K=$k, N=$n, S=$s, SK=$sk"
                ./spmm_test_sputnik $m $k $n $s $sk
                
                # 添加错误检查
                if [ $? -ne 0 ]; then
                    echo "Error: Sputnik test failed for M=$m, K=$k, N=$n, S=$s, SK=$sk"
                    continue
                fi
            done
        done
    done
done

echo "Sputnik benchmark completed for V100" 