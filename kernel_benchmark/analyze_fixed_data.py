#!/usr/bin/env python3
"""
分析修正后的数据
"""

import csv
from collections import defaultdict

def load_csv_data(filename):
    """加载CSV数据"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data.append({
                        'M': int(row['M']),
                        'K': int(row['K']),
                        'N': int(row['N']),
                        'Sparsity': int(row['Sparsity']),
                        'Kernel': row['Kernel'],
                        'TFLOPS': float(row['TFLOPS'])
                    })
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"警告: {filename} 文件不存在，跳过...")
    return data

def analyze_speedup():
    """分析加速比"""
    
    files_fixed = [
        "main_res_fixed.csv",
        "cusparse_performance_results_fixed.csv", 
        "sputnik_performance_results_fixed.csv",
        "SparTA_v100_performance_results_fixed.csv"
    ]
    
    methods = ['cuSPARSE', 'Sputnik', 'SparTA', 'Flash-LLM', 'SpInfer']
    
    print("分析修正后的数据...")
    print("=" * 80)
    
    all_data = []
    for file in files_fixed:
        data = load_csv_data(file)
        all_data.extend(data)
        print(f"从 {file} 加载了 {len(data)} 条记录")
    
    print(f"总共加载了 {len(all_data)} 条记录")
    print("-" * 80)
    
    # 过滤数据
    filtered_data = []
    for row in all_data:
        if row['Kernel'] in methods + ['cuBLAS_TC']:
            if 'SpInfer' in row['Kernel'] and row['Kernel'] not in ['SpInfer']:
                continue
            filtered_data.append(row)
    
    print(f"过滤后保留 {len(filtered_data)} 条记录")
    
    # 按(M, K, N, Sparsity)分组计算加速比
    groups = defaultdict(list)
    for row in filtered_data:
        key = (row['M'], row['K'], row['N'], row['Sparsity'])
        groups[key].append(row)
    
    print(f"共有 {len(groups)} 个测试配置")
    print("-" * 80)
    
    # 分析每个N值的情况
    Ns = [8, 16, 32]
    sparsities = [40, 50, 60, 70]
    
    for N in Ns:
        print(f"\nN = {N} 的加速比分析:")
        print("=" * 50)
        
        speedup_data = defaultdict(list)
        
        for key, group in groups.items():
            M, K, N_val, sparsity = key
            if N_val != N:
                continue
                
            # 找到cuBLAS_TC的性能
            cublas_tflops = None
            for row in group:
                if row['Kernel'] == 'cuBLAS_TC':
                    cublas_tflops = row['TFLOPS']
                    break
            
            if cublas_tflops is None or cublas_tflops == 0:
                continue
            
            # 计算其他方法的加速比
            for row in group:
                if row['Kernel'] != 'cuBLAS_TC':
                    speedup = row['TFLOPS'] / cublas_tflops
                    speedup_data[row['Kernel']].append(speedup)
        
        # 显示统计结果
        for method in methods:
            if method in speedup_data and speedup_data[method]:
                speedups = speedup_data[method]
                avg_speedup = sum(speedups) / len(speedups)
                min_speedup = min(speedups)
                max_speedup = max(speedups)
                
                print(f"{method:12}: 平均={avg_speedup:.3f}x, 范围=[{min_speedup:.3f}, {max_speedup:.3f}]x, 数据点={len(speedups)}")
            else:
                print(f"{method:12}: 无数据")
    
    # 显示一些具体的例子
    print("\n具体示例 (M=4096, K=4096):")
    print("=" * 50)
    
    for N in [8, 16, 32]:
        for sparsity in [40, 70]:  # 只显示两个稀疏度
            key = (4096, 4096, N, sparsity)
            if key in groups:
                group = groups[key]
                
                # 找到cuBLAS_TC
                cublas_tflops = None
                for row in group:
                    if row['Kernel'] == 'cuBLAS_TC':
                        cublas_tflops = row['TFLOPS']
                        break
                
                if cublas_tflops:
                    print(f"\nM=4096, K=4096, N={N}, 稀疏度={sparsity}%:")
                    print(f"  cuBLAS_TC: {cublas_tflops:.3f} TFLOPS")
                    
                    for row in group:
                        if row['Kernel'] != 'cuBLAS_TC':
                            speedup = row['TFLOPS'] / cublas_tflops
                            print(f"  {row['Kernel']:12}: {row['TFLOPS']:7.3f} TFLOPS ({speedup:.3f}x)")

def main():
    analyze_speedup()

if __name__ == "__main__":
    main()