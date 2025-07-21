#!/usr/bin/env python3
"""
使用修正后数据的绘图脚本
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 使用修正后的文件
files_fixed = [
    "sputnik_performance_results_fixed.csv",
    "cusparse_performance_results_fixed.csv",
    "main_res_fixed.csv",
    "SparTA_v100_performance_results_fixed.csv"
]

methods = ['cuSPARSE', 'Sputnik', 'SparTA', 'Flash-LLM', 'SpInfer']
colors = ['#000', '#C00000', '#800080','#0000FF','#4d8076']

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

def process_data(files):
    """处理数据"""
    all_data = []
    
    for file in files:
        data = load_csv_data(file)
        all_data.extend(data)
    
    if not all_data:
        print("错误: 没有找到数据文件。请先运行基准测试。")
        return []
    
    # 过滤数据，只保留我们关心的方法
    filtered_data = []
    for row in all_data:
        if row['Kernel'] in methods + ['cuBLAS_TC']:
            # 排除SpInfer的子变体
            if 'SpInfer' in row['Kernel'] and row['Kernel'] not in ['SpInfer']:
                continue
            filtered_data.append(row)
    
    # 计算加速比
    result_data = []
    
    # 按(M, K, N, Sparsity)分组
    groups = defaultdict(list)
    for row in filtered_data:
        key = (row['M'], row['K'], row['N'], row['Sparsity'])
        groups[key].append(row)
    
    for key, group in groups.items():
        # 找到cuBLAS_TC的性能
        cublas_tflops = None
        for row in group:
            if row['Kernel'] == 'cuBLAS_TC':
                cublas_tflops = row['TFLOPS']
                break
        
        if cublas_tflops is None or cublas_tflops == 0:
            continue
        
        # 计算加速比
        for row in group:
            if row['Kernel'] != 'cuBLAS_TC':
                speedup = row['TFLOPS'] / cublas_tflops
                result_data.append({
                    'M': row['M'],
                    'K': row['K'],
                    'N': row['N'],
                    'Sparsity': row['Sparsity'],
                    'Kernel': row['Kernel'],
                    'TFLOPS': row['TFLOPS'],
                    'Speedup': speedup
                })
    
    return result_data

def plot_data(data):
    """绘制数据"""
    if not data:
        print("没有数据可绘制")
        return
    
    Ns = [8, 16, 32]
    sparsities = [40, 50, 60, 70]
    
    plt.rcParams.update({'font.size': 24})
    fig, axs = plt.subplots(1, 3, figsize=(30, 6), sharex=True)
    
    for i, N in enumerate(Ns):
        # 筛选N值对应的数据
        subset = [row for row in data if row['N'] == N]
        
        if not subset:
            print(f"N={N}没有数据")
            continue
        
        print(f"\nN={N} 加速比统计 (相对于cuBLAS_TC):")
        
        # 按方法和稀疏度组织数据
        method_data = defaultdict(lambda: defaultdict(list))
        for row in subset:
            method = row['Kernel']
            sparsity = row['Sparsity']
            speedup = row['Speedup']
            method_data[method][sparsity].append(speedup)
        
        # 为每个方法计算统计信息
        for method in methods:
            if method in method_data:
                speedups = []
                for sparsity in sparsities:
                    speedups.extend(method_data[method][sparsity])
                if speedups:
                    print(f"{method}: {min(speedups):.2f} - {max(speedups):.2f}")
        
        # 绘制boxplot
        plot_data_by_method = []
        plot_labels = []
        plot_colors = []
        
        for j, method in enumerate(methods):
            if method in method_data:
                for sparsity in sparsities:
                    if method_data[method][sparsity]:
                        plot_data_by_method.append(method_data[method][sparsity])
                        plot_labels.append(f"{method}\n{sparsity}%")
                        plot_colors.append(colors[j])
        
        if plot_data_by_method:
            bp = axs[i].boxplot(plot_data_by_method, patch_artist=True, 
                               showfliers=False, whis=1.5)
            
            # 设置颜色
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axs[i].axhline(1, color='red', linewidth=2, linestyle='--')
        axs[i].set_xlabel('Method & Sparsity (%)', fontsize=24)
        axs[i].set_ylabel('Speedup vs cuBLAS_TC' if i == 0 else '', fontsize=24)
        axs[i].set_ylim(bottom=0)
        axs[i].tick_params(axis='both', which='major', labelsize=20)
        axs[i].set_title(f'N={N}', fontsize=28, fontweight='bold')
        
        # 旋转x轴标签
        if plot_labels:
            axs[i].set_xticks(range(1, len(plot_labels) + 1))
            axs[i].set_xticklabels(plot_labels, rotation=45, ha='right')
    
    # 统一y轴范围
    y_max = max(ax.get_ylim()[1] for ax in axs if ax.get_ylim()[1] > 0)
    for ax in axs:
        ax.set_ylim(0, y_max)
    
    # 添加图例
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=method) 
                      for i, method in enumerate(methods)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(methods),
              bbox_to_anchor=(0.5, 1.01), fontsize=24)
    
    plt.tight_layout()
    plt.savefig('Figure10_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n修正后的性能图表已保存为 'Figure10_fixed.png'")

def main():
    """主函数"""
    print("开始绘制修正后的性能图表...")
    print("=" * 60)
    
    # 处理数据
    data = process_data(files_fixed)
    
    if not data:
        print("没有可用数据进行绘图。请先运行基准测试。")
        return
    
    # 绘制图表
    plot_data(data)
    
    print("绘图完成！")

if __name__ == "__main__":
    main()