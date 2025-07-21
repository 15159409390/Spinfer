#!/usr/bin/env python3
"""
修正kernel_benchmark测试结果中的TFLOPS计算错误
"""

import pandas as pd
import numpy as np

def fix_tflops_calculation(input_file, output_file):
    """修正CSV文件中的TFLOPS计算"""
    
    print(f"正在修正 {input_file}...")
    
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 检查必要的列是否存在
    required_cols = ['M', 'K', 'N', 'Sparsity', 'Duration(ns)', 'TFLOPS']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: {input_file} 缺少必要的列")
        return False
    
    # 计算正确的TFLOPS
    def calculate_correct_tflops(row):
        M, K, N = row['M'], row['K'], row['N']
        sparsity = row['Sparsity']
        duration_ns = row['Duration(ns)']
        
        # 计算有效的浮点运算次数 (考虑稀疏性)
        # 对于矩阵乘法: FLOPs = 2 * M * K * N * (密度)
        density = (100 - sparsity) / 100
        flops = 2 * M * K * N * density
        
        # 转换为TFLOPS (duration是纳秒)
        tflops = flops / (duration_ns * 1e3)  # ns -> TFLOPS
        
        return tflops
    
    # 应用修正
    df['TFLOPS_original'] = df['TFLOPS']  # 保存原始值用于对比
    df['TFLOPS'] = df.apply(calculate_correct_tflops, axis=1)
    
    # 统计修正情况
    tflops_diff = df['TFLOPS'] / df['TFLOPS_original']
    print(f"TFLOPS修正统计:")
    print(f"  - 平均修正比例: {tflops_diff.mean():.3f}")
    print(f"  - 修正比例范围: {tflops_diff.min():.3f} - {tflops_diff.max():.3f}")
    
    # 找出修正幅度最大的行
    max_diff_idx = (tflops_diff - 1).abs().idxmax()
    max_diff_row = df.loc[max_diff_idx]
    print(f"  - 最大修正幅度行: 原值={max_diff_row['TFLOPS_original']:.2f}, 修正值={max_diff_row['TFLOPS']:.2f}")
    
    # 删除临时列
    df = df.drop('TFLOPS_original', axis=1)
    
    # 保存修正后的数据
    df.to_csv(output_file, index=False)
    print(f"修正后的数据已保存到 {output_file}")
    
    return True

def main():
    """主函数"""
    files_to_fix = [
        ('main_res.csv', 'main_res_fixed.csv'),
        ('cusparse_performance_results.csv', 'cusparse_performance_results_fixed.csv'),
        ('sputnik_performance_results.csv', 'sputnik_performance_results_fixed.csv'),
        ('SparTA_v100_performance_results.csv', 'SparTA_v100_performance_results_fixed.csv')
    ]
    
    print("开始修正TFLOPS计算错误...")
    print("=" * 60)
    
    for input_file, output_file in files_to_fix:
        try:
            if fix_tflops_calculation(input_file, output_file):
                print(f"✓ {input_file} 修正完成")
            else:
                print(f"✗ {input_file} 修正失败")
        except FileNotFoundError:
            print(f"⚠ {input_file} 文件不存在，跳过")
        except Exception as e:
            print(f"✗ {input_file} 修正出错: {e}")
        print("-" * 40)
    
    print("修正完成！现在可以使用修正后的文件重新绘制图表。")

if __name__ == "__main__":
    main()