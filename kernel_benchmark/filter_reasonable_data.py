#!/usr/bin/env python3
"""
过滤异常数据点的脚本
"""

import csv
import os

def filter_anomalous_data(input_file, output_file, tflops_threshold=1000):
    """过滤异常的高TFLOPS数据点"""
    
    if not os.path.exists(input_file):
        print(f"⚠ {input_file} 文件不存在，跳过")
        return False
    
    print(f"正在过滤 {input_file}...")
    
    filtered_count = 0
    total_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            total_count += 1
            try:
                tflops = float(row['TFLOPS'])
                kernel = row['Kernel']
                
                # 过滤异常高的TFLOPS值（特别是cuBLAS_TC）
                if kernel == 'cuBLAS_TC' and tflops > tflops_threshold:
                    filtered_count += 1
                    print(f"  过滤异常数据: {kernel}, TFLOPS={tflops:.1f} (M={row['M']}, K={row['K']}, N={row['N']}, Sparsity={row['Sparsity']})")
                    continue
                    
                writer.writerow(row)
                
            except (ValueError, KeyError):
                writer.writerow(row)  # 保留无法解析的行
                continue
    
    print(f"  过滤了 {filtered_count}/{total_count} 个异常数据点")
    print(f"  过滤后的数据保存到 {output_file}")
    return True

def main():
    """主函数"""
    files_to_filter = [
        ('main_res_fixed.csv', 'main_res_filtered.csv'),
        ('cusparse_performance_results_fixed.csv', 'cusparse_performance_results_filtered.csv'),
        ('sputnik_performance_results_fixed.csv', 'sputnik_performance_results_filtered.csv'),
        ('SparTA_v100_performance_results_fixed.csv', 'SparTA_v100_performance_results_filtered.csv')
    ]
    
    print("开始过滤异常数据...")
    print("=" * 60)
    
    # 设置TFLOPS阈值，超过这个值的cuBLAS_TC数据将被过滤
    tflops_threshold = 500  # TFLOPS
    
    for input_file, output_file in files_to_filter:
        try:
            if filter_anomalous_data(input_file, output_file, tflops_threshold):
                print(f"✓ {input_file} 过滤完成")
            else:
                print(f"✗ {input_file} 过滤失败")
        except Exception as e:
            print(f"✗ {input_file} 过滤出错: {e}")
        print("-" * 40)
    
    print("过滤完成！现在可以使用过滤后的文件重新分析数据。")
    print(f"过滤阈值: cuBLAS_TC > {tflops_threshold} TFLOPS 的数据被移除")

if __name__ == "__main__":
    main()