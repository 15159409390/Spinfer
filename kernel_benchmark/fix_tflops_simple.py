#!/usr/bin/env python3
"""
简单的TFLOPS修正脚本，不依赖pandas
"""

import csv
import os

def fix_tflops_csv(input_file, output_file):
    """修正CSV文件中的TFLOPS计算"""
    
    if not os.path.exists(input_file):
        print(f"⚠ {input_file} 文件不存在，跳过")
        return False
    
    print(f"正在修正 {input_file}...")
    
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        corrections = []
        
        for row in reader:
            try:
                M = float(row['M'])
                K = float(row['K'])
                N = float(row['N'])
                sparsity = float(row['Sparsity'])
                duration_ns = float(row['Duration(ns)'])
                original_tflops = float(row['TFLOPS'])
                
                # 计算正确的TFLOPS
                density = (100 - sparsity) / 100
                flops = 2 * M * K * N * density
                correct_tflops = flops / (duration_ns * 1e3)  # ns -> TFLOPS
                
                # 记录修正情况
                correction_ratio = correct_tflops / original_tflops if original_tflops != 0 else 1
                corrections.append(correction_ratio)
                
                # 更新TFLOPS值
                row['TFLOPS'] = f"{correct_tflops:.5f}"
                
            except (KeyError, ValueError) as e:
                print(f"错误处理行: {e}")
                continue
                
            writer.writerow(row)
    
    # 统计修正情况
    if corrections:
        avg_correction = sum(corrections) / len(corrections)
        min_correction = min(corrections)
        max_correction = max(corrections)
        
        print(f"TFLOPS修正统计:")
        print(f"  - 平均修正比例: {avg_correction:.3f}")
        print(f"  - 修正比例范围: {min_correction:.3f} - {max_correction:.3f}")
        print(f"  - 处理了 {len(corrections)} 行数据")
    
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
            if fix_tflops_csv(input_file, output_file):
                print(f"✓ {input_file} 修正完成")
            else:
                print(f"✗ {input_file} 修正失败")
        except Exception as e:
            print(f"✗ {input_file} 修正出错: {e}")
        print("-" * 40)
    
    print("修正完成！")

if __name__ == "__main__":
    main()