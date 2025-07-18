# V100 使用指南

本目录包含了专门为Tesla V100优化的SpInfer基准测试脚本。

## 硬件要求

- **GPU**: Tesla V100 (32GB 或 16GB)
- **CUDA**: 11.0 或更高版本
- **内存**: 至少64GB系统内存

## 主要修改

### 1. Makefile适配
- 已将SMS设置为70 (V100架构)
- 支持V100的Tensor Core

### 2. 矩阵大小调整
- 原始矩阵大小减少约50%以适应V100内存限制
- 避免内存不足错误

### 3. 性能预期
- V100性能约为RTX 4090的1/3到1/2
- 这是正常的，因为V100使用较旧的Volta架构

## 使用方法

### 1. 编译
```bash
cd kernel_benchmark
make clean
make
```

### 2. 运行所有测试
```bash
./run_all_v100.sh
```

### 3. 运行单个算法测试
```bash
# SparTA测试
./run_sparta_v100.sh

# Sputnik测试  
./run_sputnik_v100.sh

# cuSPARSE测试
./run_cusparse_v100.sh
```

### 4. 生成性能图表
```bash
python draw_fig10_v100.py
```

## 输出文件

测试完成后会生成以下CSV文件：
- `SparTA_performance_results.csv`
- `sputnik_performance_results.csv`
- `cusparse_performance_results.csv`
- `main_res.csv`

图表文件：
- `Figure10_V100.png`

## 故障排除

### 1. 编译错误
```bash
# 检查CUDA版本
nvcc --version

# 检查GPU信息
nvidia-smi

# 重新编译
make clean && make
```

### 2. 内存不足
- 如果遇到内存错误，可以进一步减小矩阵大小
- 修改脚本中的M和K数组

### 3. cuSPARSELt兼容性
- V100可能不完全支持cuSPARSELt的所有功能
- 如果SparTA测试失败，这是正常的

### 4. 性能问题
- V100的性能会比RTX 4090低
- 这是硬件架构差异导致的正常现象

## 注意事项

1. **cuSPARSELt限制**: V100的cuSPARSELt支持可能有限
2. **内存管理**: 32GB V100比16GB版本更适合大矩阵测试
3. **性能基准**: 不要直接与RTX 4090结果比较
4. **错误处理**: 脚本包含错误检查，失败时会继续下一个测试

## 技术支持

如果遇到问题，请检查：
1. CUDA版本是否支持V100
2. 是否有足够的GPU内存
3. cuSPARSELt库是否正确安装
4. 系统内存是否充足 