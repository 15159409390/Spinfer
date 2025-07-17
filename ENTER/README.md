# ENTER 文件夹说明

这个文件夹包含了SpInfer项目的conda环境。由于文件较大（约14GB），没有直接上传到GitHub。

## 如何重新创建环境

### 方法1：使用提供的脚本
```bash
# 运行初始化脚本
./Init_SpInfer.sh
```

### 方法2：手动创建环境
```bash
# 创建新的conda环境
conda create -n spinfer python=3.8

# 激活环境
conda activate spinfer

# 安装依赖包
conda install -c conda-forge cudatoolkit=12.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
```

### 方法3：使用环境文件
```bash
# 使用spinfer.yml文件创建环境
conda env create -f spinfer.yml
```

## 环境内容
- Python 3.8
- CUDA 12.2
- PyTorch
- Transformers
- 其他必要的依赖包

## 注意事项
- 确保有足够的磁盘空间（至少20GB）
- 需要CUDA兼容的GPU
- 建议使用conda而不是pip来管理环境 