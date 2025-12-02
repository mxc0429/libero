# LIBERO 环境设置指南

## 问题诊断

如果你看到错误：
```
ModuleNotFoundError: No module named 'libero'
```

这说明 LIBERO 包还没有安装。

---

## 🔧 解决方案

### 步骤 1: 确认你在正确的目录

```bash
# 应该在 LIBERO 项目根目录
pwd
# 输出应该类似: /data1/MXC/libero

# 检查是否有 setup.py
ls setup.py
```

### 步骤 2: 激活环境

```bash
conda activate mxc_libero
```

### 步骤 3: 安装 LIBERO 包

```bash
# 以开发模式安装
pip install -e .
```

这会安装 LIBERO 及其所有依赖。

### 步骤 4: 验证安装

```bash
# 测试是否安装成功
python -c "import libero; print('LIBERO 安装成功！')"
python -c "from libero.libero import get_libero_path; print(get_libero_path('datasets'))"
```

如果没有错误，说明安装成功！

---

## 📋 完整安装流程

```bash
# 1. 进入项目目录
cd /data1/MXC/libero

# 2. 激活环境
conda activate mxc_libero

# 3. 安装 LIBERO
pip install -e .

# 4. 验证安装
python -c "import libero; print('安装成功')"

# 5. 开始训练
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py seed=42 benchmark_name=LIBERO_10 policy=bc_rnn_policy lifelong=base
```

---

## 🔍 检查依赖

### 检查 Python 版本

```bash
python --version
# 应该是 Python 3.8 或更高
```

### 检查关键依赖

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import robosuite; print('robosuite 已安装')"
python -c "import robomimic; print('robomimic 已安装')"
```

---

## 🛠️ 如果安装失败

### 问题 1: pip install -e . 失败

**解决方案 1: 更新 pip**
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

**解决方案 2: 手动安装依赖**
```bash
# 先安装依赖
pip install -r requirements.txt

# 再安装 LIBERO
pip install -e .
```

### 问题 2: 缺少 requirements.txt

```bash
# 检查是否有 requirements.txt
ls requirements.txt

# 如果没有，手动安装核心依赖
pip install torch torchvision
pip install numpy
pip install hydra-core
pip install robosuite
pip install robomimic
```

### 问题 3: CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvidia-smi

# 根据 CUDA 版本安装对应的 PyTorch
# CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ 验证完整设置

运行这个脚本来验证所有设置：

```bash
python << 'EOF'
import sys
print("Python 版本:", sys.version)

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  CUDA 版本:", torch.version.cuda)
        print("  GPU 数量:", torch.cuda.device_count())
except ImportError:
    print("✗ PyTorch 未安装")

try:
    import libero
    print("✓ LIBERO 已安装")
    from libero.libero import get_libero_path
    print("  数据集路径:", get_libero_path("datasets"))
except ImportError as e:
    print("✗ LIBERO 未安装:", e)

try:
    import robosuite
    print("✓ robosuite 已安装")
except ImportError:
    print("✗ robosuite 未安装")

try:
    import robomimic
    print("✓ robomimic 已安装")
except ImportError:
    print("✗ robomimic 未安装")

try:
    import hydra
    print("✓ hydra 已安装")
except ImportError:
    print("✗ hydra 未安装")

print("\n如果所有包都显示 ✓，说明环境配置正确！")
EOF
```

---

## 🚀 安装后立即开始训练

```bash
# 1. 确认安装成功
python -c "import libero; print('OK')"

# 2. 设置 GPU
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

# 3. 快速测试（5 分钟）
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=5 \
    eval.n_eval=5

# 4. 如果测试成功，开始正式训练
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base
```

---

## 📝 常见问题

### Q: pip install -e . 需要多长时间？

A: 通常 5-10 分钟，取决于网络速度和需要安装的依赖数量。

### Q: 是否需要 root 权限？

A: 不需要，conda 环境中的安装不需要 root 权限。

### Q: 如何重新安装？

```bash
# 卸载
pip uninstall libero -y

# 重新安装
pip install -e .
```

### Q: 如何更新 LIBERO？

```bash
# 拉取最新代码
git pull

# 重新安装
pip install -e . --upgrade
```

---

## 🎯 完整的首次设置流程

```bash
# 1. 克隆仓库（如果还没有）
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

# 2. 创建 conda 环境（如果还没有）
conda create -n mxc_libero python=3.8
conda activate mxc_libero

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 PyTorch（根据你的 CUDA 版本）
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 5. 安装 LIBERO
pip install -e .

# 6. 下载数据集
python benchmark_scripts/download_libero_datasets.py --datasets libero_10

# 7. 验证安装
python -c "import libero; print('安装成功')"

# 8. 开始训练
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py seed=42 benchmark_name=LIBERO_10 policy=bc_rnn_policy lifelong=base
```

---

## 💡 提示

- 安装过程中如果遇到网络问题，可以使用国内镜像：
  ```bash
  pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 如果 `pip install -e .` 很慢，可以先安装核心依赖：
  ```bash
  pip install torch numpy hydra-core
  pip install -e . --no-deps
  ```

---

## ✅ 安装成功的标志

运行以下命令应该没有错误：

```bash
python -c "from libero.libero import get_libero_path; print(get_libero_path('datasets'))"
```

输出应该类似：
```
/data1/MXC/libero/libero/datasets
```

现在你可以开始训练了！🚀
