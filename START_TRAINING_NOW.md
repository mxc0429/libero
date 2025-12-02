# 立即开始训练 - 最终版本

## ✅ 所有问题已解决

1. ✅ libero 模块已安装
2. ✅ h5py pickle 错误已修复
3. ✅ persistent_workers 错误已修复

---

## 🚀 立即开始（3 种方法）

### 方法 1: 使用脚本（最简单）⭐

```bash
./train_direct.sh 0 42 bc_rnn_policy base
```

### 方法 2: 使用命令行

```bash
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.num_workers=0
```

### 方法 3: 8 GPU 并行训练

```bash
./train_all_gpus_fixed.sh
```

---

## 📝 快速测试（5 分钟）

在正式训练前，先快速测试一下：

```bash
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.num_workers=0 \
    train.n_epochs=5 \
    eval.n_eval=5
```

如果这个测试成功运行，说明环境配置正确！

---

## 🖥️ 在不同 GPU 上训练

```bash
# GPU 0
./train_direct.sh 0 42 bc_rnn_policy base

# GPU 3
./train_direct.sh 3 42 bc_rnn_policy base

# GPU 7
./train_direct.sh 7 42 bc_rnn_policy base
```

---

## 🔄 在 8 个 GPU 上并行训练

### 选项 1: 使用自动化脚本（推荐）

```bash
./train_all_gpus_fixed.sh
```

### 选项 2: 手动启动（更灵活）

打开 8 个终端，每个运行：

```bash
# 终端 1 (GPU 0)
./train_direct.sh 0 42 bc_rnn_policy base

# 终端 2 (GPU 1)
./train_direct.sh 1 100 bc_rnn_policy base

# 终端 3 (GPU 2)
./train_direct.sh 2 200 bc_rnn_policy base

# ... 以此类推
```

---

## 📊 监控训练

### 查看 GPU 使用

```bash
watch -n 1 nvidia-smi
```

### 查看训练日志

```bash
# 使用脚本训练时
tail -f training_gpu0_base_bc_rnn_policy_seed42.log

# 使用 train_all_gpus_fixed.sh 时
tail -f logs/gpu0_base_seed42.log
```

### 查看所有训练进程

```bash
ps aux | grep "libero/lifelong/main.py"
```

---

## ⏱️ 训练时间

- **单任务**: 30-60 分钟
- **全部 10 任务**: 5-10 小时
- **8 GPU 并行**: 同时完成 8 个不同配置

---

## 🎯 不同算法

```bash
# 顺序微调（最简单）
./train_direct.sh 0 42 bc_rnn_policy base

# 经验回放
./train_direct.sh 0 42 bc_rnn_policy er

# 弹性权重巩固
./train_direct.sh 0 42 bc_rnn_policy ewc

# 网络打包
./train_direct.sh 0 42 bc_rnn_policy packnet

# 多任务学习
./train_direct.sh 0 42 bc_rnn_policy multitask
```

---

## 🎓 不同策略

```bash
# RNN 策略（推荐，快速）
./train_direct.sh 0 42 bc_rnn_policy base

# Transformer 策略（性能更好）
./train_direct.sh 0 42 bc_transformer_policy base

# ViLT 策略（最佳性能）
./train_direct.sh 0 42 bc_vilt_policy base
```

---

## 🛑 停止训练

```bash
# 查找进程
ps aux | grep "libero/lifelong/main.py"

# 停止特定进程
kill [PID]

# 停止所有 LIBERO 训练
pkill -f "libero/lifelong/main.py"
```

---

## 📝 训练完成后评估

```bash
conda activate mxc_libero

python evaluate_with_display.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --display
```

---

## ✅ 完整工作流程

```bash
# 1. 激活环境
conda activate mxc_libero

# 2. 快速测试（5 分钟）
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.num_workers=0 \
    train.n_epochs=5 \
    eval.n_eval=5

# 3. 如果测试成功，开始正式训练
./train_direct.sh 0 42 bc_rnn_policy base

# 4. 监控训练（新终端）
watch -n 1 nvidia-smi
tail -f training_gpu0_base_bc_rnn_policy_seed42.log

# 5. 等待完成（5-10 小时）

# 6. 评估模型
python evaluate_with_display.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --display
```

---

## 🎉 总结

### 最简单的开始方式

```bash
./train_direct.sh 0 42 bc_rnn_policy base
```

### 充分利用 8 GPU

```bash
./train_all_gpus_fixed.sh
```

### 手动控制

```bash
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py seed=42 benchmark_name=LIBERO_10 policy=bc_rnn_policy lifelong=base train.num_workers=0
```

---

## 📚 相关文档

- **QUICK_FIX.md** - 问题修复说明
- **SERVER_TRAINING_GUIDE.md** - 服务器训练完整指南
- **TRAINING_GUIDE_CN.md** - 详细训练指南

现在可以开始训练了！🚀
