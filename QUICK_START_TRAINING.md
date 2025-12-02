# 快速开始训练 - 3 分钟上手

## 🚀 最快开始方式

### 1. 查看可用 GPU

```bash
nvidia-smi
```

### 2. 选择一个空闲的 GPU 并开始训练

```bash
# 假设 GPU 0 空闲，使用 GPU 0 训练
./train_libero10.sh 0 42 bc_rnn_policy base
```

就这么简单！训练会自动开始。

---

## 📋 命令格式

```bash
./train_libero10.sh [GPU_ID] [SEED] [POLICY] [ALGO]
```

### 参数说明

- **GPU_ID**: 使用哪个 GPU (0-7)
- **SEED**: 随机种子 (任意整数，推荐 42)
- **POLICY**: 策略类型
  - `bc_rnn_policy` - RNN 策略（推荐，快速）
  - `bc_transformer_policy` - Transformer 策略
  - `bc_vilt_policy` - ViLT 策略
- **ALGO**: 学习算法
  - `base` - 顺序微调（推荐，简单）
  - `er` - 经验回放
  - `ewc` - 弹性权重巩固
  - `packnet` - 网络打包
  - `multitask` - 多任务学习

---

## 💡 常用示例

### 示例 1: 基础训练（推荐新手）

```bash
./train_libero10.sh 0 42 bc_rnn_policy base
```

### 示例 2: 使用不同的 GPU

```bash
./train_libero10.sh 3 42 bc_rnn_policy base  # 使用 GPU 3
./train_libero10.sh 7 42 bc_rnn_policy base  # 使用 GPU 7
```

### 示例 3: 尝试不同的算法

```bash
./train_libero10.sh 0 42 bc_rnn_policy er       # 经验回放
./train_libero10.sh 1 42 bc_rnn_policy ewc      # 弹性权重巩固
./train_libero10.sh 2 42 bc_rnn_policy multitask # 多任务学习
```

### 示例 4: 使用 Transformer 策略

```bash
./train_libero10.sh 0 42 bc_transformer_policy base
```

---

## 🖥️ 多 GPU 并行训练

### 方法 1: 手动启动多个训练

```bash
# 在 3 个 GPU 上同时训练不同种子
./train_libero10.sh 0 42 bc_rnn_policy base &
./train_libero10.sh 1 100 bc_rnn_policy base &
./train_libero10.sh 2 200 bc_rnn_policy base &
```

### 方法 2: 使用自动化脚本（推荐）

```bash
# 在所有 8 个 GPU 上并行训练
./train_all_gpus.sh
```

这会自动在 8 个 GPU 上训练不同的配置。

---

## 📊 监控训练

### 查看训练日志

```bash
tail -f training.log
```

### 监控 GPU 使用

```bash
watch -n 1 nvidia-smi
```

### 查看训练进度

```bash
# 查看已保存的模型
ls experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/run_001/
```

---

## ⏱️ 训练时间

- **单个任务**: 约 30-60 分钟
- **全部 10 个任务**: 约 5-10 小时
- **取决于**: GPU 性能、策略类型、算法

---

## 🎯 训练完成后

### 评估模型

```bash
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

### 评估所有任务

```bash
./evaluate_all_tasks.sh 0 42 bc_rnn_policy base 9
```

---

## 🔧 高级用法

### 自动选择空闲 GPU

```bash
# 自动选择最空闲的 GPU
GPU_ID=$(./select_gpu.sh)
./train_libero10.sh $GPU_ID 42 bc_rnn_policy base
```

### 后台运行

```bash
# 使用 nohup 后台运行
nohup ./train_libero10.sh 0 42 bc_rnn_policy base > training.log 2>&1 &

# 查看日志
tail -f training.log
```

### 快速测试（5 分钟）

```bash
conda activate libero

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=5 \
    eval.n_eval=5
```

---

## ❓ 常见问题

### Q: GPU 被占用怎么办？

```bash
# 查看 GPU 使用情况
nvidia-smi

# 使用其他空闲的 GPU
./train_libero10.sh 3 42 bc_rnn_policy base  # 改用 GPU 3
```

### Q: 显存不足怎么办？

```bash
# 使用更小的模型
./train_libero10.sh 0 42 bc_rnn_policy base  # RNN 策略显存需求最小
```

### Q: 如何停止训练？

```bash
# 查找进程
ps aux | grep python

# 停止进程
kill [PID]

# 或使用 Ctrl+C（如果在前台运行）
```

### Q: 训练中断了怎么办？

目前不支持断点续训，需要重新开始。建议使用 `screen` 或 `tmux` 避免中断。

---

## 📚 更多信息

- **详细训练指南**: `START_TRAINING_CN.md`
- **完整文档**: `TRAINING_GUIDE_CN.md`
- **快速参考**: `QUICK_REFERENCE_CN.md`

---

## ✅ 完整流程示例

```bash
# 1. 查看 GPU
nvidia-smi

# 2. 开始训练（使用 GPU 0）
./train_libero10.sh 0 42 bc_rnn_policy base

# 3. 监控训练（新开一个终端）
watch -n 1 nvidia-smi
tail -f training.log

# 4. 等待训练完成（5-10 小时）

# 5. 评估模型
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

就是这么简单！🎉
