# LIBERO 训练完整指南

## 📖 文档导航

| 文档 | 适合人群 | 阅读时间 |
|------|----------|----------|
| **QUICK_START_TRAINING.md** | 新手，想快速开始 | 3 分钟 ⭐ |
| **START_TRAINING_CN.md** | 需要详细步骤和多 GPU 使用 | 10 分钟 |
| **TRAINING_GUIDE_CN.md** | 需要完整参考和高级功能 | 20 分钟 |
| **QUICK_REFERENCE_CN.md** | 需要快速查询命令 | 随时查阅 |

---

## 🚀 最快开始（30 秒）

```bash
# 1. 查看可用 GPU
nvidia-smi

# 2. 开始训练（使用 GPU 0）
./train_libero10.sh 0 42 bc_rnn_policy base
```

---

## 📋 训练脚本

### 单 GPU 训练

```bash
./train_libero10.sh [GPU_ID] [SEED] [POLICY] [ALGO]
```

**示例：**
```bash
./train_libero10.sh 0 42 bc_rnn_policy base
./train_libero10.sh 3 100 bc_transformer_policy er
./train_libero10.sh 7 42 bc_rnn_policy multitask
```

### 多 GPU 并行训练

```bash
# 在 8 个 GPU 上并行训练不同配置
./train_all_gpus.sh
```

### 自动选择 GPU

```bash
# 自动选择最空闲的 GPU
GPU_ID=$(./select_gpu.sh)
./train_libero10.sh $GPU_ID 42 bc_rnn_policy base
```

---

## 🎯 命令行训练

### 基本命令

```bash
# 1. 激活环境
conda activate libero

# 2. 设置 GPU
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

# 3. 开始训练
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base
```

### 指定不同 GPU

```bash
# 使用 GPU 3
export CUDA_VISIBLE_DEVICES=3
export MUJOCO_EGL_DEVICE_ID=3
python libero/lifelong/main.py ...

# 或者在命令前设置
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 python libero/lifelong/main.py ...
```

---

## 🖥️ 多 GPU 使用策略

### 策略 1: 训练不同种子（推荐）

```bash
./train_libero10.sh 0 42 bc_rnn_policy base &
./train_libero10.sh 1 100 bc_rnn_policy base &
./train_libero10.sh 2 200 bc_rnn_policy base &
./train_libero10.sh 3 300 bc_rnn_policy base &
wait
```

### 策略 2: 训练不同算法

```bash
./train_libero10.sh 0 42 bc_rnn_policy base &
./train_libero10.sh 1 42 bc_rnn_policy er &
./train_libero10.sh 2 42 bc_rnn_policy ewc &
./train_libero10.sh 3 42 bc_rnn_policy packnet &
wait
```

### 策略 3: 使用自动化脚本

```bash
# 充分利用 8 张 GPU
./train_all_gpus.sh
```

---

## 📊 监控训练

### 实时监控 GPU

```bash
watch -n 1 nvidia-smi
```

### 查看训练日志

```bash
tail -f training.log
```

### 查看特定 GPU 的日志

```bash
tail -f logs/gpu0_base_seed42.log
```

---

## ⏱️ 训练时间估计

| 配置 | 单任务 | 全部 10 任务 |
|------|--------|--------------|
| RNN + base | 30-40 分钟 | 5-7 小时 |
| RNN + er/ewc | 40-50 分钟 | 7-8 小时 |
| Transformer + base | 50-60 分钟 | 8-10 小时 |
| ViLT + base | 60-90 分钟 | 10-15 小时 |

*基于 RTX 3090 / A100 的估计*

---

## 🎓 参数选择建议

### 策略选择

| 策略 | 显存需求 | 训练速度 | 性能 | 推荐场景 |
|------|----------|----------|------|----------|
| bc_rnn_policy | 4-6 GB | 快 | 中等 | 快速实验 ⭐ |
| bc_transformer_policy | 8-12 GB | 中等 | 好 | 追求性能 |
| bc_vilt_policy | 10-16 GB | 慢 | 最好 | 最佳性能 |

### 算法选择

| 算法 | 复杂度 | 性能 | 推荐场景 |
|------|--------|------|----------|
| base | 简单 | 基线 | 快速测试 ⭐ |
| er | 中等 | 好 | 减少遗忘 |
| ewc | 中等 | 好 | 保护旧知识 |
| packnet | 复杂 | 很好 | 最小遗忘 |
| multitask | 简单 | 最好 | 上限性能 |

---

## 🔧 常用配置

### 快速测试（5 分钟）

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=5 \
    eval.n_eval=5
```

### 标准训练（5-10 小时）

```bash
./train_libero10.sh 0 42 bc_rnn_policy base
```

### 高质量训练（10-20 小时）

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=100 \
    train.batch_size=64 \
    eval.n_eval=50
```

---

## 📁 输出文件结构

```
experiments/
└── LIBERO_10/
    └── Sequential/  # 或其他算法
        └── BCRNNPolicy_seed42/
            └── run_001/
                ├── config.json          # 配置文件
                ├── task0_model.pth      # 任务 0 的模型
                ├── task1_model.pth      # 任务 1 的模型
                ├── ...
                ├── task9_model.pth      # 任务 9 的模型
                ├── task0_auc.log        # 任务 0 的学习曲线
                └── result.pt            # 最终结果
```

---

## 🎯 训练后评估

### 实时显示评估

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

### 批量评估所有任务

```bash
./evaluate_all_tasks.sh 0 42 bc_rnn_policy base 9
```

---

## ❓ 常见问题

### Q: 如何指定使用哪个 GPU？

**A:** 有三种方法：

```bash
# 方法 1: 使用训练脚本
./train_libero10.sh 3 42 bc_rnn_policy base  # 使用 GPU 3

# 方法 2: 设置环境变量
export CUDA_VISIBLE_DEVICES=3
export MUJOCO_EGL_DEVICE_ID=3
python libero/lifelong/main.py ...

# 方法 3: 在命令前设置
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 python libero/lifelong/main.py ...
```

### Q: 如何在多个 GPU 上并行训练？

**A:** 使用后台运行：

```bash
# 方法 1: 手动启动
./train_libero10.sh 0 42 bc_rnn_policy base &
./train_libero10.sh 1 100 bc_rnn_policy base &
./train_libero10.sh 2 200 bc_rnn_policy base &

# 方法 2: 使用自动化脚本
./train_all_gpus.sh
```

### Q: 如何查看哪些 GPU 空闲？

**A:**

```bash
# 查看所有 GPU
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi

# 自动选择最空闲的 GPU
GPU_ID=$(./select_gpu.sh)
echo "使用 GPU: $GPU_ID"
```

### Q: 显存不足怎么办？

**A:**

```bash
# 1. 使用更小的模型
./train_libero10.sh 0 42 bc_rnn_policy base  # RNN 最小

# 2. 减小批次大小
python libero/lifelong/main.py ... train.batch_size=16

# 3. 使用显存更大的 GPU
./train_libero10.sh 7 42 bc_rnn_policy base  # 换到 GPU 7
```

### Q: 如何后台运行训练？

**A:**

```bash
# 方法 1: nohup
nohup ./train_libero10.sh 0 42 bc_rnn_policy base > training.log 2>&1 &

# 方法 2: screen
screen -S training
./train_libero10.sh 0 42 bc_rnn_policy base
# Ctrl+A, D 分离

# 方法 3: tmux
tmux new -s training
./train_libero10.sh 0 42 bc_rnn_policy base
# Ctrl+B, D 分离
```

---

## 📚 完整示例

### 示例 1: 单 GPU 训练

```bash
# 1. 查看 GPU
nvidia-smi

# 2. 开始训练
./train_libero10.sh 0 42 bc_rnn_policy base

# 3. 监控（新终端）
watch -n 1 nvidia-smi
tail -f training.log

# 4. 等待完成（5-10 小时）

# 5. 评估
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

### 示例 2: 多 GPU 并行训练

```bash
# 1. 创建日志目录
mkdir -p logs

# 2. 启动并行训练
./train_all_gpus.sh

# 3. 监控所有 GPU
watch -n 1 nvidia-smi

# 4. 查看各个训练的日志
tail -f logs/gpu0_base_seed42.log
tail -f logs/gpu1_er_seed42.log

# 5. 等待所有训练完成

# 6. 批量评估
for gpu in {0..7}; do
    echo "评估 GPU $gpu 的模型..."
done
```

---

## 🎉 总结

### 最简单的开始方式

```bash
./train_libero10.sh 0 42 bc_rnn_policy base
```

### 充分利用 8 张 GPU

```bash
./train_all_gpus.sh
```

### 查看详细文档

- 新手入门: `QUICK_START_TRAINING.md`
- 详细步骤: `START_TRAINING_CN.md`
- 完整指南: `TRAINING_GUIDE_CN.md`
- 快速参考: `QUICK_REFERENCE_CN.md`

祝你训练顺利！🚀
