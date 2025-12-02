# 服务器训练快速参考

**服务器环境**: `mxc_libero`  
**GPU 数量**: 8 张

---

## 🚀 立即开始

### 最简单的方式（推荐）

```bash
./train_direct.sh 0 42 bc_rnn_policy base
```

### 充分利用 8 GPU

```bash
./train_all_gpus.sh
```

---

## 📋 命令格式

```bash
./train_direct.sh [GPU_ID] [SEED] [POLICY] [ALGO]
```

**示例**:
```bash
./train_direct.sh 0 42 bc_rnn_policy base     # GPU 0
./train_direct.sh 3 100 bc_rnn_policy er      # GPU 3
./train_direct.sh 7 42 bc_transformer_policy base  # GPU 7
```

---

## 🖥️ 指定 GPU

### 方法 1: 使用脚本

```bash
./train_direct.sh 0 42 bc_rnn_policy base  # GPU 0
./train_direct.sh 3 42 bc_rnn_policy base  # GPU 3
./train_direct.sh 7 42 bc_rnn_policy base  # GPU 7
```

### 方法 2: 使用命令行

```bash
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=3  # 指定 GPU 3
export MUJOCO_EGL_DEVICE_ID=3
python libero/lifelong/main.py seed=42 benchmark_name=LIBERO_10 policy=bc_rnn_policy lifelong=base
```

---

## 🔄 多 GPU 并行

### 自动化（推荐）

```bash
./train_all_gpus.sh
```

### 手动启动

```bash
# 打开 8 个终端，每个运行：
./train_direct.sh 0 42 bc_rnn_policy base  # 终端 1
./train_direct.sh 1 100 bc_rnn_policy base # 终端 2
./train_direct.sh 2 200 bc_rnn_policy base # 终端 3
# ... 以此类推
```

### 后台运行

```bash
mkdir -p logs
nohup ./train_direct.sh 0 42 bc_rnn_policy base > logs/gpu0.log 2>&1 &
nohup ./train_direct.sh 1 100 bc_rnn_policy base > logs/gpu1.log 2>&1 &
nohup ./train_direct.sh 2 200 bc_rnn_policy base > logs/gpu2.log 2>&1 &
# ... 以此类推
```

---

## 📊 监控

```bash
# GPU 使用
watch -n 1 nvidia-smi

# 训练日志
tail -f training_gpu0_base_bc_rnn_policy_seed42.log

# 所有进程
ps aux | grep "libero/lifelong/main.py"
```

---

## 🎯 参数选择

### 策略 (POLICY)
- `bc_rnn_policy` - RNN（推荐，快速）
- `bc_transformer_policy` - Transformer（性能好）
- `bc_vilt_policy` - ViLT（最佳性能）

### 算法 (ALGO)
- `base` - 顺序微调（推荐）
- `er` - 经验回放
- `ewc` - 弹性权重巩固
- `packnet` - 网络打包
- `multitask` - 多任务学习

---

## ⏱️ 训练时间

- **单任务**: 30-60 分钟
- **全部 10 任务**: 5-10 小时

---

## 📝 评估

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

## 📚 详细文档

- **SERVER_TRAINING_GUIDE.md** - 服务器训练完整指南 ⭐
- **START_NOW.md** - 快速开始
- **TRAINING_GUIDE_CN.md** - 详细训练指南
- **QUICK_REFERENCE_CN.md** - 命令速查

---

## ✅ 完整流程

```bash
# 1. 查看 GPU
nvidia-smi

# 2. 开始训练
./train_direct.sh 0 42 bc_rnn_policy base

# 3. 监控（新终端）
watch -n 1 nvidia-smi
tail -f training_gpu0_base_bc_rnn_policy_seed42.log

# 4. 等待完成（5-10 小时）

# 5. 评估
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

现在就开始吧！🚀
