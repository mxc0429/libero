# 服务器训练指南

**你的服务器环境**: `mxc_libero`  
**GPU 数量**: 8 张

---

## 🚀 快速开始（3 种方法）

### 方法 1: 使用 train_direct.sh（最推荐）✨

```bash
./train_direct.sh 0 42 bc_rnn_policy base
```

**优点**: 
- 自动使用 `mxc_libero` 环境
- 不需要手动激活环境
- 自动保存日志

**参数说明**:
- `0` - GPU ID (0-7)
- `42` - 随机种子
- `bc_rnn_policy` - 策略类型
- `base` - 算法类型

### 方法 2: 使用 train_simple.sh

```bash
# 先激活环境
conda activate mxc_libero

# 然后运行
./train_simple.sh 0 42 bc_rnn_policy base
```

### 方法 3: 直接使用命令行

```bash
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base
```

---

## 🖥️ 在不同 GPU 上训练

### 使用脚本（推荐）

```bash
# GPU 0
./train_direct.sh 0 42 bc_rnn_policy base

# GPU 3
./train_direct.sh 3 42 bc_rnn_policy base

# GPU 7
./train_direct.sh 7 42 bc_rnn_policy base
```

### 使用命令行

```bash
# GPU 0
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py seed=42 benchmark_name=LIBERO_10 policy=bc_rnn_policy lifelong=base

# GPU 3
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=3
export MUJOCO_EGL_DEVICE_ID=3
python libero/lifelong/main.py seed=42 benchmark_name=LIBERO_10 policy=bc_rnn_policy lifelong=base
```

---

## 🔄 充分利用 8 张 GPU

### 方法 1: 使用自动化脚本（最简单）

```bash
./train_all_gpus.sh
```

这会自动在 8 个 GPU 上训练：
- GPU 0: base 算法, seed=42
- GPU 1: er 算法, seed=42
- GPU 2: ewc 算法, seed=42
- GPU 3: packnet 算法, seed=42
- GPU 4: multitask 算法, seed=42
- GPU 5: transformer 策略, seed=42
- GPU 6: base 算法, seed=100
- GPU 7: base 算法, seed=200

### 方法 2: 手动启动（更灵活）

打开 8 个 SSH 终端，每个终端运行：

**终端 1 (GPU 0):**
```bash
./train_direct.sh 0 42 bc_rnn_policy base
```

**终端 2 (GPU 1):**
```bash
./train_direct.sh 1 100 bc_rnn_policy base
```

**终端 3 (GPU 2):**
```bash
./train_direct.sh 2 200 bc_rnn_policy base
```

...以此类推

### 方法 3: 后台运行（推荐用于长时间训练）

```bash
# 创建日志目录
mkdir -p logs

# 在后台启动所有训练
nohup ./train_direct.sh 0 42 bc_rnn_policy base > logs/gpu0.log 2>&1 &
nohup ./train_direct.sh 1 100 bc_rnn_policy base > logs/gpu1.log 2>&1 &
nohup ./train_direct.sh 2 200 bc_rnn_policy base > logs/gpu2.log 2>&1 &
nohup ./train_direct.sh 3 300 bc_rnn_policy base > logs/gpu3.log 2>&1 &
nohup ./train_direct.sh 4 42 bc_rnn_policy er > logs/gpu4.log 2>&1 &
nohup ./train_direct.sh 5 42 bc_rnn_policy ewc > logs/gpu5.log 2>&1 &
nohup ./train_direct.sh 6 42 bc_rnn_policy packnet > logs/gpu6.log 2>&1 &
nohup ./train_direct.sh 7 42 bc_rnn_policy multitask > logs/gpu7.log 2>&1 &

# 查看所有进程
ps aux | grep python | grep libero

# 查看日志
tail -f logs/gpu0.log
```

---

## 📊 监控训练

### 查看 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

### 查看训练日志

```bash
# 查看特定 GPU 的日志
tail -f training_gpu0_base_bc_rnn_policy_seed42.log

# 或者（如果使用 nohup）
tail -f logs/gpu0.log

# 查看所有日志
tail -f logs/*.log
```

### 查看训练进程

```bash
# 查看所有 Python 进程
ps aux | grep python

# 查看 LIBERO 训练进程
ps aux | grep "libero/lifelong/main.py"

# 统计正在运行的训练数量
ps aux | grep "libero/lifelong/main.py" | wc -l
```

---

## 🎯 不同算法和策略

### 算法选择

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

### 策略选择

```bash
# RNN 策略（推荐，快速，显存需求小）
./train_direct.sh 0 42 bc_rnn_policy base

# Transformer 策略（性能更好，显存需求中等）
./train_direct.sh 0 42 bc_transformer_policy base

# ViLT 策略（性能最好，显存需求大）
./train_direct.sh 0 42 bc_vilt_policy base
```

---

## ⏱️ 训练时间估计

| 配置 | 单任务 | 全部 10 任务 |
|------|--------|--------------|
| RNN + base | 30-40 分钟 | 5-7 小时 |
| RNN + er/ewc | 40-50 分钟 | 7-8 小时 |
| Transformer + base | 50-60 分钟 | 8-10 小时 |
| ViLT + base | 60-90 分钟 | 10-15 小时 |

---

## 🔧 常见操作

### 快速测试（5 分钟）

```bash
conda activate mxc_libero
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

### 停止训练

```bash
# 查找进程 ID
ps aux | grep "libero/lifelong/main.py"

# 停止特定进程
kill [PID]

# 停止所有 LIBERO 训练
pkill -f "libero/lifelong/main.py"
```

### 查看训练进度

```bash
# 查看已保存的模型
ls experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/run_001/

# 查看训练结果
ls experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/run_001/*.pth
```

---

## 📝 训练完成后评估

### 实时显示评估

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

### 批量评估所有任务

```bash
conda activate mxc_libero

for task_id in {0..9}; do
    echo "评估任务 $task_id"
    python libero/lifelong/evaluate.py \
        --benchmark libero_10 \
        --task_id $task_id \
        --algo base \
        --policy bc_rnn_policy \
        --seed 42 \
        --load_task $task_id \
        --device_id 0
done
```

---

## ✅ 完整工作流程示例

### 单 GPU 训练

```bash
# 1. SSH 连接到服务器
ssh your_server

# 2. 进入项目目录
cd ~/Robot/LIBERO

# 3. 查看 GPU 使用情况
nvidia-smi

# 4. 开始训练（使用空闲的 GPU）
./train_direct.sh 0 42 bc_rnn_policy base

# 5. 监控训练（新开一个 SSH 终端）
watch -n 1 nvidia-smi
tail -f training_gpu0_base_bc_rnn_policy_seed42.log

# 6. 等待训练完成（5-10 小时）

# 7. 评估模型
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

### 8 GPU 并行训练

```bash
# 1. SSH 连接到服务器
ssh your_server

# 2. 进入项目目录
cd ~/Robot/LIBERO

# 3. 查看 GPU 使用情况
nvidia-smi

# 4. 启动并行训练
./train_all_gpus.sh

# 5. 监控所有 GPU（新开一个 SSH 终端）
watch -n 1 nvidia-smi

# 6. 查看各个训练的日志
tail -f logs/gpu0_base_seed42.log
tail -f logs/gpu1_er_seed42.log

# 7. 等待所有训练完成

# 8. 批量评估
conda activate mxc_libero
for task_id in {0..9}; do
    python libero/lifelong/evaluate.py \
        --benchmark libero_10 \
        --task_id $task_id \
        --algo base \
        --policy bc_rnn_policy \
        --seed 42 \
        --load_task $task_id \
        --device_id 0
done
```

---

## 🎓 推荐的训练策略

### 策略 1: 快速验证流程（1 小时）

```bash
# 在 1 个 GPU 上快速测试
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=10 \
    eval.n_eval=10
```

### 策略 2: 标准训练（5-10 小时）

```bash
# 在 1 个 GPU 上完整训练
./train_direct.sh 0 42 bc_rnn_policy base
```

### 策略 3: 充分利用资源（5-10 小时）

```bash
# 在 8 个 GPU 上并行训练不同配置
./train_all_gpus.sh
```

### 策略 4: 多种子训练（用于论文）

```bash
# 在 8 个 GPU 上训练 8 个不同种子
./train_direct.sh 0 42 bc_rnn_policy base &
./train_direct.sh 1 100 bc_rnn_policy base &
./train_direct.sh 2 200 bc_rnn_policy base &
./train_direct.sh 3 300 bc_rnn_policy base &
./train_direct.sh 4 400 bc_rnn_policy base &
./train_direct.sh 5 500 bc_rnn_policy base &
./train_direct.sh 6 600 bc_rnn_policy base &
./train_direct.sh 7 700 bc_rnn_policy base &
```

---

## 📚 脚本说明

| 脚本 | 说明 | 使用场景 |
|------|------|----------|
| `train_direct.sh` | 自动使用 mxc_libero 环境 | 最推荐 ⭐ |
| `train_simple.sh` | 需要手动激活环境 | 需要更多控制 |
| `train_all_gpus.sh` | 8 GPU 并行训练 | 充分利用资源 |
| `train_libero10.sh` | 原始脚本（已更新） | 兼容性 |

---

## 🎉 总结

### 最简单的开始方式

```bash
./train_direct.sh 0 42 bc_rnn_policy base
```

### 充分利用 8 张 GPU

```bash
./train_all_gpus.sh
```

### 手动控制（最灵活）

```bash
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py seed=42 benchmark_name=LIBERO_10 policy=bc_rnn_policy lifelong=base
```

---

## 📞 需要帮助？

查看其他文档：
- `START_NOW.md` - 快速开始指南
- `TRAINING_GUIDE_CN.md` - 详细训练指南
- `QUICK_REFERENCE_CN.md` - 命令速查表

现在就在你的服务器上开始训练吧！🚀
