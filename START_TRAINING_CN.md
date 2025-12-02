# 开始训练 - 详细步骤

## 方法 1: 使用训练脚本（最简单）⭐

### 基本用法

```bash
./train_libero10.sh [GPU_ID] [SEED] [POLICY] [ALGO]
```

### 示例

```bash
# 使用 GPU 0，种子 42，RNN 策略，基础算法
./train_libero10.sh 0 42 bc_rnn_policy base

# 使用 GPU 3，种子 100，Transformer 策略，经验回放算法
./train_libero10.sh 3 100 bc_transformer_policy er

# 使用 GPU 7，种子 42，RNN 策略，多任务学习
./train_libero10.sh 7 42 bc_rnn_policy multitask
```

---

## 方法 2: 直接使用命令行（完全控制）

### 步骤 1: 激活环境

```bash
conda activate libero
```

### 步骤 2: 设置 GPU

```bash
# 指定使用哪个 GPU（0-7）
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

### 步骤 3: 开始训练

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base
```

### 完整示例

```bash
# 激活环境
conda activate libero

# 使用 GPU 3
export CUDA_VISIBLE_DEVICES=3
export MUJOCO_EGL_DEVICE_ID=3

# 开始训练
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base
```

---

## 多 GPU 服务器使用指南 🖥️

### 查看可用 GPU

```bash
# 查看所有 GPU
nvidia-smi

# 查看 GPU 使用情况
watch -n 1 nvidia-smi
```

### 指定单个 GPU

```bash
# 方法 1: 使用环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用 GPU 0
export CUDA_VISIBLE_DEVICES=3  # 使用 GPU 3
export CUDA_VISIBLE_DEVICES=7  # 使用 GPU 7

# 方法 2: 在命令前设置
CUDA_VISIBLE_DEVICES=3 python libero/lifelong/main.py ...

# 方法 3: 使用训练脚本
./train_libero10.sh 3 42 bc_rnn_policy base  # 使用 GPU 3
```

### 在 8 张显卡上并行训练不同任务

#### 方案 1: 训练不同的种子（推荐）

```bash
# 在不同 GPU 上同时训练不同种子
./train_libero10.sh 0 42 bc_rnn_policy base &
./train_libero10.sh 1 100 bc_rnn_policy base &
./train_libero10.sh 2 200 bc_rnn_policy base &
./train_libero10.sh 3 300 bc_rnn_policy base &
./train_libero10.sh 4 400 bc_rnn_policy base &
./train_libero10.sh 5 500 bc_rnn_policy base &
./train_libero10.sh 6 600 bc_rnn_policy base &
./train_libero10.sh 7 700 bc_rnn_policy base &

# 等待所有任务完成
wait
echo "所有训练完成！"
```

#### 方案 2: 训练不同的算法

```bash
# 在不同 GPU 上同时训练不同算法
./train_libero10.sh 0 42 bc_rnn_policy base &
./train_libero10.sh 1 42 bc_rnn_policy er &
./train_libero10.sh 2 42 bc_rnn_policy ewc &
./train_libero10.sh 3 42 bc_rnn_policy packnet &
./train_libero10.sh 4 42 bc_rnn_policy multitask &
./train_libero10.sh 5 42 bc_transformer_policy base &
./train_libero10.sh 6 42 bc_transformer_policy er &
./train_libero10.sh 7 42 bc_vilt_policy base &

wait
echo "所有算法训练完成！"
```

#### 方案 3: 使用批量训练脚本

创建 `train_parallel.sh`：

```bash
#!/bin/bash

# 在 8 张 GPU 上并行训练

echo "开始并行训练..."

# GPU 0: base + seed 42
./train_libero10.sh 0 42 bc_rnn_policy base > logs/gpu0.log 2>&1 &

# GPU 1: base + seed 100
./train_libero10.sh 1 100 bc_rnn_policy base > logs/gpu1.log 2>&1 &

# GPU 2: er + seed 42
./train_libero10.sh 2 42 bc_rnn_policy er > logs/gpu2.log 2>&1 &

# GPU 3: ewc + seed 42
./train_libero10.sh 3 42 bc_rnn_policy ewc > logs/gpu3.log 2>&1 &

# GPU 4: packnet + seed 42
./train_libero10.sh 4 42 bc_rnn_policy packnet > logs/gpu4.log 2>&1 &

# GPU 5: multitask + seed 42
./train_libero10.sh 5 42 bc_rnn_policy multitask > logs/gpu5.log 2>&1 &

# GPU 6: transformer + seed 42
./train_libero10.sh 6 42 bc_transformer_policy base > logs/gpu6.log 2>&1 &

# GPU 7: vilt + seed 42
./train_libero10.sh 7 42 bc_vilt_policy base > logs/gpu7.log 2>&1 &

# 等待所有任务完成
wait

echo "所有训练完成！"
echo "查看日志: ls logs/"
```

使用方法：

```bash
# 创建日志目录
mkdir -p logs

# 赋予执行权限
chmod +x train_parallel.sh

# 开始并行训练
./train_parallel.sh

# 监控所有 GPU
watch -n 1 nvidia-smi

# 查看某个 GPU 的训练日志
tail -f logs/gpu0.log
```

---

## 监控训练进度

### 方法 1: 实时查看日志

```bash
# 查看训练日志
tail -f training.log

# 查看特定 GPU 的日志
tail -f logs/gpu0.log
```

### 方法 2: 监控 GPU 使用

```bash
# 实时监控所有 GPU
watch -n 1 nvidia-smi

# 只显示 GPU 使用率
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# 持续监控
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv'
```

### 方法 3: 检查训练进度

```bash
# 查看最新的实验目录
ls -lt experiments/LIBERO_10/Sequential/

# 查看训练结果
ls experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/run_001/

# 查看已保存的模型
ls experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/run_001/*.pth
```

---

## 常见场景

### 场景 1: 快速测试（使用 1 个 GPU）

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

### 场景 2: 正式训练（使用 1 个 GPU）

```bash
./train_libero10.sh 0 42 bc_rnn_policy base
```

### 场景 3: 并行训练多个种子（使用 3 个 GPU）

```bash
./train_libero10.sh 0 42 bc_rnn_policy base &
./train_libero10.sh 1 100 bc_rnn_policy base &
./train_libero10.sh 2 200 bc_rnn_policy base &
wait
```

### 场景 4: 充分利用 8 张 GPU

```bash
# 创建并运行并行训练脚本
./train_parallel.sh
```

---

## GPU 选择建议

### 根据模型大小选择

| 策略 | 显存需求 | 推荐 GPU |
|------|----------|----------|
| bc_rnn_policy | 4-6 GB | 任意 GPU |
| bc_transformer_policy | 8-12 GB | 显存 ≥ 12GB |
| bc_vilt_policy | 10-16 GB | 显存 ≥ 16GB |

### 查看 GPU 显存

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

### 选择空闲的 GPU

```bash
# 查看哪些 GPU 空闲
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# 自动选择最空闲的 GPU（示例脚本）
FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | nl -v 0 | sort -nrk 2 | head -n 1 | cut -f 1)
echo "使用 GPU: $FREE_GPU"
./train_libero10.sh $FREE_GPU 42 bc_rnn_policy base
```

---

## 训练参数调整

### 基本参数

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=50 \          # 训练轮数
    train.batch_size=32 \        # 批次大小
    train.lr=1e-4 \              # 学习率
    eval.n_eval=20 \             # 评估次数
    eval.eval_every=5            # 每 N 轮评估一次
```

### 快速测试参数

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=5 \
    eval.n_eval=5 \
    eval.eval_every=1
```

### 高质量训练参数

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=100 \
    train.batch_size=64 \
    eval.n_eval=50 \
    eval.eval_every=5
```

---

## 后台运行训练

### 方法 1: 使用 nohup

```bash
nohup ./train_libero10.sh 0 42 bc_rnn_policy base > training_gpu0.log 2>&1 &

# 查看进程
ps aux | grep python

# 查看日志
tail -f training_gpu0.log
```

### 方法 2: 使用 screen

```bash
# 创建新会话
screen -S training_gpu0

# 在会话中运行训练
./train_libero10.sh 0 42 bc_rnn_policy base

# 分离会话: Ctrl+A, 然后按 D

# 重新连接
screen -r training_gpu0

# 列出所有会话
screen -ls
```

### 方法 3: 使用 tmux

```bash
# 创建新会话
tmux new -s training_gpu0

# 在会话中运行训练
./train_libero10.sh 0 42 bc_rnn_policy base

# 分离会话: Ctrl+B, 然后按 D

# 重新连接
tmux attach -t training_gpu0

# 列出所有会话
tmux ls
```

---

## 完整示例：在 8 张 GPU 上训练

```bash
#!/bin/bash

# 1. 激活环境
conda activate libero

# 2. 创建日志目录
mkdir -p logs

# 3. 在每个 GPU 上启动训练
echo "在 GPU 0 上训练 base 算法..."
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    > logs/gpu0_base.log 2>&1 &

echo "在 GPU 1 上训练 er 算法..."
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=er \
    > logs/gpu1_er.log 2>&1 &

echo "在 GPU 2 上训练 ewc 算法..."
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=ewc \
    > logs/gpu2_ewc.log 2>&1 &

echo "在 GPU 3 上训练 packnet 算法..."
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=packnet \
    > logs/gpu3_packnet.log 2>&1 &

echo "在 GPU 4 上训练 multitask 算法..."
CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=multitask \
    > logs/gpu4_multitask.log 2>&1 &

echo "在 GPU 5 上训练 transformer 策略..."
CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_transformer_policy \
    lifelong=base \
    > logs/gpu5_transformer.log 2>&1 &

echo "在 GPU 6 上训练不同种子..."
CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6 \
python libero/lifelong/main.py \
    seed=100 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    > logs/gpu6_seed100.log 2>&1 &

echo "在 GPU 7 上训练不同种子..."
CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 \
python libero/lifelong/main.py \
    seed=200 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    > logs/gpu7_seed200.log 2>&1 &

# 4. 等待所有训练完成
echo "所有训练已启动！"
echo "监控 GPU: watch -n 1 nvidia-smi"
echo "查看日志: tail -f logs/gpu0_base.log"

wait
echo "所有训练完成！"
```

保存为 `train_all_gpus.sh`，然后运行：

```bash
chmod +x train_all_gpus.sh
./train_all_gpus.sh
```

---

## 故障排除

### 问题 1: GPU 被占用

```bash
# 查看哪些 GPU 空闲
nvidia-smi

# 使用空闲的 GPU
./train_libero10.sh 3 42 bc_rnn_policy base  # 使用 GPU 3
```

### 问题 2: 显存不足

```bash
# 减小批次大小
python libero/lifelong/main.py ... train.batch_size=16

# 或使用更小的模型
python libero/lifelong/main.py ... policy=bc_rnn_policy
```

### 问题 3: 环境变量未设置

```bash
# 确保设置了这两个变量
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

# 或在命令前设置
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python libero/lifelong/main.py ...
```

---

## 总结

### 最简单的开始方式

```bash
# 1. 查看可用 GPU
nvidia-smi

# 2. 选择一个空闲的 GPU（例如 GPU 3）
./train_libero10.sh 3 42 bc_rnn_policy base

# 3. 监控训练
tail -f training.log
```

### 充分利用 8 张 GPU

```bash
# 创建并运行并行训练脚本
./train_all_gpus.sh

# 监控所有 GPU
watch -n 1 nvidia-smi
```

祝你训练顺利！🚀
