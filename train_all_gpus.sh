#!/bin/bash

# 在 8 张 GPU 上并行训练不同配置
# 使用方法: ./train_all_gpus.sh

echo "=========================================="
echo "在 8 张 GPU 上并行训练"
echo "=========================================="

# 检查 conda 环境
if ! conda info --envs | grep -q "libero"; then
    echo "错误: libero conda 环境不存在"
    exit 1
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate libero

# 创建日志目录
mkdir -p logs

echo ""
echo "训练配置:"
echo "  GPU 0: base 算法, seed=42"
echo "  GPU 1: er 算法, seed=42"
echo "  GPU 2: ewc 算法, seed=42"
echo "  GPU 3: packnet 算法, seed=42"
echo "  GPU 4: multitask 算法, seed=42"
echo "  GPU 5: transformer 策略, seed=42"
echo "  GPU 6: base 算法, seed=100"
echo "  GPU 7: base 算法, seed=200"
echo ""
echo "日志保存在: logs/"
echo "=========================================="
echo ""

# GPU 0: base 算法
echo "[GPU 0] 启动 base 算法训练..."
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    > logs/gpu0_base_seed42.log 2>&1 &
PID0=$!

# GPU 1: er 算法
echo "[GPU 1] 启动 er 算法训练..."
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=er \
    > logs/gpu1_er_seed42.log 2>&1 &
PID1=$!

# GPU 2: ewc 算法
echo "[GPU 2] 启动 ewc 算法训练..."
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=ewc \
    > logs/gpu2_ewc_seed42.log 2>&1 &
PID2=$!

# GPU 3: packnet 算法
echo "[GPU 3] 启动 packnet 算法训练..."
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=packnet \
    > logs/gpu3_packnet_seed42.log 2>&1 &
PID3=$!

# GPU 4: multitask 算法
echo "[GPU 4] 启动 multitask 算法训练..."
CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=multitask \
    > logs/gpu4_multitask_seed42.log 2>&1 &
PID4=$!

# GPU 5: transformer 策略
echo "[GPU 5] 启动 transformer 策略训练..."
CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_transformer_policy \
    lifelong=base \
    > logs/gpu5_transformer_seed42.log 2>&1 &
PID5=$!

# GPU 6: base 算法, seed=100
echo "[GPU 6] 启动 base 算法训练 (seed=100)..."
CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6 \
python libero/lifelong/main.py \
    seed=100 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    > logs/gpu6_base_seed100.log 2>&1 &
PID6=$!

# GPU 7: base 算法, seed=200
echo "[GPU 7] 启动 base 算法训练 (seed=200)..."
CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 \
python libero/lifelong/main.py \
    seed=200 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    > logs/gpu7_base_seed200.log 2>&1 &
PID7=$!

echo ""
echo "=========================================="
echo "所有训练已启动！"
echo "=========================================="
echo ""
echo "进程 ID:"
echo "  GPU 0: $PID0"
echo "  GPU 1: $PID1"
echo "  GPU 2: $PID2"
echo "  GPU 3: $PID3"
echo "  GPU 4: $PID4"
echo "  GPU 5: $PID5"
echo "  GPU 6: $PID6"
echo "  GPU 7: $PID7"
echo ""
echo "监控命令:"
echo "  查看所有 GPU: watch -n 1 nvidia-smi"
echo "  查看日志: tail -f logs/gpu0_base_seed42.log"
echo "  查看所有进程: ps aux | grep python"
echo ""
echo "等待所有训练完成..."
echo "=========================================="

# 等待所有训练完成
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7

echo ""
echo "=========================================="
echo "所有训练完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  ls experiments/LIBERO_10/"
echo "  cat logs/gpu0_base_seed42.log"
echo ""
echo "评估模型:"
echo "  ./evaluate_all_tasks.sh 0 42 bc_rnn_policy base 9"
echo "=========================================="
