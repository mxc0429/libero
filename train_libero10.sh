#!/bin/bash

# LIBERO-10 训练脚本
# 使用方法: ./train_libero10.sh [GPU_ID] [SEED] [POLICY] [ALGO]

# 默认参数
GPU_ID=${1:-0}
SEED=${2:-42}
POLICY=${3:-bc_rnn_policy}
ALGO=${4:-base}

echo "=========================================="
echo "LIBERO-10 训练脚本"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "随机种子: $SEED"
echo "策略: $POLICY"
echo "算法: $ALGO"
echo "=========================================="
echo ""

# 检查 conda 环境
if ! conda info --envs | grep -q "mxc_libero"; then
    echo "错误: mxc_libero conda 环境不存在"
    echo "请先创建环境"
    exit 1
fi

# 激活环境
echo "激活 mxc_libero 环境..."
eval "$(conda shell.bash hook)"
conda activate mxc_libero

# 检查环境是否激活成功
if [ $? -ne 0 ]; then
    echo "错误: 无法激活 mxc_libero 环境"
    echo "请手动激活环境后再运行训练命令"
    exit 1
fi

echo "环境已激活: $(which python)"

# 检查数据集
DATASET_DIR="./libero/datasets/datasets/libero_10"
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    echo "请先下载数据集"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_EGL_DEVICE_ID=$GPU_ID

# 开始训练
echo ""
echo "开始训练..."
echo "日志将保存到 training.log"
echo ""

python libero/lifelong/main.py \
    seed=$SEED \
    benchmark_name=LIBERO_10 \
    policy=$POLICY \
    lifelong=$ALGO \
    train.num_workers=0 \
    2>&1 | tee training.log

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: experiments/LIBERO_10/${ALGO^}/${POLICY^}_seed${SEED}/"
echo "日志保存在: training.log"
echo ""
echo "评估模型:"
echo "  python libero/lifelong/evaluate.py \\"
echo "    --benchmark libero_10 \\"
echo "    --task_id 0 \\"
echo "    --algo $ALGO \\"
echo "    --policy $POLICY \\"
echo "    --seed $SEED \\"
echo "    --load_task 0 \\"
echo "    --device_id $GPU_ID"
echo ""
echo "实时显示评估:"
echo "  python evaluate_with_display.py \\"
echo "    --benchmark libero_10 \\"
echo "    --task_id 0 \\"
echo "    --algo $ALGO \\"
echo "    --policy $POLICY \\"
echo "    --seed $SEED \\"
echo "    --load_task 0 \\"
echo "    --device_id $GPU_ID \\"
echo "    --display"
echo "=========================================="
