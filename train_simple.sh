#!/bin/bash

# 简单的训练脚本 - 需要先手动激活 conda 环境
# 使用方法: 
#   conda activate libero
#   ./train_simple.sh [GPU_ID] [SEED] [POLICY] [ALGO]

# 默认参数
GPU_ID=${1:-0}
SEED=${2:-42}
POLICY=${3:-bc_rnn_policy}
ALGO=${4:-base}

echo "=========================================="
echo "LIBERO-10 训练"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "随机种子: $SEED"
echo "策略: $POLICY"
echo "算法: $ALGO"
echo "Python: $(which python)"
echo "=========================================="
echo ""

# 检查是否在 libero 环境中
if ! python -c "import libero" 2>/dev/null; then
    echo "错误: 未找到 libero 模块"
    echo "请先激活 conda 环境:"
    echo "  conda activate libero"
    echo "然后再运行此脚本"
    exit 1
fi

# 检查数据集
DATASET_DIR="./libero/datasets/libero_10"
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    echo "请先下载数据集"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_EGL_DEVICE_ID=$GPU_ID

# 开始训练
echo "开始训练..."
echo "日志将保存到 training_gpu${GPU_ID}_${ALGO}_${POLICY}_seed${SEED}.log"
echo ""

python libero/lifelong/main.py \
    seed=$SEED \
    benchmark_name=LIBERO_10 \
    policy=$POLICY \
    lifelong=$ALGO \
    train.num_workers=0 \
    2>&1 | tee training_gpu${GPU_ID}_${ALGO}_${POLICY}_seed${SEED}.log

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: experiments/LIBERO_10/"
echo "日志保存在: training_gpu${GPU_ID}_${ALGO}_${POLICY}_seed${SEED}.log"
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
