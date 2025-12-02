#!/bin/bash

# 直接训练脚本 - 使用 conda run
# 使用方法: ./train_direct.sh [GPU_ID] [SEED] [POLICY] [ALGO]

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
echo "=========================================="
echo ""

# 检查数据集
DATASET_DIR="./libero/datasets/libero_10"
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    echo "请先下载数据集"
    exit 1
fi

# 开始训练
echo "开始训练..."
LOG_FILE="training_gpu${GPU_ID}_${ALGO}_${POLICY}_seed${SEED}.log"
echo "日志将保存到 $LOG_FILE"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID \
MUJOCO_EGL_DEVICE_ID=$GPU_ID \
conda run -n libero --no-capture-output \
python libero/lifelong/main.py \
    seed=$SEED \
    benchmark_name=LIBERO_10 \
    policy=$POLICY \
    lifelong=$ALGO \
    train.num_workers=0 \
    2>&1 | tee $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "训练完成！"
else
    echo "训练失败，退出码: $EXIT_CODE"
fi
echo "=========================================="
echo "日志保存在: $LOG_FILE"
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
echo "=========================================="
