#!/bin/bash

# 评估所有 LIBERO-10 任务
# 使用方法: ./evaluate_all_tasks.sh [GPU_ID] [SEED] [POLICY] [ALGO] [LOAD_TASK]

GPU_ID=${1:-0}
SEED=${2:-42}
POLICY=${3:-bc_rnn_policy}
ALGO=${4:-base}
LOAD_TASK=${5:-9}  # 默认加载最后一个任务的模型

echo "=========================================="
echo "评估所有 LIBERO-10 任务"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "随机种子: $SEED"
echo "策略: $POLICY"
echo "算法: $ALGO"
echo "加载任务: $LOAD_TASK"
echo "=========================================="
echo ""

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate libero

# 创建结果目录
RESULT_DIR="evaluation_results"
mkdir -p $RESULT_DIR

# 结果文件
RESULT_FILE="$RESULT_DIR/results_${ALGO}_${POLICY}_seed${SEED}_load${LOAD_TASK}.txt"
echo "评估结果" > $RESULT_FILE
echo "========================================" >> $RESULT_FILE
echo "算法: $ALGO" >> $RESULT_FILE
echo "策略: $POLICY" >> $RESULT_FILE
echo "种子: $SEED" >> $RESULT_FILE
echo "加载任务: $LOAD_TASK" >> $RESULT_FILE
echo "评估时间: $(date)" >> $RESULT_FILE
echo "========================================" >> $RESULT_FILE
echo "" >> $RESULT_FILE

# 评估每个任务
for task_id in {0..9}; do
    echo ""
    echo "=========================================="
    echo "评估任务 $task_id"
    echo "=========================================="
    
    python libero/lifelong/evaluate.py \
        --benchmark libero_10 \
        --task_id $task_id \
        --algo $ALGO \
        --policy $POLICY \
        --seed $SEED \
        --load_task $LOAD_TASK \
        --device_id $GPU_ID \
        2>&1 | tee -a $RESULT_FILE
    
    echo "" >> $RESULT_FILE
done

echo ""
echo "=========================================="
echo "所有任务评估完成！"
echo "=========================================="
echo "结果保存在: $RESULT_FILE"
echo ""

# 提取成功率
echo "成功率汇总:"
echo "=========================================="
grep "success_rate" $RESULT_FILE | awk '{print "任务", NR-1, ":", $2}'
echo "=========================================="
