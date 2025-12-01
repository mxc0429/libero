#!/bin/bash
# SmolVLA-LIBERO 快速启动脚本

echo "=========================================="
echo "SmolVLA-LIBERO 快速启动"
echo "=========================================="

# 检查 LIBERO 是否已安装
if ! python -c "import libero" 2>/dev/null; then
    echo "[错误] LIBERO 未安装，请先安装 LIBERO:"
    echo "  pip install -e ."
    exit 1
fi

echo "[1/4] 安装 SmolVLA 依赖..."
pip install -r requirements_smolvla.txt

echo ""
echo "[2/4] 下载 LIBERO-10 数据集..."
python benchmark_scripts/download_libero_datasets.py --datasets libero_10 --use-huggingface

echo ""
echo "[3/4] 测试环境..."
python test_smolvla_setup.py

echo ""
echo "[4/4] 完成！"
echo ""
echo "=========================================="
echo "下一步："
echo "=========================================="
echo "1. 查看训练指南:"
echo "   cat TRAINING_GUIDE.md"
echo ""
echo "2. 训练 SmolVLA:"
echo "   python train_smolvla.py --benchmark libero_10 --task_ids 0 --num_epochs 50"
echo ""
echo "3. 评估模型:"
echo "   python evaluate_smolvla.py --checkpoint <path> --task_id 0 --save_videos"
echo "=========================================="
