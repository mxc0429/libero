#!/bin/bash

# 自动选择最空闲的 GPU
# 使用方法: GPU_ID=$(./select_gpu.sh)

# 查找显存使用最少的 GPU
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
           awk '{print $1, $2}' | \
           sort -k2 -n | \
           head -n 1 | \
           awk '{print $1}')

if [ -z "$FREE_GPU" ]; then
    echo "错误: 无法找到可用的 GPU" >&2
    exit 1
fi

echo $FREE_GPU
