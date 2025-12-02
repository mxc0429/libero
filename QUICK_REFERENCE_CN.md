# LIBERO 快速参考手册

## 一、数据集可视化

### 查看图像和动作轨迹
```bash
conda run -n libero python3 quick_visualize.py \
    "./libero/datasets/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"
```

### 生成视频
```bash
conda run -n libero python3 create_video.py \
    "./libero/datasets/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5" \
    --side_by_side
```

### 批量处理
```bash
./visualize_all_libero10.sh  # 生成所有图像
./create_all_videos.sh        # 生成所有视频
```

---

## 二、模型训练

### 快速训练
```bash
./train_libero10.sh 0 42 bc_rnn_policy base
# 参数: GPU_ID SEED POLICY ALGO
```

### 手动训练
```bash
conda activate libero

export CUDA_VISIBLE_DEVICES=0 && \
export MUJOCO_EGL_DEVICE_ID=0 && \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base
```

### 策略选择
- `bc_rnn_policy` - RNN 策略（推荐，快速）
- `bc_transformer_policy` - Transformer 策略
- `bc_vilt_policy` - ViLT 视觉-语言策略

### 算法选择
- `base` - 顺序微调（最简单）
- `er` - 经验回放
- `ewc` - 弹性权重巩固
- `packnet` - 网络打包
- `multitask` - 多任务学习

---

## 三、模型评估

### 实时显示评估（推荐）
```bash
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

### 标准评估
```bash
python libero/lifelong/evaluate.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0
```

### 评估所有任务
```bash
./evaluate_all_tasks.sh 0 42 bc_rnn_policy base 9
# 参数: GPU_ID SEED POLICY ALGO LOAD_TASK
```

---

## 四、常用命令

### 查看数据集
```bash
# 列出所有任务
ls ./libero/datasets/datasets/libero_10/

# 查看任务数量
ls ./libero/datasets/datasets/libero_10/*.hdf5 | wc -l
```

### 查看训练结果
```bash
# 查看实验目录
ls experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/

# 查看训练日志
cat training.log

# 查看结果
python -c "import torch; print(torch.load('experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/run_001/result.pt'))"
```

### 监控训练
```bash
# 实时查看日志
tail -f training.log

# 监控 GPU 使用
watch -n 1 nvidia-smi
```

---

## 五、文件结构

```
LIBERO/
├── libero/
│   ├── datasets/datasets/libero_10/  # 数据集
│   ├── lifelong/                      # 训练代码
│   └── configs/                       # 配置文件
├── experiments/                       # 训练结果
├── videos/                            # 生成的视频
├── visualizations/                    # 可视化图像
├── quick_visualize.py                 # 快速可视化
├── create_video.py                    # 视频生成
├── evaluate_with_display.py           # 实时显示评估
├── train_libero10.sh                  # 训练脚本
└── evaluate_all_tasks.sh              # 批量评估
```

---

## 六、故障排除

### 问题：找不到数据集
```bash
# 检查数据集路径
ls ./libero/datasets/datasets/libero_10/

# 如果不存在，下载数据集
python benchmark_scripts/download_libero_datasets.py --datasets libero_10
```

### 问题：CUDA 内存不足
```bash
# 减小批次大小
python libero/lifelong/main.py ... train.batch_size=16

# 使用更小的模型
python libero/lifelong/main.py ... policy=bc_rnn_policy
```

### 问题：找不到检查点
```bash
# 检查实验目录
ls experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/

# 确认参数匹配
# --seed, --algo, --policy 必须与训练时一致
```

### 问题：实时显示不工作
```bash
# 检查 X11 转发（远程服务器）
echo $DISPLAY

# 或者不使用实时显示
python evaluate_with_display.py ... # 不加 --display
```

---

## 七、性能优化

### 加速训练
```bash
# 减少训练轮数（快速测试）
python libero/lifelong/main.py ... train.n_epochs=20

# 增加批次大小（如果显存足够）
python libero/lifelong/main.py ... train.batch_size=64

# 减少评估频率
python libero/lifelong/main.py ... eval.eval_every=10
```

### 加速评估
```bash
# 减少评估次数
python evaluate_with_display.py ... --n_eval 10

# 不使用实时显示
python evaluate_with_display.py ... # 不加 --display
```

---

## 八、完整工作流程

```bash
# 1. 可视化数据集
conda run -n libero python3 quick_visualize.py \
    "./libero/datasets/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"

# 2. 训练模型
./train_libero10.sh 0 42 bc_rnn_policy base

# 3. 等待训练完成（5-10 小时）

# 4. 实时显示评估
python evaluate_with_display.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --display

# 5. 评估所有任务
./evaluate_all_tasks.sh 0 42 bc_rnn_policy base 9

# 6. 查看结果
cat evaluation_results/results_base_bc_rnn_policy_seed42_load9.txt
```

---

## 九、参数速查表

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train.n_epochs` | 50 | 训练轮数 |
| `train.batch_size` | 32 | 批次大小 |
| `train.lr` | 1e-4 | 学习率 |
| `eval.n_eval` | 20 | 评估次数 |
| `eval.max_steps` | 600 | 最大步数 |

### 数据参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.seq_len` | 10 | 序列长度 |
| `data.img_h` | 128 | 图像高度 |
| `data.img_w` | 128 | 图像宽度 |

---

## 十、相关文档

| 文档 | 说明 |
|------|------|
| `QUICK_START_CN.md` | 快速入门指南 |
| `TRAINING_GUIDE_CN.md` | 训练详细指南 |
| `README_VISUALIZATION_CN.md` | 可视化完整指南 |
| `VIDEO_GUIDE_CN.md` | 视频生成指南 |
| `README.md` | LIBERO 项目主文档 |

---

## 十一、常见任务

### 任务 1: 快速测试流程
```bash
# 使用少量轮数快速测试
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=5 \
    eval.n_eval=5
```

### 任务 2: 生成演示视频
```bash
# 为所有任务生成视频
./create_all_videos.sh

# 查看生成的视频
ls libero10_videos/
```

### 任务 3: 比较不同算法
```bash
# 训练多个算法
for algo in base er ewc; do
    ./train_libero10.sh 0 42 bc_rnn_policy $algo
done

# 评估比较
for algo in base er ewc; do
    ./evaluate_all_tasks.sh 0 42 bc_rnn_policy $algo 9
done
```

---

## 需要帮助？

1. 查看详细文档（见"相关文档"部分）
2. 检查 GitHub Issues
3. 查看 LIBERO 官方文档

祝你使用愉快！🎉
