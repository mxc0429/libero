# LIBERO-10 训练和评估指南

## 快速开始 🚀

### 1. 训练模型

使用 LIBERO-10 数据集训练一个基础的 BC (Behavior Cloning) 模型：

```bash
# 激活环境
conda activate libero

# 基础训练命令
export CUDA_VISIBLE_DEVICES=0 && \
export MUJOCO_EGL_DEVICE_ID=0 && \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base
```

### 2. 评估模型

```bash
python libero/lifelong/evaluate.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --ep 0 \
    --load_task 0 \
    --device_id 0
```

---

## 详细说明

### 训练参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `benchmark_name` | 任务套件 | `LIBERO_10`, `LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, `LIBERO_90` |
| `policy` | 策略网络 | `bc_rnn_policy`, `bc_transformer_policy`, `bc_vilt_policy` |
| `lifelong` | 学习算法 | `base`, `er`, `ewc`, `packnet`, `multitask` |
| `seed` | 随机种子 | 任意整数 |

### 算法说明

1. **base**: 顺序微调（Sequential Finetuning）- 最简单的基线
2. **er**: Experience Replay - 经验回放
3. **ewc**: Elastic Weight Consolidation - 弹性权重巩固
4. **packnet**: PackNet - 网络打包
5. **multitask**: 多任务学习 - 同时学习所有任务

### 策略网络说明

1. **bc_rnn_policy**: 基于 RNN 的行为克隆策略
2. **bc_transformer_policy**: 基于 Transformer 的策略
3. **bc_vilt_policy**: 基于 ViLT 的视觉-语言策略

---

## 训练示例

### 示例 1: 基础顺序训练

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

这会：
- 顺序学习 LIBERO-10 中的 10 个任务
- 使用 RNN 策略
- 每个任务训练完后评估所有已学习的任务

### 示例 2: 多任务学习

```bash
export CUDA_VISIBLE_DEVICES=0 && \
export MUJOCO_EGL_DEVICE_ID=0 && \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=multitask
```

这会同时在所有 10 个任务上训练模型。

### 示例 3: 使用 Transformer 策略

```bash
export CUDA_VISIBLE_DEVICES=0 && \
export MUJOCO_EGL_DEVICE_ID=0 && \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_transformer_policy \
    lifelong=base
```

### 示例 4: 使用经验回放

```bash
export CUDA_VISIBLE_DEVICES=0 && \
export MUJOCO_EGL_DEVICE_ID=0 && \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=er
```

---

## 评估模型

### 基本评估

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

### 评估参数说明

| 参数 | 说明 |
|------|------|
| `--benchmark` | 任务套件名称 |
| `--task_id` | 要评估的任务 ID (0-9) |
| `--algo` | 训练时使用的算法 |
| `--policy` | 训练时使用的策略 |
| `--seed` | 训练时使用的种子 |
| `--load_task` | 加载哪个任务的检查点 |
| `--device_id` | GPU ID |
| `--save-videos` | 是否保存评估视频 |

### 保存评估视频

```bash
python libero/lifelong/evaluate.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --save-videos
```

视频会保存在 `experiments_saved/` 目录下。

---

## 实时显示评估过程 👁️

### 方法 1: 使用实时显示评估脚本（推荐）⭐

我已经为你创建了一个支持实时显示的评估脚本 `evaluate_with_display.py`：

```bash
# 基本使用
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

#### 功能特点

- ✅ 实时显示机器人执行任务的画面
- ✅ 显示当前步数、评估进度、成功率
- ✅ 支持两个视角切换（agentview_rgb / eye_in_hand_rgb）
- ✅ 按 'q' 键可以随时退出
- ✅ 自动统计成功率

#### 完整参数

```bash
python evaluate_with_display.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --display \
    --display_view agentview_rgb \
    --n_eval 20 \
    --max_steps 600
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--display` | 启用实时显示 | False |
| `--display_view` | 显示视角 | agentview_rgb |
| `--n_eval` | 评估次数 | 20 |
| `--max_steps` | 每次评估最大步数 | 600 |

#### 切换视角

```bash
# 显示机器人外部视角（第三人称）
python evaluate_with_display.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --display \
    --display_view agentview_rgb

# 显示手眼相机视角（第一人称）
python evaluate_with_display.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --display \
    --display_view eye_in_hand_rgb
```

### 方法 2: 不显示（仅评估）

如果不需要实时显示，可以省略 `--display` 参数：

```bash
python evaluate_with_display.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0
```

这样会更快，适合批量评估。

### 方法 3: 使用原始评估脚本

原始的评估脚本不支持实时显示，但可以保存视频：

```bash
python libero/lifelong/evaluate.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --save-videos
```

### 实时显示效果

运行实时显示评估时，你会看到：

```
========================================
评估配置:
  基准测试: libero_10
  任务 ID: 0
  任务描述: KITCHEN_SCENE8 put both the moka pots on the stove
  算法: base
  策略: bc_rnn_policy
  种子: 42
  设备: cuda:0
========================================

开始评估任务 0: KITCHEN_SCENE8 put both the moka pots on the stove
评估次数: 20
最大步数: 600
实时显示: 启用 (视角: agentview_rgb)
按 'q' 键退出显示
================================================================================

评估 1/20: ✓ 成功 (步数: 234)
   当前成功率: 100.00% (1/1)

评估 2/20: ✗ 失败 (超时)
   当前成功率: 50.00% (1/2)

...

================================================================================
最终成功率: 65.00% (13/20)
================================================================================
```

同时会弹出一个窗口显示机器人执行任务的实时画面。

### 注意事项

1. **显示需要 X11 支持**：如果在远程服务器上运行，需要 X11 转发或使用 VNC
2. **性能影响**：实时显示会略微降低评估速度（约 10-20%）
3. **窗口大小**：可以手动调整显示窗口的大小
4. **退出方式**：按 'q' 键可以随时退出评估

---

## 训练输出

训练过程中会看到：

```
=================== Lifelong Benchmark Information  ===================
 Name: LIBERO_10
 # Tasks: 10
    - Task 1:
        KITCHEN_SCENE8 put both the moka pots on the stove
    - Task 2:
        ...
 # demonstrations: (50) (50) (50) ...
 # sequences: (457) (445) (423) ...
=======================================================================

[info] start lifelong learning with algo Sequential
[info] policy has 12.3 GFLOPs and 45.6 MParams

[info] start training on task 0
[info] Epoch:   0 | train loss:  2.45 | time: 0.12
[info] Epoch:   0 | succ: 0.15 ± 0.08 | best succ: 0.15 | succ. AoC 0.15 | time: 2.34
[info] Epoch:   5 | train loss:  1.23 | time: 0.11
[info] Epoch:   5 | succ: 0.45 ± 0.11 | best succ: 0.45 | succ. AoC 0.30 | time: 2.31
...
```

### 输出说明

- **train loss**: 训练损失
- **succ**: 成功率（0-1）
- **best succ**: 当前最佳成功率
- **succ. AoC**: 成功率曲线下面积（Area of Curve）
- **time**: 训练/评估时间（分钟）

---

## 模型保存位置

训练后的模型保存在：

```
experiments/
└── LIBERO_10/
    └── Sequential/  # 或其他算法名
        └── BCRNNPolicy_seed42/
            └── run_001/
                ├── config.json
                ├── task0_model.pth
                ├── task1_model.pth
                ├── ...
                └── result.pt
```

---

## 配置文件

### 查看默认配置

```bash
cat libero/configs/config.yaml
```

### 修改配置

可以通过命令行覆盖配置：

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=100 \
    train.batch_size=64 \
    eval.n_eval=50
```

### 常用配置项

```yaml
train:
  n_epochs: 50              # 训练轮数
  batch_size: 32            # 批次大小
  lr: 1e-4                  # 学习率
  eval_every: 5             # 每 N 轮评估一次

eval:
  n_eval: 20                # 评估次数
  max_steps: 600            # 每次评估的最大步数
  eval: true                # 是否在训练后评估

data:
  seq_len: 10               # 序列长度
  img_h: 128                # 图像高度
  img_w: 128                # 图像宽度
```

---

## 常见问题

### Q: 训练需要多长时间？

- 单个任务：约 30-60 分钟（取决于 GPU）
- 全部 10 个任务：约 5-10 小时

### Q: 需要多少显存？

- RNN 策略：约 4-6 GB
- Transformer 策略：约 8-12 GB
- ViLT 策略：约 10-16 GB

### Q: 如何使用多 GPU？

```bash
export CUDA_VISIBLE_DEVICES=0,1 && \
python libero/lifelong/main.py ...
```

### Q: 如何恢复训练？

目前不支持直接恢复，但可以：
1. 加载预训练模型
2. 从特定任务开始训练

### Q: 评估时出现 "cannot find checkpoint" 错误？

检查：
1. 模型是否训练完成
2. `--seed` 是否与训练时一致
3. `--algo` 和 `--policy` 是否正确

### Q: 如何提高成功率？

1. 增加训练轮数：`train.n_epochs=100`
2. 调整学习率：`train.lr=5e-5`
3. 使用更大的批次：`train.batch_size=64`
4. 尝试不同的策略网络

---

## 完整工作流程示例

```bash
# 1. 激活环境
conda activate libero

# 2. 训练模型
export CUDA_VISIBLE_DEVICES=0 && \
export MUJOCO_EGL_DEVICE_ID=0 && \
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base

# 3. 等待训练完成（约 5-10 小时）

# 4. 评估任务 0
python libero/lifelong/evaluate.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo base \
    --policy bc_rnn_policy \
    --seed 42 \
    --load_task 0 \
    --device_id 0 \
    --save-videos

# 5. 评估所有任务
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

# 6. 查看结果
cat experiments/LIBERO_10/Sequential/BCRNNPolicy_seed42/run_001/result.pt
```

---

## 进阶使用

### 使用 WandB 记录训练

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    use_wandb=true
```

### 自定义任务顺序

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    data.task_order_index=1  # 使用不同的任务顺序
```

### 调试模式（快速测试）

```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.n_epochs=5 \
    eval.n_eval=5
```

---

## 相关文档

- `README.md` - LIBERO 项目主文档
- `README_VISUALIZATION_CN.md` - 数据集可视化指南
- `VIDEO_GUIDE_CN.md` - 视频生成指南
- `QUICK_START_CN.md` - 快速入门

---

## 下一步

训练完成后，你可以：
1. 评估模型性能
2. 可视化评估过程
3. 分析成功率和损失
4. 尝试不同的算法和策略
5. 准备切换到 SmolVLA 模型

祝你训练顺利！🎉
