# LIBERO 训练指南

本文档介绍如何使用 LIBERO 进行模型训练和评估，包括原始方法和 SmolVLA 方法。

---

## 📋 目录

1. [环境准备](#环境准备)
2. [数据集下载](#数据集下载)
3. [方法一：原始 LIBERO 训练](#方法一原始-libero-训练)
4. [方法二：SmolVLA 训练](#方法二smolvla-训练)
5. [对比总结](#对比总结)

---

## 环境准备

### 安装 LIBERO
```bash
conda create -n libero python=3.8
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

### 安装 SmolVLA 额外依赖（如果使用 SmolVLA）
```bash
pip install -r requirements_smolvla.txt
```

### 测试环境
```bash
# 测试 LIBERO
python -c "import libero; print('LIBERO OK')"

# 测试 SmolVLA（可选）
python test_smolvla_setup.py
```

---

## 数据集下载

### 推荐：使用 HuggingFace 下载
```bash
# 下载 LIBERO-10（推荐新手，2GB）
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_10 \
    --use-huggingface

# 或下载所有数据集（30GB）
python benchmark_scripts/download_libero_datasets.py \
    --datasets all \
    --use-huggingface
```

### 可用数据集
| 数据集 | 大小 | 任务数 | 用途 |
|--------|------|--------|------|
| libero_10 | 2GB | 10 | 快速测试 |
| libero_spatial | 2GB | 10 | 空间关系 |
| libero_object | 2GB | 10 | 物体泛化 |
| libero_goal | 2GB | 10 | 目标泛化 |
| libero_100 | 20GB | 100 | 大规模训练 |

---

## 方法一：原始 LIBERO 训练

### 特点
- ✅ 支持多种终身学习算法（ER, EWC, PackNet等）
- ✅ 三种策略网络（RNN, Transformer, ViLT）
- ✅ 官方基准，结果可复现
- ✅ 适合终身学习研究

### 1. 基础训练

#### 单任务训练
```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_transformer_policy \
    lifelong=base
```

#### 多任务训练
```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_transformer_policy \
    lifelong=multitask
```

#### 终身学习（Experience Replay）
```bash
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_transformer_policy \
    lifelong=er
```

### 2. 可用算法

| 算法 | 参数 | 说明 |
|------|------|------|
| Sequential | `lifelong=base` | 顺序微调基线 |
| Multitask | `lifelong=multitask` | 多任务学习 |
| ER | `lifelong=er` | 经验回放 |
| AGEM | `lifelong=agem` | 平均梯度情景记忆 |
| EWC | `lifelong=ewc` | 弹性权重巩固 |
| PackNet | `lifelong=packnet` | 网络打包 |

### 3. 可用策略

| 策略 | 参数 | 说明 |
|------|------|------|
| BC-RNN | `policy=bc_rnn_policy` | 基于 RNN |
| BC-Transformer | `policy=bc_transformer_policy` | 基于 Transformer |
| BC-ViLT | `policy=bc_vilt_policy` | 视觉-语言 Transformer |

### 4. 评估

```bash
python libero/lifelong/evaluate.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo multitask \
    --policy bc_transformer_policy \
    --seed 42 \
    --ep 50 \
    --device_id 0 \
    --save-videos
```

### 5. 配置文件

修改 `libero/configs/config.yaml` 或创建自定义配置：
```yaml
seed: 42
benchmark_name: "LIBERO_10"
policy:
  policy_type: "bc_transformer_policy"
train:
  n_epochs: 50
  batch_size: 16
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 0.0001
```

---

## 方法二：SmolVLA 训练

### 特点
- ✅ 基于预训练的视觉-语言模型
- ✅ 端到端自然语言理解
- ✅ 简单的命令行接口
- ✅ 适合快速原型和迁移学习

### 1. 基础训练

#### 单任务训练
```bash
python train_smolvla.py \
    --benchmark libero_10 \
    --task_ids 0 \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 1e-4 \
    --device cuda:0 \
    --seed 42
```

#### 多任务训练
```bash
python train_smolvla.py \
    --benchmark libero_10 \
    --task_ids all \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4 \
    --device cuda:0 \
    --seed 42
```

#### 使用 WandB 监控
```bash
python train_smolvla.py \
    --benchmark libero_10 \
    --task_ids all \
    --batch_size 16 \
    --num_epochs 100 \
    --use_wandb \
    --wandb_project my-smolvla-project \
    --device cuda:0
```

### 2. 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--benchmark` | libero_10 | 数据集选择 |
| `--task_ids` | all | 任务ID（0,1,2 或 all） |
| `--batch_size` | 8 | 批次大小 |
| `--num_epochs` | 50 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--img_size` | 224 | 图像大小 |
| `--seq_len` | 10 | 动作序列长度 |
| `--device` | cuda:0 | 设备 |
| `--seed` | 42 | 随机种子 |

### 3. 评估

```bash
python evaluate_smolvla.py \
    --checkpoint ./smolvla_experiments/libero_10_smolvla_seed42/best_model.pth \
    --benchmark libero_10 \
    --task_id 0 \
    --num_episodes 20 \
    --save_videos \
    --device cuda:0
```

### 4. 高级配置

修改 `smolvla_libero/config.py`:
```python
@dataclass
class SmolVLAConfig:
    model_name: str = "HuggingFaceTB/SmolVLM-Instruct"
    action_dim: int = 7
    img_size: int = 224
    seq_len: int = 10
    
    # 冻结部分网络
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    
    # 动作预测
    action_normalization: bool = True
```

### 5. 内存优化

```bash
# 减小批次大小
python train_smolvla.py --batch_size 4 ...

# 使用梯度累积
python train_smolvla.py \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    ...

# 减小图像大小
python train_smolvla.py --img_size 128 ...
```

---

## 对比总结

### 架构对比

| 特性 | 原始 LIBERO | SmolVLA |
|------|-------------|---------|
| **模型基础** | 从头训练 | 预训练 VLM |
| **语言处理** | BERT 嵌入 | 端到端理解 |
| **参数量** | 10M-50M | 100M-1B+ |
| **训练时间** | 快 | 较慢 |
| **样本效率** | 中等 | 高 |
| **零样本能力** | 无 | 有潜力 |

### 使用场景

#### 选择原始 LIBERO 当你需要：
- ✅ 研究终身学习算法
- ✅ 对比不同的持续学习方法
- ✅ 复现官方基准结果
- ✅ 较小的模型和快速训练

#### 选择 SmolVLA 当你需要：
- ✅ 利用预训练模型的先验知识
- ✅ 端到端的自然语言理解
- ✅ 快速原型开发
- ✅ 研究迁移学习和泛化

### 性能对比（预期）

| 指标 | 原始 LIBERO | SmolVLA |
|------|-------------|---------|
| **训练速度** | ⚡⚡⚡ | ⚡⚡ |
| **内存占用** | 2-4GB | 8-16GB |
| **样本效率** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **泛化能力** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **易用性** | ⭐⭐ | ⭐⭐⭐⭐ |

---

## 完整示例

### 示例 1: 原始 LIBERO 完整流程

```bash
# 1. 下载数据
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_10 \
    --use-huggingface

# 2. 训练模型
python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_transformer_policy \
    lifelong=multitask

# 3. 评估模型
python libero/lifelong/evaluate.py \
    --benchmark libero_10 \
    --task_id 0 \
    --algo multitask \
    --policy bc_transformer_policy \
    --seed 42 \
    --ep 50 \
    --device_id 0 \
    --save-videos
```

### 示例 2: SmolVLA 完整流程

```bash
# 1. 测试环境
python test_smolvla_setup.py

# 2. 下载数据
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_10 \
    --use-huggingface

# 3. 训练模型
python train_smolvla.py \
    --benchmark libero_10 \
    --task_ids all \
    --batch_size 8 \
    --num_epochs 50 \
    --device cuda:0 \
    --seed 42

# 4. 评估模型
python evaluate_smolvla.py \
    --checkpoint ./smolvla_experiments/libero_10_smolvla_seed42/best_model.pth \
    --benchmark libero_10 \
    --task_id 0 \
    --save_videos \
    --device cuda:0
```

---

## 常见问题

### Q1: 如何选择方法？
- **研究终身学习** → 使用原始 LIBERO
- **快速原型/迁移学习** → 使用 SmolVLA
- **资源受限** → 使用原始 LIBERO
- **需要预训练优势** → 使用 SmolVLA

### Q2: 可以混合使用吗？
可以！两套代码完全独立，可以：
1. 用 SmolVLA 预训练
2. 用原始 LIBERO 做终身学习
3. 对比两种方法的结果

### Q3: 训练时间多长？
- **原始 LIBERO**: 单任务 ~30分钟，多任务 ~2-3小时
- **SmolVLA**: 单任务 ~1小时，多任务 ~5-10小时

### Q4: 需要多少 GPU 内存？
- **原始 LIBERO**: 4GB 足够
- **SmolVLA**: 推荐 16GB，最少 8GB

---

## 获取帮助

- **原始 LIBERO 文档**: https://lifelong-robot-learning.github.io/LIBERO/
- **原始 LIBERO GitHub**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **SmolVLA 示例**: 运行 `python example_usage.py`
- **环境测试**: 运行 `python test_smolvla_setup.py`

---

**祝训练顺利！** 🚀
