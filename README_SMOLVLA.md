# SmolVLA for LIBERO

在 LIBERO 机器人操作数据集上训练 SmolVLA 视觉-语言-动作模型。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements_smolvla.txt
```

### 2. 下载数据集
```bash
python benchmark_scripts/download_libero_datasets.py --datasets libero_10 --use-huggingface
```

### 3. 训练模型
```bash
python train_smolvla.py \
    --benchmark libero_10 \
    --task_ids all \
    --batch_size 8 \
    --num_epochs 50 \
    --device cuda:0
```

### 4. 评估模型
```bash
python evaluate_smolvla.py \
    --checkpoint ./smolvla_experiments/libero_10_smolvla_seed42/best_model.pth \
    --benchmark libero_10 \
    --task_id 0 \
    --save_videos
```

## 📚 文档

- **TRAINING_GUIDE.md** - 完整训练指南（包含原始 LIBERO 和 SmolVLA 两种方法）
- **example_usage.py** - 代码使用示例
- **test_smolvla_setup.py** - 环境测试脚本

## 📁 项目结构

```
.
├── train_smolvla.py              # 训练脚本
├── evaluate_smolvla.py           # 评估脚本
├── smolvla_libero/               # SmolVLA 包
│   ├── config.py                 # 配置
│   ├── dataset.py                # 数据加载
│   ├── model.py                  # 模型定义
│   └── trainer.py                # 训练器
├── TRAINING_GUIDE.md             # 训练指南
└── requirements_smolvla.txt      # 依赖
```

## 🎯 主要特点

- ✅ 基于预训练的视觉-语言模型
- ✅ 端到端自然语言理解
- ✅ 简单的命令行接口
- ✅ 支持多任务训练
- ✅ 完整的评估和可视化

## 💡 常用命令

### 训练
```bash
# 单任务
python train_smolvla.py --benchmark libero_10 --task_ids 0 --num_epochs 50

# 多任务
python train_smolvla.py --benchmark libero_10 --task_ids all --num_epochs 100

# 使用 WandB
python train_smolvla.py --benchmark libero_10 --task_ids all --use_wandb
```

### 评估
```bash
# 基础评估
python evaluate_smolvla.py --checkpoint <path> --benchmark libero_10 --task_id 0

# 保存视频
python evaluate_smolvla.py --checkpoint <path> --task_id 0 --save_videos
```

### 测试
```bash
# 测试环境
python test_smolvla_setup.py

# 运行示例
python example_usage.py
```

## 🔧 配置

修改 `smolvla_libero/config.py` 自定义模型配置：

```python
@dataclass
class SmolVLAConfig:
    model_name: str = "HuggingFaceTB/SmolVLM-Instruct"
    action_dim: int = 7
    img_size: int = 224
    seq_len: int = 10
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
```

## 📊 与原始 LIBERO 对比

| 特性 | 原始 LIBERO | SmolVLA |
|------|-------------|---------|
| 模型 | 从头训练 | 预训练 VLM |
| 语言 | BERT 嵌入 | 端到端理解 |
| 内存 | 2-4GB | 8-16GB |
| 样本效率 | 中等 | 高 |

详见 **TRAINING_GUIDE.md**

## 🐛 常见问题

### CUDA 内存不足
```bash
python train_smolvla.py --batch_size 4 --gradient_accumulation_steps 4
```

### 数据集未找到
```bash
python benchmark_scripts/download_libero_datasets.py --datasets libero_10 --use-huggingface
```

### 环境问题
```bash
python test_smolvla_setup.py
```

## 📄 许可证

MIT License - 与 LIBERO 保持一致

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**查看 TRAINING_GUIDE.md 了解完整使用方法** 📖
