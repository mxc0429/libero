# SmolVLA-LIBERO 项目结构

## 📁 文件组织（已优化）

```
LIBERO/
│
├── 📄 核心训练脚本
│   ├── train_smolvla.py              (8.6 KB)  - SmolVLA 训练脚本
│   ├── evaluate_smolvla.py           (9.2 KB)  - SmolVLA 评估脚本
│   ├── example_usage.py              (7.0 KB)  - 代码使用示例
│   └── test_smolvla_setup.py         (7.7 KB)  - 环境测试脚本
│
├── 📦 SmolVLA 包 (smolvla_libero/)
│   ├── __init__.py                   (395 B)   - 包初始化
│   ├── config.py                     (1.3 KB)  - 配置类
│   ├── dataset.py                    (6.9 KB)  - 数据加载器
│   ├── model.py                      (11 KB)   - SmolVLA 模型
│   └── trainer.py                    (11 KB)   - 训练器
│
├── 📖 文档
│   ├── README_SMOLVLA.md             (3.3 KB)  - SmolVLA 快速指南
│   ├── TRAINING_GUIDE.md             (8.9 KB)  - 完整训练指南
│   └── PROJECT_STRUCTURE.md          (本文件)  - 项目结构说明
│
├── ⚙️ 配置文件
│   ├── requirements_smolvla.txt      (586 B)   - Python 依赖
│   └── quick_start.sh                (1.2 KB)  - 快速启动脚本
│
└── 📂 原始 LIBERO 代码（不变）
    ├── libero/                                  - LIBERO 核心代码
    ├── benchmark_scripts/                       - 基准测试脚本
    ├── scripts/                                 - 工具脚本
    └── ...
```

## 📊 文件统计

- **核心脚本**: 4 个文件 (~33 KB)
- **SmolVLA 包**: 5 个文件 (~30 KB)
- **文档**: 3 个文件 (~13 KB)
- **配置**: 2 个文件 (~2 KB)
- **总计**: 14 个文件 (~78 KB)

## 🎯 文件用途

### 核心脚本

#### train_smolvla.py
- **用途**: SmolVLA 模型训练
- **功能**: 
  - 单任务/多任务训练
  - WandB 集成
  - 自动保存最佳模型
- **使用**: `python train_smolvla.py --benchmark libero_10 --task_ids all`

#### evaluate_smolvla.py
- **用途**: 模型评估
- **功能**:
  - 在仿真环境中测试
  - 计算成功率
  - 保存评估视频
- **使用**: `python evaluate_smolvla.py --checkpoint <path> --task_id 0 --save_videos`

#### example_usage.py
- **用途**: 代码示例
- **功能**:
  - 6 个完整示例
  - 演示所有主要 API
- **使用**: `python example_usage.py`

#### test_smolvla_setup.py
- **用途**: 环境测试
- **功能**:
  - 检查依赖安装
  - 验证 LIBERO 配置
  - 测试模型创建
- **使用**: `python test_smolvla_setup.py`

### SmolVLA 包

#### config.py
- **用途**: 配置管理
- **内容**: SmolVLAConfig 数据类
- **修改**: 调整模型超参数

#### dataset.py
- **用途**: 数据加载
- **内容**: 
  - LiberoSmolVLADataset - 单任务数据集
  - LiberoMultiTaskDataset - 多任务数据集
- **修改**: 自定义数据增强

#### model.py
- **用途**: 模型定义
- **内容**:
  - SmolVLAForLibero - 主模型
  - ActionHead - 动作预测头
- **修改**: 替换为其他 VLA 模型

#### trainer.py
- **用途**: 训练管理
- **内容**: SmolVLATrainer 类
- **修改**: 自定义训练循环

### 文档

#### README_SMOLVLA.md
- **用途**: 快速入门
- **内容**: 
  - 快速开始步骤
  - 常用命令
  - 常见问题
- **阅读**: 首先阅读此文件

#### TRAINING_GUIDE.md
- **用途**: 完整训练指南
- **内容**:
  - 原始 LIBERO 训练方法
  - SmolVLA 训练方法
  - 详细对比
- **阅读**: 了解两种训练方法

#### PROJECT_STRUCTURE.md
- **用途**: 项目结构说明
- **内容**: 本文件
- **阅读**: 了解项目组织

### 配置文件

#### requirements_smolvla.txt
- **用途**: Python 依赖列表
- **使用**: `pip install -r requirements_smolvla.txt`

#### quick_start.sh
- **用途**: 快速启动脚本
- **使用**: `bash quick_start.sh`

## 🔄 工作流程

### 新手流程
```
1. test_smolvla_setup.py     → 测试环境
2. README_SMOLVLA.md          → 快速了解
3. 下载数据集                 → 准备数据
4. train_smolvla.py           → 训练模型
5. evaluate_smolvla.py        → 评估模型
```

### 进阶流程
```
1. TRAINING_GUIDE.md          → 了解两种方法
2. example_usage.py           → 学习 API
3. 修改 config.py             → 自定义配置
4. 训练和评估                 → 实验
5. 修改 model.py/trainer.py   → 深度定制
```

## 🎨 设计原则

### 1. 零侵入
- ✅ 所有 SmolVLA 代码独立
- ✅ 不修改原始 LIBERO 代码
- ✅ 可以随时删除

### 2. 模块化
- ✅ 清晰的功能分离
- ✅ 易于维护和扩展
- ✅ 代码复用性高

### 3. 简洁性
- ✅ 只保留必要文件
- ✅ 文档精简实用
- ✅ 避免冗余

### 4. 易用性
- ✅ 简单的命令行接口
- ✅ 清晰的文档
- ✅ 完整的示例

## 📝 与原始 LIBERO 的关系

```
LIBERO (原始项目)
├── libero/                    ← 原始代码（不变）
├── benchmark_scripts/         ← 原始脚本（不变）
├── scripts/                   ← 原始工具（不变）
└── ...

SmolVLA 扩展（新增）
├── train_smolvla.py          ← 新增
├── evaluate_smolvla.py       ← 新增
├── smolvla_libero/           ← 新增
├── TRAINING_GUIDE.md         ← 新增
└── ...
```

**关键点**:
- SmolVLA 代码完全独立
- 不影响原始 LIBERO 功能
- 可以同时使用两种方法
- 可以随时删除 SmolVLA 部分

## 🔧 自定义指南

### 修改模型
编辑 `smolvla_libero/model.py`:
```python
# 替换为其他 VLA 模型
from openvla import OpenVLA
self.base_model = OpenVLA.from_pretrained("openvla-7b")
```

### 修改数据处理
编辑 `smolvla_libero/dataset.py`:
```python
# 添加数据增强
self.image_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 新增
    ...
])
```

### 修改训练流程
编辑 `smolvla_libero/trainer.py`:
```python
# 自定义训练步骤
def train_step(self, batch):
    # 你的自定义逻辑
    ...
```

### 修改配置
编辑 `smolvla_libero/config.py`:
```python
@dataclass
class SmolVLAConfig:
    # 修改默认值
    batch_size: int = 16
    learning_rate: float = 5e-5
    ...
```

## 📦 依赖关系

```
train_smolvla.py
    ↓
smolvla_libero/
    ├── config.py
    ├── dataset.py  → LIBERO (读取 HDF5)
    ├── model.py    → transformers (SmolVLM)
    └── trainer.py  → wandb (可选)

evaluate_smolvla.py
    ↓
smolvla_libero/model.py
    ↓
LIBERO (仿真环境)
```

## 🚀 快速命令参考

```bash
# 测试
python test_smolvla_setup.py

# 训练
python train_smolvla.py --benchmark libero_10 --task_ids all

# 评估
python evaluate_smolvla.py --checkpoint <path> --task_id 0 --save_videos

# 示例
python example_usage.py

# 快速启动
bash quick_start.sh
```

## 📚 推荐阅读顺序

1. **README_SMOLVLA.md** - 快速了解项目
2. **TRAINING_GUIDE.md** - 学习如何训练
3. **PROJECT_STRUCTURE.md** (本文件) - 理解项目结构
4. **example_usage.py** - 查看代码示例

---

**项目结构已优化，保持简洁高效！** ✨
