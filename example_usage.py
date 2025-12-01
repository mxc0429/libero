"""
示例：如何使用 SmolVLA 进行训练和推理
"""

import torch
from smolvla_libero.config import SmolVLAConfig
from smolvla_libero.model import SmolVLAForLibero
from smolvla_libero.dataset import LiberoSmolVLADataset
from torch.utils.data import DataLoader


def example_1_create_model():
    """示例 1: 创建 SmolVLA 模型"""
    print("=" * 60)
    print("示例 1: 创建 SmolVLA 模型")
    print("=" * 60)
    
    # 创建配置
    config = SmolVLAConfig(
        model_name="HuggingFaceTB/SmolVLM-Instruct",
        action_dim=7,
        img_size=224,
        seq_len=10,
        device="cuda:0"
    )
    
    # 创建模型
    model = SmolVLAForLibero(config)
    print(f"模型参数量: {model.count_parameters():,}")
    
    return model


def example_2_load_dataset():
    """示例 2: 加载 LIBERO 数据集"""
    print("\n" + "=" * 60)
    print("示例 2: 加载 LIBERO 数据集")
    print("=" * 60)
    
    # 创建数据集
    dataset = LiberoSmolVLADataset(
        hdf5_path="datasets/libero_10/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet_demo.hdf5",
        task_description="put the black bowl on top of the cabinet",
        img_size=224,
        seq_len=10,
        train=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"图像形状: {sample['image'].shape}")
    print(f"动作形状: {sample['actions'].shape}")
    print(f"任务描述: {sample['text']}")
    
    # 获取动作统计信息
    stats = dataset.get_action_stats()
    print(f"动作均值: {stats['mean']}")
    print(f"动作标准差: {stats['std']}")
    
    return dataset


def example_3_forward_pass():
    """示例 3: 模型前向传播"""
    print("\n" + "=" * 60)
    print("示例 3: 模型前向传播")
    print("=" * 60)
    
    # 创建模型
    config = SmolVLAConfig(
        model_name="HuggingFaceTB/SmolVLM-Instruct",
        action_dim=7,
        img_size=224,
        seq_len=10,
        device="cpu"  # 使用 CPU 进行演示
    )
    model = SmolVLAForLibero(config)
    
    # 创建虚拟输入
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    if model.processor is not None:
        text_inputs = model.processor(
            text=["pick up the red block", "put the bowl on the table"],
            return_tensors="pt",
            padding=True
        )
    else:
        # 占位符
        text_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 10))
        }
    
    actions = torch.randn(batch_size, 10, 7)
    
    # 前向传播
    outputs = model(images, text_inputs, actions)
    
    print(f"预测动作形状: {outputs['predicted_actions'].shape}")
    print(f"损失: {outputs['loss'].item():.4f}")


def example_4_training_loop():
    """示例 4: 简单的训练循环"""
    print("\n" + "=" * 60)
    print("示例 4: 简单的训练循环")
    print("=" * 60)
    
    # 创建模型和优化器
    config = SmolVLAConfig(
        model_name="HuggingFaceTB/SmolVLM-Instruct",
        action_dim=7,
        device="cpu"
    )
    model = SmolVLAForLibero(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 创建虚拟数据加载器
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10
        
        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 224, 224),
                "text": "pick up the object",
                "actions": torch.randn(10, 7),
                "action_mask": torch.ones(10, dtype=torch.bool)
            }
    
    dataloader = DataLoader(DummyDataset(), batch_size=2)
    
    # 训练几步
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= 3:  # 只演示 3 步
            break
        
        images = batch["image"]
        actions = batch["actions"]
        texts = batch["text"]
        
        # 准备文本输入
        if model.processor is not None:
            text_inputs = model.processor(
                text=texts,
                return_tensors="pt",
                padding=True
            )
        else:
            text_inputs = {
                "input_ids": torch.randint(0, 1000, (len(texts), 10))
            }
        
        # 前向传播
        outputs = model(images, text_inputs, actions)
        loss = outputs["loss"]
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"步骤 {step + 1}, 损失: {loss.item():.4f}")


def example_5_save_and_load():
    """示例 5: 保存和加载模型"""
    print("\n" + "=" * 60)
    print("示例 5: 保存和加载模型")
    print("=" * 60)
    
    # 创建模型
    config = SmolVLAConfig(
        model_name="HuggingFaceTB/SmolVLM-Instruct",
        action_dim=7,
        device="cpu"
    )
    model = SmolVLAForLibero(config)
    
    # 保存模型
    save_path = "example_model.pth"
    model.save_pretrained(save_path)
    print(f"模型已保存到: {save_path}")
    
    # 加载模型
    model_loaded = SmolVLAForLibero(config)
    model_loaded.load_pretrained(save_path)
    print(f"模型已从 {save_path} 加载")
    
    # 清理
    import os
    os.remove(save_path)
    print("示例文件已清理")


def example_6_inference():
    """示例 6: 推理（预测动作）"""
    print("\n" + "=" * 60)
    print("示例 6: 推理（预测动作）")
    print("=" * 60)
    
    # 创建模型
    config = SmolVLAConfig(
        model_name="HuggingFaceTB/SmolVLM-Instruct",
        action_dim=7,
        device="cpu"
    )
    model = SmolVLAForLibero(config)
    model.eval()
    
    # 创建虚拟图像
    from PIL import Image
    import numpy as np
    
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    
    # 预测动作
    action = model.predict_action(
        image=dummy_image,
        text="pick up the red block"
    )
    
    print(f"预测的动作: {action}")
    print(f"动作维度: {action.shape}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print(" " * 20 + "SmolVLA-LIBERO 使用示例")
    print("=" * 80)
    
    try:
        # 示例 1: 创建模型
        model = example_1_create_model()
        
        # 示例 2: 加载数据集（需要实际的数据文件，这里跳过）
        # dataset = example_2_load_dataset()
        
        # 示例 3: 前向传播
        example_3_forward_pass()
        
        # 示例 4: 训练循环
        example_4_training_loop()
        
        # 示例 5: 保存和加载
        example_5_save_and_load()
        
        # 示例 6: 推理
        example_6_inference()
        
        print("\n" + "=" * 80)
        print(" " * 25 + "所有示例运行完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("某些示例可能需要实际的数据文件或 GPU 才能运行")


if __name__ == "__main__":
    main()
