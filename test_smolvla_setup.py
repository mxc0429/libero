"""
测试 SmolVLA-LIBERO 设置是否正确
"""

import sys
import importlib


def test_imports():
    """测试必要的包是否可以导入"""
    print("=" * 60)
    print("测试 1: 检查依赖包")
    print("=" * 60)
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("h5py", "HDF5"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {name:20s} - 已安装")
        except ImportError:
            print(f"✗ {name:20s} - 未安装")
            all_ok = False
    
    return all_ok


def test_libero():
    """测试 LIBERO 是否正确安装"""
    print("\n" + "=" * 60)
    print("测试 2: 检查 LIBERO")
    print("=" * 60)
    
    try:
        import libero
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        print("✓ LIBERO 已正确安装")
        
        # 尝试获取路径
        try:
            datasets_path = get_libero_path("datasets")
            print(f"✓ 数据集路径: {datasets_path}")
        except Exception as e:
            print(f"⚠ 获取数据集路径失败: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ LIBERO 未安装: {e}")
        return False


def test_smolvla_package():
    """测试 SmolVLA 包是否可以导入"""
    print("\n" + "=" * 60)
    print("测试 3: 检查 SmolVLA 包")
    print("=" * 60)
    
    try:
        from smolvla_libero import (
            LiberoSmolVLADataset,
            SmolVLAForLibero,
            SmolVLATrainer,
            SmolVLAConfig
        )
        print("✓ SmolVLA 包可以导入")
        
        # 测试创建配置
        config = SmolVLAConfig(device="cpu")
        print(f"✓ 配置创建成功")
        print(f"  - 动作维度: {config.action_dim}")
        print(f"  - 图像大小: {config.img_size}")
        print(f"  - 序列长度: {config.seq_len}")
        
        return True
    except Exception as e:
        print(f"✗ SmolVLA 包导入失败: {e}")
        return False


def test_model_creation():
    """测试模型是否可以创建"""
    print("\n" + "=" * 60)
    print("测试 4: 创建模型")
    print("=" * 60)
    
    try:
        from smolvla_libero import SmolVLAForLibero, SmolVLAConfig
        import torch
        
        config = SmolVLAConfig(
            model_name="HuggingFaceTB/SmolVLM-Instruct",
            device="cpu"
        )
        
        print("正在创建模型（可能需要下载，请稍候）...")
        model = SmolVLAForLibero(config)
        
        print(f"✓ 模型创建成功")
        print(f"  - 参数量: {model.count_parameters():,}")
        print(f"  - 设备: {config.device}")
        
        # 测试前向传播
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        text_inputs = {"input_ids": torch.randint(0, 1000, (batch_size, 10))}
        actions = torch.randn(batch_size, 10, 7)
        
        outputs = model(images, text_inputs, actions)
        print(f"✓ 前向传播成功")
        print(f"  - 预测动作形状: {outputs['predicted_actions'].shape}")
        print(f"  - 损失: {outputs['loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """测试数据集加载（如果数据存在）"""
    print("\n" + "=" * 60)
    print("测试 5: 数据集加载（可选）")
    print("=" * 60)
    
    try:
        from libero.libero import get_libero_path
        import os
        
        datasets_path = get_libero_path("datasets")
        
        # 查找任意一个 demo 文件
        demo_file = None
        for root, dirs, files in os.walk(datasets_path):
            for file in files:
                if file.endswith("_demo.hdf5"):
                    demo_file = os.path.join(root, file)
                    break
            if demo_file:
                break
        
        if demo_file is None:
            print("⚠ 未找到数据集文件（这是正常的，如果你还没下载数据）")
            print("  运行以下命令下载数据:")
            print("  python benchmark_scripts/download_libero_datasets.py --datasets libero_10")
            return True
        
        print(f"找到数据文件: {os.path.basename(demo_file)}")
        
        from smolvla_libero import LiberoSmolVLADataset
        
        dataset = LiberoSmolVLADataset(
            hdf5_path=demo_file,
            task_description="test task",
            img_size=224,
            seq_len=10,
            train=True
        )
        
        print(f"✓ 数据集加载成功")
        print(f"  - 样本数量: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"✓ 样本获取成功")
        print(f"  - 图像形状: {sample['image'].shape}")
        print(f"  - 动作形状: {sample['actions'].shape}")
        
        return True
    except Exception as e:
        print(f"⚠ 数据集测试跳过: {e}")
        return True  # 不算失败


def test_cuda():
    """测试 CUDA 是否可用"""
    print("\n" + "=" * 60)
    print("测试 6: CUDA 支持")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  - CUDA 版本: {torch.version.cuda}")
            print(f"  - GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA 不可用（将使用 CPU 训练，速度较慢）")
        
        return True
    except Exception as e:
        print(f"✗ CUDA 检查失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" " * 15 + "SmolVLA-LIBERO 环境测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("依赖包", test_imports()))
    results.append(("LIBERO", test_libero()))
    results.append(("SmolVLA 包", test_smolvla_package()))
    results.append(("模型创建", test_model_creation()))
    results.append(("数据集加载", test_dataset_loading()))
    results.append(("CUDA 支持", test_cuda()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！环境配置正确。")
        print("\n下一步:")
        print("1. 下载数据集:")
        print("   python benchmark_scripts/download_libero_datasets.py --datasets libero_10")
        print("\n2. 运行示例:")
        print("   python example_usage.py")
        print("\n3. 开始训练:")
        print("   python train_smolvla.py --benchmark libero_10 --task_ids 0")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
        print("\n常见解决方案:")
        print("1. 安装缺失的包:")
        print("   pip install -r requirements_smolvla.txt")
        print("\n2. 安装 LIBERO:")
        print("   pip install -e .")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
