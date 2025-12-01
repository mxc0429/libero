"""
将 libero_100 分离为 libero_90 和 libero_10
"""
import os
import shutil
from pathlib import Path
from libero.libero import get_libero_path

def split_libero_100():
    """将 libero_100 分离为 libero_90 和 libero_10"""
    
    datasets_dir = get_libero_path("datasets")
    libero_100_dir = os.path.join(datasets_dir, "libero_100")
    libero_90_dir = os.path.join(datasets_dir, "libero_90")
    libero_10_dir = os.path.join(datasets_dir, "libero_10")
    
    # 检查 libero_100 是否存在
    if not os.path.exists(libero_100_dir):
        print(f"[错误] libero_100 目录不存在: {libero_100_dir}")
        print("请先运行: python benchmark_scripts/download_libero_datasets.py --datasets libero_100 --use-huggingface")
        return False
    
    # 获取所有 hdf5 文件
    hdf5_files = sorted(list(Path(libero_100_dir).glob("*.hdf5")))
    
    if len(hdf5_files) != 100:
        print(f"[警告] libero_100 应该有100个文件，但只找到 {len(hdf5_files)} 个")
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    
    # 创建目录
    os.makedirs(libero_90_dir, exist_ok=True)
    os.makedirs(libero_10_dir, exist_ok=True)
    
    # 前90个文件 -> libero_90
    print("\n复制前90个文件到 libero_90...")
    for i, file_path in enumerate(hdf5_files[:90]):
        dest = os.path.join(libero_90_dir, file_path.name)
        if not os.path.exists(dest):
            shutil.copy2(file_path, dest)
            if (i + 1) % 10 == 0:
                print(f"  已复制 {i + 1}/90 个文件")
    
    # 后10个文件 -> libero_10
    print("\n复制后10个文件到 libero_10...")
    for i, file_path in enumerate(hdf5_files[90:]):
        dest = os.path.join(libero_10_dir, file_path.name)
        if not os.path.exists(dest):
            shutil.copy2(file_path, dest)
        print(f"  已复制 {i + 1}/10 个文件")
    
    print("\n✅ 分离完成！")
    print(f"libero_90: {libero_90_dir} ({len(list(Path(libero_90_dir).glob('*.hdf5')))} 个文件)")
    print(f"libero_10: {libero_10_dir} ({len(list(Path(libero_10_dir).glob('*.hdf5')))} 个文件)")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("将 libero_100 分离为 libero_90 和 libero_10")
    print("=" * 60)
    
    success = split_libero_100()
    
    if success:
        print("\n现在你可以使用:")
        print("  --benchmark libero_10")
        print("  --benchmark libero_90")
    else:
        print("\n分离失败，请检查错误信息")
