"""
修复版 LIBERO 数据集下载脚本
支持从 HuggingFace 下载所有数据集，包括 libero_10 和 libero_90
"""
import argparse
import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from libero.libero import get_libero_path


def download_from_huggingface(dataset_name, download_dir):
    """从 HuggingFace 下载指定数据集"""
    
    print(f"\n{'='*60}")
    print(f"下载 {dataset_name}")
    print(f"{'='*60}")
    
    # 检查是否已存在
    dataset_dir = os.path.join(download_dir, dataset_name)
    if os.path.exists(dataset_dir):
        file_count = len(list(Path(dataset_dir).glob("*.hdf5")))
        print(f"⚠️  {dataset_name} 已存在 ({file_count} 个文件)")
        response = input("是否重新下载？(y/n): ")
        if response.lower() != 'y':
            print(f"跳过 {dataset_name}")
            return True
        
        # 删除现有目录
        import shutil
        print(f"删除现有目录: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    
    try:
        # 从 HuggingFace 下载
        print(f"正在从 HuggingFace 下载 {dataset_name}...")
        snapshot_download(
            repo_id="yifengzhu-hf/LIBERO-datasets",
            repo_type="dataset",
            local_dir=download_dir,
            allow_patterns=f"{dataset_name}/*",
            local_dir_use_symlinks=False,
        )
        
        # 验证下载
        if os.path.exists(dataset_dir):
            file_count = len(list(Path(dataset_dir).glob("*.hdf5")))
            print(f"✅ {dataset_name} 下载成功！({file_count} 个文件)")
            return True
        else:
            print(f"❌ {dataset_name} 下载失败：目录不存在")
            return False
            
    except Exception as e:
        print(f"❌ {dataset_name} 下载失败: {e}")
        return False


def check_datasets(download_dir):
    """检查所有数据集的完整性"""
    
    print(f"\n{'='*60}")
    print("检查数据集完整性")
    print(f"{'='*60}")
    
    datasets_info = {
        "libero_object": 10,
        "libero_goal": 10,
        "libero_spatial": 10,
        "libero_10": 10,
        "libero_90": 90,
    }
    
    all_complete = True
    
    for dataset_name, expected_count in datasets_info.items():
        dataset_dir = os.path.join(download_dir, dataset_name)
        
        if os.path.exists(dataset_dir):
            file_count = len(list(Path(dataset_dir).glob("*.hdf5")))
            
            if file_count == expected_count:
                print(f"✅ {dataset_name:20s} 完整 ({file_count}/{expected_count} 个文件)")
            else:
                print(f"⚠️  {dataset_name:20s} 不完整 ({file_count}/{expected_count} 个文件)")
                all_complete = False
        else:
            print(f"❌ {dataset_name:20s} 未找到")
            all_complete = False
    
    return all_complete


def main():
    parser = argparse.ArgumentParser(description="从 HuggingFace 下载 LIBERO 数据集")
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="下载目录（默认使用 LIBERO 配置的路径）"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        choices=["all", "libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"],
        help="要下载的数据集"
    )
    
    args = parser.parse_args()
    
    # 获取下载目录
    if args.download_dir is None:
        download_dir = get_libero_path("datasets")
    else:
        download_dir = args.download_dir
    
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("LIBERO 数据集下载工具（修复版）")
    print(f"{'='*60}")
    print(f"下载目录: {download_dir}")
    print(f"数据源: HuggingFace (yifengzhu-hf/LIBERO-datasets)")
    print()
    
    # 确定要下载的数据集
    if args.datasets == "all":
        datasets_to_download = [
            "libero_10",
            "libero_90", 
            "libero_spatial",
            "libero_object",
            "libero_goal"
        ]
        print("将下载所有数据集:")
        for ds in datasets_to_download:
            print(f"  - {ds}")
    else:
        datasets_to_download = [args.datasets]
        print(f"将下载: {args.datasets}")
    
    print()
    
    # 下载数据集
    success_count = 0
    for dataset_name in datasets_to_download:
        if download_from_huggingface(dataset_name, download_dir):
            success_count += 1
        time.sleep(1)  # 避免请求过快
    
    # 检查完整性
    print()
    all_complete = check_datasets(download_dir)
    
    # 总结
    print(f"\n{'='*60}")
    print("下载总结")
    print(f"{'='*60}")
    print(f"成功下载: {success_count}/{len(datasets_to_download)} 个数据集")
    
    if all_complete:
        print("✅ 所有数据集完整！")
    else:
        print("⚠️  部分数据集不完整，请检查上面的信息")
    
    print(f"\n数据集位置: {download_dir}")
    print("\n现在可以开始训练:")
    print("  python train_smolvla.py --benchmark libero_10 --task_ids all")


if __name__ == "__main__":
    main()
