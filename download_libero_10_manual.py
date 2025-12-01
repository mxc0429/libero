"""
从 HuggingFace 下载 libero_10 和 libero_90
"""
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from libero.libero import get_libero_path


def download_dataset(dataset_name):
    """从 HuggingFace 下载指定数据集"""
    
    datasets_dir = get_libero_path("datasets")
    
    print(f"从 HuggingFace 下载 {dataset_name}...")
    print(f"下载目录: {datasets_dir}")
    
    try:
        # 下载指定数据集文件夹
        snapshot_download(
            repo_id="yifengzhu-hf/LIBERO-datasets",
            repo_type="dataset",
            local_dir=datasets_dir,
            allow_patterns=f"{dataset_name}/*",
            local_dir_use_symlinks=False,
        )
        print(f"✅ {dataset_name} 下载成功！")
        
        # 检查文件数量
        dataset_dir = os.path.join(datasets_dir, dataset_name)
        if os.path.exists(dataset_dir):
            file_count = len(list(Path(dataset_dir).glob("*.hdf5")))
            print(f"找到 {file_count} 个 HDF5 文件")
            
            # 验证文件数量
            expected_count = 90 if dataset_name == "libero_90" else 10
            if file_count == expected_count:
                print(f"✅ {dataset_name} 完整！")
            else:
                print(f"⚠️ 预期 {expected_count} 个文件，但只找到 {file_count} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="从 HuggingFace 下载 LIBERO 数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_90", "both"],
        help="要下载的数据集"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("从 HuggingFace 下载 LIBERO 数据集")
    print("=" * 60)
    print()
    
    if args.dataset == "both":
        print("下载 libero_10 和 libero_90...")
        success_10 = download_dataset("libero_10")
        print()
        success_90 = download_dataset("libero_90")
        
        if success_10 and success_90:
            print("\n✅ 两个数据集都下载成功！")
        else:
            print("\n⚠️ 部分数据集下载失败")
    else:
        success = download_dataset(args.dataset)
        if success:
            print(f"\n✅ {args.dataset} 下载完成！")
        else:
            print(f"\n❌ {args.dataset} 下载失败")
    
    print("\n" + "=" * 60)
    print("验证所有数据集...")
    print("=" * 60)
    from libero.libero.utils.download_utils import check_libero_dataset
    check_libero_dataset()


if __name__ == "__main__":
    main()
