"""
从 LIBERO 数据集创建视频
"""

import h5py
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse


def create_video_from_demo(hdf5_path, demo_id=0, output_dir="videos", fps=30, view="agentview_rgb"):
    """
    从演示数据创建视频
    
    参数:
        hdf5_path: HDF5 文件路径
        demo_id: 演示 ID
        output_dir: 输出目录
        fps: 视频帧率
        view: 视角名称 (agentview_rgb 或 eye_in_hand_rgb)
    """
    
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        print(f"错误: 文件不存在: {hdf5_path}")
        return False
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n正在处理: {hdf5_path.name}")
    print(f"演示 ID: {demo_id}")
    print(f"视角: {view}")
    print("=" * 80)
    
    try:
        with h5py.File(hdf5_path, "r") as f:
            demo_key = f"demo_{demo_id}"
            
            if demo_key not in f["data"]:
                print(f"错误: {demo_key} 不存在")
                return False
            
            demo = f["data"][demo_key]
            
            if "obs" not in demo:
                print("错误: 该演示中没有观测数据")
                return False
            
            obs = demo["obs"]
            
            if view not in obs:
                available_views = [k for k in obs.keys() if "rgb" in k.lower()]
                print(f"错误: 视角 '{view}' 不存在")
                print(f"可用视角: {available_views}")
                return False
            
            # 读取图像数据
            images = obs[view][()]
            print(f"图像形状: {images.shape}")
            print(f"总帧数: {len(images)}")
            
            if len(images) == 0:
                print("错误: 没有图像数据")
                return False
            
            # 获取图像尺寸
            h, w = images[0].shape[:2]
            
            # 创建视频文件名
            video_name = f"{hdf5_path.stem}_demo{demo_id}_{view}.mp4"
            video_path = output_dir / video_name
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            
            if not out.isOpened():
                print("错误: 无法创建视频文件")
                return False
            
            # 写入每一帧
            print(f"正在生成视频 (FPS={fps})...")
            for i, img in enumerate(images):
                # 确保是 BGR 格式（OpenCV 默认）
                if img.shape[-1] == 3:
                    out.write(img)
                
                # 显示进度
                if (i + 1) % 50 == 0 or i == len(images) - 1:
                    progress = (i + 1) / len(images) * 100
                    print(f"  进度: {i+1}/{len(images)} ({progress:.1f}%)")
            
            out.release()
            
            # 获取文件大小
            file_size = video_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"\n✓ 视频已保存到: {video_path}")
            print(f"  文件大小: {file_size:.2f} MB")
            print(f"  时长: {len(images) / fps:.2f} 秒")
            print("=" * 80)
            
            return True
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_side_by_side_video(hdf5_path, demo_id=0, output_dir="videos", fps=30):
    """
    创建并排显示两个视角的视频
    """
    
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        print(f"错误: 文件不存在: {hdf5_path}")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n正在创建并排视频: {hdf5_path.name}")
    print(f"演示 ID: {demo_id}")
    print("=" * 80)
    
    try:
        with h5py.File(hdf5_path, "r") as f:
            demo_key = f"demo_{demo_id}"
            
            if demo_key not in f["data"]:
                print(f"错误: {demo_key} 不存在")
                return False
            
            demo = f["data"][demo_key]
            
            if "obs" not in demo:
                print("错误: 该演示中没有观测数据")
                return False
            
            obs = demo["obs"]
            
            # 检查两个视角是否都存在
            if "agentview_rgb" not in obs or "eye_in_hand_rgb" not in obs:
                print("错误: 需要两个视角的图像")
                return False
            
            # 读取两个视角的图像
            images1 = obs["agentview_rgb"][()]
            images2 = obs["eye_in_hand_rgb"][()]
            
            print(f"视角1 (agentview) 形状: {images1.shape}")
            print(f"视角2 (eye_in_hand) 形状: {images2.shape}")
            print(f"总帧数: {len(images1)}")
            
            # 确保两个视角的帧数相同
            min_frames = min(len(images1), len(images2))
            images1 = images1[:min_frames]
            images2 = images2[:min_frames]
            
            # 获取图像尺寸
            h, w = images1[0].shape[:2]
            
            # 创建视频文件名
            video_name = f"{hdf5_path.stem}_demo{demo_id}_side_by_side.mp4"
            video_path = output_dir / video_name
            
            # 创建视频写入器（宽度是两倍）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (w * 2, h))
            
            if not out.isOpened():
                print("错误: 无法创建视频文件")
                return False
            
            # 写入每一帧
            print(f"正在生成并排视频 (FPS={fps})...")
            for i in range(min_frames):
                # 将两个视角并排放置
                combined = np.hstack([images1[i], images2[i]])
                out.write(combined)
                
                # 显示进度
                if (i + 1) % 50 == 0 or i == min_frames - 1:
                    progress = (i + 1) / min_frames * 100
                    print(f"  进度: {i+1}/{min_frames} ({progress:.1f}%)")
            
            out.release()
            
            # 获取文件大小
            file_size = video_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"\n✓ 并排视频已保存到: {video_path}")
            print(f"  文件大小: {file_size:.2f} MB")
            print(f"  时长: {min_frames / fps:.2f} 秒")
            print("=" * 80)
            
            return True
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="从 LIBERO 数据集创建视频")
    parser.add_argument(
        "hdf5_path",
        type=str,
        help="HDF5 文件路径"
    )
    parser.add_argument(
        "--demo_id",
        type=int,
        default=0,
        help="演示 ID (默认: 0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="videos",
        help="输出目录 (默认: videos)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率 (默认: 30)"
    )
    parser.add_argument(
        "--view",
        type=str,
        default="agentview_rgb",
        choices=["agentview_rgb", "eye_in_hand_rgb"],
        help="视角名称 (默认: agentview_rgb)"
    )
    parser.add_argument(
        "--side_by_side",
        action="store_true",
        help="创建并排显示两个视角的视频"
    )
    parser.add_argument(
        "--all_views",
        action="store_true",
        help="为所有视角创建视频"
    )
    
    args = parser.parse_args()
    
    if args.side_by_side:
        # 创建并排视频
        success = create_side_by_side_video(
            args.hdf5_path,
            args.demo_id,
            args.output_dir,
            args.fps
        )
    elif args.all_views:
        # 为所有视角创建视频
        views = ["agentview_rgb", "eye_in_hand_rgb"]
        success = True
        for view in views:
            result = create_video_from_demo(
                args.hdf5_path,
                args.demo_id,
                args.output_dir,
                args.fps,
                view
            )
            success = success and result
    else:
        # 创建单个视角的视频
        success = create_video_from_demo(
            args.hdf5_path,
            args.demo_id,
            args.output_dir,
            args.fps,
            args.view
        )
    
    if success:
        print("\n✓ 所有视频创建成功！")
    else:
        print("\n✗ 视频创建失败")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("使用方法:")
        print("  python3 create_video.py <hdf5文件路径> [选项]")
        print("\n示例:")
        print("  # 创建单个视角的视频")
        print("  python3 create_video.py ./libero/datasets/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5")
        print("\n  # 创建并排视频")
        print("  python3 create_video.py ./libero/datasets/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5 --side_by_side")
        print("\n  # 创建所有视角的视频")
        print("  python3 create_video.py ./libero/datasets/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5 --all_views")
        print("\n  # 指定演示 ID 和帧率")
        print("  python3 create_video.py ./libero/datasets/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5 --demo_id 2 --fps 60")
        print("\n选项:")
        print("  --demo_id ID        演示 ID (默认: 0)")
        print("  --output_dir DIR    输出目录 (默认: videos)")
        print("  --fps FPS           视频帧率 (默认: 30)")
        print("  --view VIEW         视角 (agentview_rgb 或 eye_in_hand_rgb)")
        print("  --side_by_side      创建并排视频")
        print("  --all_views         为所有视角创建视频")
        sys.exit(1)
    
    main()
