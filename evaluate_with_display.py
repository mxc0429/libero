"""
支持实时显示的评估脚本
基于 libero/lifelong/evaluate.py 修改
"""

import argparse
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import numpy as np
import time
import torch
import cv2
from pathlib import Path
from easydict import EasyDict

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.time_utils import Timer
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, GroupedTaskDataset
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
)
from libero.lifelong.main import get_task_embs


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script with Display")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--display", action="store_true", help="实时显示评估过程")
    parser.add_argument("--display_view", type=str, default="agentview_rgb", 
                       choices=["agentview_rgb", "eye_in_hand_rgb"],
                       help="显示的视角")
    parser.add_argument("--n_eval", type=int, default=20, help="评估次数")
    parser.add_argument("--max_steps", type=int, default=600, help="每次评估的最大步数")
    
    args = parser.parse_args()
    args.device_id = f"cuda:{args.device_id}"
    
    if args.algo == "multitask":
        assert args.ep in list(range(0, 50, 5)), "[error] ep should be in [0, 5, ..., 50]"
    else:
        assert args.load_task in list(range(10)), "[error] load_task should be in [0, ..., 9]"
    
    return args


def evaluate_with_display(cfg, algo, task, task_emb, task_id, args):
    """
    评估单个任务并实时显示
    """
    
    env_args = {
        "bddl_file_name": os.path.join(
            cfg.bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }
    
    # 创建单个环境（用于显示）
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    env.seed(cfg.seed)
    algo.reset()
    
    # 加载初始状态
    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path)
    
    # 统计成功次数
    num_success = 0
    
    # 如果启用显示，创建窗口
    if args.display:
        window_name = f"Task {task_id}: {task.language}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 512, 512)
    
    print(f"\n开始评估任务 {task_id}: {task.language}")
    print(f"评估次数: {args.n_eval}")
    print(f"最大步数: {args.max_steps}")
    if args.display:
        print(f"实时显示: 启用 (视角: {args.display_view})")
        print("按 'q' 键退出显示")
    print("=" * 80)
    
    for eval_idx in range(args.n_eval):
        # 重置环境
        init_state_idx = eval_idx % init_states.shape[0]
        obs = env.set_init_state(init_states[init_state_idx])
        
        # 模拟物理（无动作）
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))
        
        done = False
        steps = 0
        
        print(f"\n评估 {eval_idx + 1}/{args.n_eval}:", end=" ", flush=True)
        
        with torch.no_grad():
            while steps < args.max_steps and not done:
                steps += 1
                
                # 获取动作
                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                action = algo.policy.get_action(data)
                
                # 执行动作
                obs, reward, done, info = env.step(action)
                
                # 实时显示
                if args.display:
                    img = obs[args.display_view]
                    # BGR to RGB for display
                    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 添加信息文本
                    img_display = img_display.copy()
                    cv2.putText(img_display, f"Step: {steps}/{args.max_steps}", 
                               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(img_display, f"Eval: {eval_idx + 1}/{args.n_eval}", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(img_display, f"Success: {num_success}/{eval_idx}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow(window_name, img_display)
                    
                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n用户中断评估")
                        env.close()
                        if args.display:
                            cv2.destroyAllWindows()
                        return num_success / (eval_idx + 1) if eval_idx > 0 else 0.0
                
                if done:
                    break
        
        # 检查是否成功
        if done:
            num_success += 1
            print(f"✓ 成功 (步数: {steps})")
        else:
            print(f"✗ 失败 (超时)")
        
        # 显示当前成功率
        current_success_rate = num_success / (eval_idx + 1)
        print(f"   当前成功率: {current_success_rate:.2%} ({num_success}/{eval_idx + 1})")
    
    env.close()
    if args.display:
        cv2.destroyAllWindows()
    
    success_rate = num_success / args.n_eval
    
    print("\n" + "=" * 80)
    print(f"最终成功率: {success_rate:.2%} ({num_success}/{args.n_eval})")
    print("=" * 80)
    
    return success_rate


def main():
    args = parse_args()
    
    # 查找实验目录
    experiment_dir = os.path.join(
        args.experiment_dir,
        f"{benchmark_map[args.benchmark]}/"
        + f"{algo_map[args.algo]}/"
        + f"{policy_map[args.policy]}_seed{args.seed}",
    )
    
    # 查找最新的运行
    experiment_id = 0
    for path in Path(experiment_dir).glob("run_*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run_")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    
    if experiment_id == 0:
        print(f"[error] 在 {experiment_dir} 下找不到检查点")
        sys.exit(0)
    
    run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
    
    # 加载模型
    try:
        if args.algo == "multitask":
            model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
        else:
            model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")
        
        sd, cfg, previous_mask = torch_load_model(model_path, map_location=args.device_id)
        print(f"[info] 从 {model_path} 加载模型")
    except:
        print(f"[error] 在 {str(model_path)} 找不到检查点")
        sys.exit(0)
    
    # 设置路径
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.device = args.device_id
    
    # 覆盖评估参数
    cfg.eval.n_eval = args.n_eval
    cfg.eval.max_steps = args.max_steps
    
    # 创建算法
    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask
    algo.policy.load_state_dict(sd)
    algo.eval()
    
    # 获取基准测试
    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0
    
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    
    task = benchmark.get_task(args.task_id)
    task_emb = benchmark.get_task_emb(args.task_id)
    
    # 开始评估
    print("\n" + "=" * 80)
    print(f"评估配置:")
    print(f"  基准测试: {args.benchmark}")
    print(f"  任务 ID: {args.task_id}")
    print(f"  任务描述: {task.language}")
    print(f"  算法: {args.algo}")
    print(f"  策略: {args.policy}")
    print(f"  种子: {args.seed}")
    print(f"  设备: {args.device_id}")
    print("=" * 80)
    
    with Timer() as t:
        success_rate = evaluate_with_display(cfg, algo, task, task_emb, args.task_id, args)
    
    print(f"\n[info] 评估完成，用时 {t.get_elapsed_time():.2f} 秒")
    print(f"[info] 最终成功率: {success_rate:.2%}")


if __name__ == "__main__":
    main()
