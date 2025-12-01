"""
Evaluation script for SmolVLA on LIBERO environments.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from smolvla_libero.model import SmolVLAForLibero
from smolvla_libero.config import SmolVLAConfig

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.video_utils import VideoWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on LIBERO")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
        help="Which LIBERO benchmark to evaluate on"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        help="Task ID to evaluate"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save evaluation videos"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./smolvla_eval_videos",
        help="Directory to save videos"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def raw_obs_to_image(obs, camera_name="agentview_image"):
    """Convert raw observation to image tensor."""
    from torchvision import transforms
    from PIL import Image
    
    # Get image from observation
    img = obs[camera_name]  # (H, W, C) numpy array
    img = Image.fromarray(img)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(img)


def evaluate_task(
    model,
    task,
    task_description,
    init_states,
    num_episodes=20,
    max_steps=600,
    device="cuda:0",
    save_videos=False,
    video_dir=None,
    seed=42
):
    """
    Evaluate model on a single task.
    
    Returns:
        success_rate: float
        avg_steps: float
    """
    # Setup environment
    bddl_folder = get_libero_path("bddl_files")
    env_args = {
        "bddl_file_name": os.path.join(bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    
    # Create vectorized environment
    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_episodes)]
    )
    env.reset()
    env.seed(seed)
    
    # Initialize states
    indices = np.arange(num_episodes) % init_states.shape[0]
    init_states_subset = init_states[indices]
    
    # Reset environment
    obs = env.set_init_state(init_states_subset)
    
    # Tracking
    dones = [False] * num_episodes
    steps_per_episode = [0] * num_episodes
    
    # Video writer
    video_writer = VideoWriter(video_dir, save_videos, fps=30, single_video=False)
    
    # Simulate physics for a few steps
    for _ in range(5):
        env.step(np.zeros((num_episodes, 7)))
    
    # Evaluation loop
    model.eval()
    with torch.no_grad():
        for step in tqdm(range(max_steps), desc="Evaluating"):
            # Prepare observations
            images = []
            for i in range(num_episodes):
                if not dones[i]:
                    img = raw_obs_to_image(obs[i])
                    images.append(img)
                else:
                    # Dummy image for done episodes
                    images.append(torch.zeros(3, 224, 224))
            
            images = torch.stack(images).to(device)
            
            # Prepare text inputs
            if model.processor is not None:
                text_inputs = model.processor(
                    text=[task_description] * num_episodes,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            else:
                text_inputs = {
                    "input_ids": torch.randint(0, 1000, (num_episodes, 10)).to(device)
                }
            
            # Predict actions
            outputs = model(images, text_inputs)
            predicted_actions = outputs["predicted_actions"]
            
            # Denormalize actions
            if model.config.action_normalization:
                predicted_actions = model.denormalize_actions(predicted_actions)
            
            # Take first action from sequence
            actions = predicted_actions[:, 0].cpu().numpy()
            
            # Step environment
            obs, rewards, done, info = env.step(actions)
            
            # Update tracking
            for i in range(num_episodes):
                if not dones[i]:
                    steps_per_episode[i] += 1
                    if done[i]:
                        dones[i] = True
            
            # Save video frames
            video_writer.append_vector_obs(obs, dones, camera_name="agentview_image")
            
            # Check if all done
            if all(dones):
                break
    
    # Save videos
    video_writer.save()
    
    # Close environment
    env.close()
    
    # Compute metrics
    num_success = sum(dones)
    success_rate = num_success / num_episodes
    avg_steps = np.mean([s for i, s in enumerate(steps_per_episode) if dones[i]]) if num_success > 0 else max_steps
    
    return success_rate, avg_steps


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    config.device = args.device
    
    # Create model
    model = SmolVLAForLibero(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.action_mean = checkpoint["action_mean"]
    model.action_std = checkpoint["action_std"]
    model = model.to(args.device)
    model.eval()
    
    print(f"[INFO] Model loaded successfully")
    
    # Load benchmark
    benchmark_map = {
        "libero_10": "LIBERO_10",
        "libero_spatial": "LIBERO_SPATIAL",
        "libero_object": "LIBERO_OBJECT",
        "libero_goal": "LIBERO_GOAL",
    }
    benchmark_name = benchmark_map[args.benchmark]
    benchmark = get_benchmark(benchmark_name)(task_order_index=0)
    
    # Get task
    task = benchmark.get_task(args.task_id)
    task_description = task.language
    
    print(f"[INFO] Evaluating on task {args.task_id}: {task.name}")
    print(f"[INFO] Description: {task_description}")
    
    # Load initial states
    init_states_folder = get_libero_path("init_states")
    init_states_path = os.path.join(
        init_states_folder,
        task.problem_folder,
        task.init_states_file
    )
    init_states = torch.load(init_states_path)
    
    # Setup video directory
    if args.save_videos:
        video_dir = os.path.join(
            args.video_dir,
            f"{args.benchmark}_task{args.task_id}"
        )
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = None
    
    # Evaluate
    print(f"[INFO] Running {args.num_episodes} episodes...")
    success_rate, avg_steps = evaluate_task(
        model=model,
        task=task,
        task_description=task_description,
        init_states=init_states,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=args.device,
        save_videos=args.save_videos,
        video_dir=video_dir,
        seed=args.seed
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Task: {task.name}")
    print(f"Success Rate: {success_rate:.2%} ({int(success_rate * args.num_episodes)}/{args.num_episodes})")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        "task_id": args.task_id,
        "task_name": task.name,
        "task_description": task_description,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "num_episodes": args.num_episodes
    }
    
    results_path = Path(args.checkpoint).parent / f"eval_task{args.task_id}_results.json"
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
