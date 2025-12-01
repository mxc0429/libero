"""
SmolVLA Training Script for LIBERO Dataset
===========================================
This script trains SmolVLA model on LIBERO manipulation tasks.

Usage:
    python train_smolvla.py --benchmark libero_10 --seed 42 --device cuda:0
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from smolvla_libero.dataset import LiberoSmolVLADataset
from smolvla_libero.model import SmolVLAForLibero
from smolvla_libero.trainer import SmolVLATrainer
from smolvla_libero.config import SmolVLAConfig

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmolVLA on LIBERO")
    
    # Dataset arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
        help="Which LIBERO benchmark to use"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to LIBERO datasets (default: use LIBERO default path)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolVLM-Instruct",
        help="SmolVLA model name or path"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained SmolVLA checkpoint"
    )
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    
    # Data arguments
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./smolvla_experiments")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="smolvla-libero")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    # Multi-task or single-task
    parser.add_argument(
        "--task_ids",
        type=str,
        default="all",
        help="Task IDs to train on (comma-separated or 'all')"
    )
    
    args = parser.parse_args()
    return args


def setup_experiment(args):
    """Setup experiment directory and logging."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment directory
    exp_name = f"{args.benchmark}_smolvla_seed{args.seed}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"[INFO] Experiment directory: {exp_dir}")
    
    # Setup wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=vars(args)
        )
    
    return exp_dir


def load_libero_benchmark(args):
    """Load LIBERO benchmark and tasks."""
    benchmark_map = {
        "libero_10": "LIBERO_10",
        "libero_spatial": "LIBERO_SPATIAL",
        "libero_object": "LIBERO_OBJECT",
        "libero_goal": "LIBERO_GOAL",
    }
    
    benchmark_name = benchmark_map[args.benchmark]
    benchmark = get_benchmark(benchmark_name)(task_order_index=0)
    
    # Get task IDs
    if args.task_ids == "all":
        task_ids = list(range(benchmark.n_tasks))
    else:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    
    print(f"[INFO] Loaded benchmark: {benchmark_name}")
    print(f"[INFO] Training on {len(task_ids)} tasks: {task_ids}")
    
    return benchmark, task_ids


def create_datasets(args, benchmark, task_ids):
    """Create training and validation datasets."""
    data_dir = args.data_dir or get_libero_path("datasets")
    
    train_datasets = []
    val_datasets = []
    
    for task_id in task_ids:
        task = benchmark.get_task(task_id)
        task_name = task.name
        task_description = task.language
        
        # Get dataset path
        dataset_path = os.path.join(
            data_dir,
            benchmark.get_task_demonstration(task_id)
        )
        
        if not os.path.exists(dataset_path):
            print(f"[WARNING] Dataset not found: {dataset_path}")
            continue
        
        print(f"[INFO] Loading task {task_id}: {task_name}")
        print(f"       Description: {task_description}")
        
        # Create dataset
        dataset = LiberoSmolVLADataset(
            hdf5_path=dataset_path,
            task_description=task_description,
            img_size=args.img_size,
            seq_len=args.seq_len,
            train=True
        )
        
        # Split into train/val (90/10)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
    
    # Concatenate all tasks
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    print(f"[INFO] Total training samples: {len(train_dataset)}")
    print(f"[INFO] Total validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def main():
    args = parse_args()
    
    # Setup experiment
    exp_dir = setup_experiment(args)
    
    # Load benchmark
    benchmark, task_ids = load_libero_benchmark(args)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args, benchmark, task_ids)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model config
    config = SmolVLAConfig(
        model_name=args.model_name,
        action_dim=7,  # LIBERO uses 7-dim actions
        img_size=args.img_size,
        seq_len=args.seq_len,
        device=args.device
    )
    
    # Create model
    print(f"[INFO] Loading SmolVLA model: {args.model_name}")
    model = SmolVLAForLibero(config)
    
    if args.pretrained_path:
        print(f"[INFO] Loading pretrained weights from: {args.pretrained_path}")
        model.load_pretrained(args.pretrained_path)
    
    model = model.to(args.device)
    
    # Create trainer
    trainer = SmolVLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=exp_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        use_wandb=args.use_wandb
    )
    
    # Train
    print("[INFO] Starting training...")
    trainer.train(num_epochs=args.num_epochs)
    
    print(f"[INFO] Training completed! Models saved to: {exp_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
