"""
Dataset class for loading LIBERO data for SmolVLA training.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class LiberoSmolVLADataset(Dataset):
    """
    Dataset for loading LIBERO demonstrations for SmolVLA training.
    
    Each sample contains:
    - images: sequence of RGB images from robot camera
    - text: natural language task description
    - actions: sequence of robot actions (7-dim: position + rotation + gripper)
    """
    
    def __init__(
        self,
        hdf5_path,
        task_description,
        img_size=224,
        seq_len=10,
        train=True,
        camera_names=None
    ):
        """
        Args:
            hdf5_path: Path to LIBERO HDF5 dataset file
            task_description: Natural language description of the task
            img_size: Size to resize images to
            seq_len: Length of action sequence to predict
            train: Whether this is training data (for augmentation)
            camera_names: List of camera names to use (default: agentview + eye_in_hand)
        """
        self.hdf5_path = hdf5_path
        self.task_description = task_description
        self.img_size = img_size
        self.seq_len = seq_len
        self.train = train
        
        if camera_names is None:
            self.camera_names = ["agentview_image", "robot0_eye_in_hand_image"]
        else:
            self.camera_names = camera_names
        
        # Load dataset metadata
        with h5py.File(hdf5_path, "r") as f:
            self.demo_keys = list(f["data"].keys())
            self.num_demos = len(self.demo_keys)
            
            # Get total number of transitions
            self.demo_lengths = []
            for demo_key in self.demo_keys:
                demo_length = f[f"data/{demo_key}/actions"].shape[0]
                self.demo_lengths.append(demo_length)
            
            self.total_transitions = sum(self.demo_lengths)
        
        # Create index mapping (demo_idx, timestep)
        self.indices = []
        for demo_idx, demo_length in enumerate(self.demo_lengths):
            for t in range(demo_length):
                self.indices.append((demo_idx, t))
        
        # Image transforms
        if train:
            self.image_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)
                ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        print(f"[Dataset] Loaded {self.num_demos} demos, {self.total_transitions} transitions")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary with:
        - image: (C, H, W) tensor
        - text: string
        - actions: (seq_len, action_dim) tensor
        - action_mask: (seq_len,) boolean tensor indicating valid actions
        """
        demo_idx, timestep = self.indices[idx]
        demo_key = self.demo_keys[demo_idx]
        
        with h5py.File(self.hdf5_path, "r") as f:
            demo = f[f"data/{demo_key}"]
            
            # Load image from first camera
            # Shape: (H, W, C) in uint8
            img = demo["obs"][self.camera_names[0]][timestep]
            img = Image.fromarray(img)
            img = self.image_transform(img)
            
            # Load actions
            # Get next seq_len actions
            demo_length = demo["actions"].shape[0]
            end_idx = min(timestep + self.seq_len, demo_length)
            actions = demo["actions"][timestep:end_idx]
            
            # Pad if necessary
            if len(actions) < self.seq_len:
                pad_length = self.seq_len - len(actions)
                actions = np.concatenate([
                    actions,
                    np.zeros((pad_length, actions.shape[1]))
                ], axis=0)
                action_mask = np.array(
                    [True] * (self.seq_len - pad_length) + [False] * pad_length
                )
            else:
                action_mask = np.ones(self.seq_len, dtype=bool)
            
            actions = torch.from_numpy(actions).float()
            action_mask = torch.from_numpy(action_mask)
        
        return {
            "image": img,
            "text": self.task_description,
            "actions": actions,
            "action_mask": action_mask,
            "demo_idx": demo_idx,
            "timestep": timestep
        }
    
    def get_action_stats(self):
        """Compute action statistics for normalization."""
        all_actions = []
        
        with h5py.File(self.hdf5_path, "r") as f:
            for demo_key in self.demo_keys:
                actions = f[f"data/{demo_key}/actions"][:]
                all_actions.append(actions)
        
        all_actions = np.concatenate(all_actions, axis=0)
        
        return {
            "mean": all_actions.mean(axis=0),
            "std": all_actions.std(axis=0),
            "min": all_actions.min(axis=0),
            "max": all_actions.max(axis=0)
        }


class LiberoMultiTaskDataset(Dataset):
    """
    Dataset that combines multiple LIBERO tasks for multi-task learning.
    """
    
    def __init__(self, task_datasets):
        """
        Args:
            task_datasets: List of LiberoSmolVLADataset instances
        """
        self.task_datasets = task_datasets
        self.num_tasks = len(task_datasets)
        
        # Compute cumulative lengths
        self.cumulative_lengths = [0]
        for dataset in task_datasets:
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(dataset)
            )
        
        self.total_length = self.cumulative_lengths[-1]
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """Get item from appropriate task dataset."""
        # Find which task this index belongs to
        task_idx = 0
        for i in range(self.num_tasks):
            if idx < self.cumulative_lengths[i + 1]:
                task_idx = i
                break
        
        # Get local index within that task
        local_idx = idx - self.cumulative_lengths[task_idx]
        
        # Get item from task dataset
        item = self.task_datasets[task_idx][local_idx]
        item["task_idx"] = task_idx
        
        return item
