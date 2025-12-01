"""
SmolVLA for LIBERO
==================
A package for training SmolVLA models on LIBERO manipulation tasks.
"""

from .dataset import LiberoSmolVLADataset
from .model import SmolVLAForLibero
from .trainer import SmolVLATrainer
from .config import SmolVLAConfig

__version__ = "0.1.0"
__all__ = [
    "LiberoSmolVLADataset",
    "SmolVLAForLibero",
    "SmolVLATrainer",
    "SmolVLAConfig",
]
