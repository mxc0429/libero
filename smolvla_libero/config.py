"""
Configuration for SmolVLA training on LIBERO.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SmolVLAConfig:
    """Configuration for SmolVLA model."""
    
    # Model configuration
    model_name: str = "HuggingFaceTB/SmolVLM-Instruct"
    action_dim: int = 7  # LIBERO action dimension
    img_size: int = 224
    seq_len: int = 10
    
    # Action head configuration
    action_head_hidden_dim: int = 512
    action_head_num_layers: int = 2
    action_head_dropout: float = 0.1
    
    # Training configuration
    device: str = "cuda:0"
    mixed_precision: bool = True
    
    # Action prediction configuration
    predict_delta: bool = True  # Predict delta actions or absolute
    action_normalization: bool = True
    
    # Vision encoder configuration
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    
    # LoRA configuration (optional)
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.action_dim > 0, "action_dim must be positive"
        assert self.img_size > 0, "img_size must be positive"
        assert self.seq_len > 0, "seq_len must be positive"
