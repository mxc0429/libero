"""
SmolVLA model adapted for LIBERO manipulation tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Dict, Optional


class ActionHead(nn.Module):
    """
    Action prediction head that takes vision-language features
    and predicts robot actions.
    """
    
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Final action prediction layer
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            actions: (batch_size, seq_len, action_dim)
        """
        return self.network(x)


class SmolVLAForLibero(nn.Module):
    """
    SmolVLA model adapted for LIBERO robot manipulation.
    
    Architecture:
    1. Vision encoder: processes RGB images
    2. Language encoder: processes task descriptions
    3. Vision-Language fusion: combines visual and language features
    4. Action head: predicts robot actions
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load SmolVLM base model
        print(f"[Model] Loading SmolVLM: {config.model_name}")
        try:
            self.processor = AutoProcessor.from_pretrained(config.model_name)
            self.base_model = AutoModelForVision2Seq.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.mixed_precision else torch.float32,
            )
        except Exception as e:
            print(f"[Warning] Could not load {config.model_name}: {e}")
            print("[Info] Using placeholder model for demonstration")
            # Create a simple placeholder for demonstration
            self.processor = None
            self.base_model = self._create_placeholder_model()
        
        # Freeze parts of the model if specified
        if config.freeze_vision_encoder:
            self._freeze_vision_encoder()
        
        if config.freeze_language_model:
            self._freeze_language_model()
        
        # Get hidden dimension from base model
        # This depends on the specific model architecture
        try:
            self.hidden_dim = self.base_model.config.hidden_size
        except:
            self.hidden_dim = 768  # Default for many models
        
        # Action prediction head
        self.action_head = ActionHead(
            input_dim=self.hidden_dim,
            action_dim=config.action_dim * config.seq_len,  # Predict all actions at once
            hidden_dim=config.action_head_hidden_dim,
            num_layers=config.action_head_num_layers,
            dropout=config.action_head_dropout
        )
        
        # Action statistics for normalization
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))
        
        print(f"[Model] SmolVLA initialized with {self.count_parameters():,} parameters")
    
    def _create_placeholder_model(self):
        """Create a simple placeholder model for demonstration."""
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.text_encoder = nn.Embedding(1000, 768)
                self.fusion = nn.Linear(64 + 768, 768)
                
                class Config:
                    hidden_size = 768
                self.config = Config()
            
            def forward(self, pixel_values, input_ids):
                # Simple placeholder forward
                batch_size = pixel_values.shape[0]
                vision_feat = self.vision_encoder(pixel_values).squeeze(-1).squeeze(-1)
                text_feat = self.text_encoder(input_ids[:, 0])
                combined = torch.cat([vision_feat, text_feat], dim=-1)
                output = self.fusion(combined)
                return output.unsqueeze(1)
        
        return PlaceholderModel()
    
    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        if hasattr(self.base_model, 'vision_model'):
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = False
            print("[Model] Vision encoder frozen")
    
    def _freeze_language_model(self):
        """Freeze language model parameters."""
        if hasattr(self.base_model, 'language_model'):
            for param in self.base_model.language_model.parameters():
                param.requires_grad = False
            print("[Model] Language model frozen")
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_action_stats(self, mean, std):
        """Set action normalization statistics."""
        self.action_mean.copy_(torch.from_numpy(mean))
        self.action_std.copy_(torch.from_numpy(std))
        print("[Model] Action normalization stats updated")
    
    def normalize_actions(self, actions):
        """Normalize actions using stored statistics."""
        return (actions - self.action_mean) / (self.action_std + 1e-8)
    
    def denormalize_actions(self, actions):
        """Denormalize actions."""
        return actions * self.action_std + self.action_mean
    
    def forward(
        self,
        images: torch.Tensor,
        text_inputs: Dict[str, torch.Tensor],
        actions: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            images: (batch_size, C, H, W) image tensor
            text_inputs: Dictionary with tokenized text
            actions: (batch_size, seq_len, action_dim) ground truth actions (optional)
        
        Returns:
            Dictionary with:
            - predicted_actions: (batch_size, seq_len, action_dim)
            - loss: scalar (if actions provided)
        """
        batch_size = images.shape[0]
        
        # Process through base model
        if self.processor is not None:
            # Use actual SmolVLM model
            outputs = self.base_model(
                pixel_values=images,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                output_hidden_states=True
            )
            # Get last hidden state
            hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden_dim)
        else:
            # Use placeholder
            hidden_states = self.base_model(images, text_inputs["input_ids"])
        
        # Pool hidden states (take mean over sequence)
        pooled_features = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        
        # Predict actions
        action_logits = self.action_head(pooled_features)  # (batch, seq_len * action_dim)
        
        # Reshape to (batch, seq_len, action_dim)
        predicted_actions = action_logits.view(
            batch_size, self.config.seq_len, self.config.action_dim
        )
        
        # Compute loss if ground truth provided
        loss = None
        if actions is not None:
            # Normalize actions if configured
            if self.config.action_normalization:
                actions = self.normalize_actions(actions)
                predicted_actions_norm = predicted_actions
            else:
                predicted_actions_norm = predicted_actions
            
            # MSE loss
            loss = nn.functional.mse_loss(predicted_actions_norm, actions)
        
        return {
            "predicted_actions": predicted_actions,
            "loss": loss
        }
    
    def predict_action(self, image, text):
        """
        Predict action for a single observation.
        
        Args:
            image: (C, H, W) tensor or PIL Image
            text: string
        
        Returns:
            action: (action_dim,) numpy array
        """
        self.eval()
        with torch.no_grad():
            # Prepare inputs
            if self.processor is not None:
                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                image_tensor = inputs["pixel_values"]
                text_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs.get("attention_mask")
                }
            else:
                # Simple preprocessing for placeholder
                if not isinstance(image, torch.Tensor):
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ])
                    image = transform(image)
                image_tensor = image.unsqueeze(0).to(self.config.device)
                # Simple tokenization
                text_inputs = {
                    "input_ids": torch.randint(0, 1000, (1, 10)).to(self.config.device)
                }
            
            # Forward pass
            outputs = self.forward(image_tensor, text_inputs)
            predicted_actions = outputs["predicted_actions"]
            
            # Denormalize if needed
            if self.config.action_normalization:
                predicted_actions = self.denormalize_actions(predicted_actions)
            
            # Return first action
            action = predicted_actions[0, 0].cpu().numpy()
        
        return action
    
    def save_pretrained(self, save_path):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "action_mean": self.action_mean,
            "action_std": self.action_std
        }, save_path)
        print(f"[Model] Saved to {save_path}")
    
    def load_pretrained(self, load_path):
        """Load model checkpoint."""
        checkpoint = torch.load(load_path, map_location=self.config.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.action_mean = checkpoint["action_mean"]
        self.action_std = checkpoint["action_std"]
        print(f"[Model] Loaded from {load_path}")
