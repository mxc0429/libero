"""
Trainer for SmolVLA on LIBERO tasks.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from pathlib import Path
import json


class SmolVLATrainer:
    """Trainer class for SmolVLA model."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        output_dir,
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        log_every=10,
        eval_every=500,
        save_every=1000,
        use_wandb=False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.use_wandb = use_wandb
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * 100  # Estimate
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=lr * 0.1
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def train(self, num_epochs):
        """Main training loop."""
        print(f"[Trainer] Starting training for {num_epochs} epochs")
        print(f"[Trainer] Total training batches: {len(self.train_loader)}")
        print(f"[Trainer] Total validation batches: {len(self.val_loader)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth")
                print(f"[Epoch {epoch + 1}] New best model saved!")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
        
        print(f"\n[Trainer] Training completed!")
        print(f"[Trainer] Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint("final_model.pth")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            # Log
            if self.global_step % self.log_every == 0:
                if self.use_wandb:
                    wandb.log({
                        "train_step_loss": loss,
                        "global_step": self.global_step
                    })
            
            # Periodic validation
            if self.global_step % self.eval_every == 0 and self.global_step > 0:
                val_loss = self.validate()
                print(f"\n[Step {self.global_step}] Validation Loss: {val_loss:.4f}")
                self.model.train()
            
            # Periodic save
            if self.global_step % self.save_every == 0 and self.global_step > 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_step(self, batch):
        """Single training step."""
        # Move batch to device
        images = batch["image"].to(self.config.device)
        actions = batch["actions"].to(self.config.device)
        action_mask = batch["action_mask"].to(self.config.device)
        texts = batch["text"]
        
        # Prepare text inputs
        if self.model.processor is not None:
            text_inputs = self.model.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            text_inputs = {k: v.to(self.config.device) for k, v in text_inputs.items()}
        else:
            # Placeholder tokenization
            batch_size = len(texts)
            text_inputs = {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)).to(self.config.device)
            }
        
        # Forward pass
        outputs = self.model(images, text_inputs, actions)
        loss = outputs["loss"]
        
        # Apply action mask to loss
        if action_mask is not None:
            # Compute per-timestep loss
            predicted = outputs["predicted_actions"]
            per_step_loss = nn.functional.mse_loss(
                predicted, actions, reduction='none'
            )  # (batch, seq_len, action_dim)
            
            # Mask and average
            per_step_loss = per_step_loss.mean(dim=-1)  # (batch, seq_len)
            masked_loss = (per_step_loss * action_mask.float()).sum() / action_mask.float().sum()
            loss = masked_loss
        
        # Backward pass with gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        # Update weights
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item() * self.gradient_accumulation_steps
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move batch to device
            images = batch["image"].to(self.config.device)
            actions = batch["actions"].to(self.config.device)
            action_mask = batch["action_mask"].to(self.config.device)
            texts = batch["text"]
            
            # Prepare text inputs
            if self.model.processor is not None:
                text_inputs = self.model.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                text_inputs = {k: v.to(self.config.device) for k, v in text_inputs.items()}
            else:
                batch_size = len(texts)
                text_inputs = {
                    "input_ids": torch.randint(0, 1000, (batch_size, 10)).to(self.config.device)
                }
            
            # Forward pass
            outputs = self.model(images, text_inputs, actions)
            loss = outputs["loss"]
            
            # Apply mask
            if action_mask is not None:
                predicted = outputs["predicted_actions"]
                per_step_loss = nn.functional.mse_loss(
                    predicted, actions, reduction='none'
                ).mean(dim=-1)
                masked_loss = (per_step_loss * action_mask.float()).sum() / action_mask.float().sum()
                loss = masked_loss
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filename):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / filename
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "action_mean": self.model.action_mean,
            "action_std": self.model.action_std
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"[Trainer] Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"[Trainer] Checkpoint loaded from {checkpoint_path}")
        print(f"[Trainer] Resuming from epoch {self.epoch}, step {self.global_step}")
