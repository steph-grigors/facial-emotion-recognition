"""
Training orchestrator with support for two-phase training.

Implements:
- Standard single-phase training (for Baseline CNN)
- Two-phase training (for Enhanced CNN/ResNet50):
  * Phase 1: Frozen backbone, train classifier only
  * Phase 2: Full fine-tuning with MixUp/CutMix
"""

import torch
from torch.cuda.amp import GradScaler
from pathlib import Path
from typing import Optional, Dict, Any
from .trainer import train_epoch, validate
from .early_stopping import EarlyStopping


class TrainingOrchestrator:
    """
    Orchestrates the training process with optional two-phase strategy.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        scheduler: Learning rate scheduler (optional)
        early_stopping_patience: Patience for early stopping
        save_dir: Directory to save checkpoints
        use_amp: Use Automatic Mixed Precision (default: True)
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scheduler=None,
        early_stopping_patience: int = 10,
        save_dir: str = "models/checkpoints",
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Initialize AMP scaler
        if use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
    
    def train(
        self,
        epochs: int,
        model_name: str = "model",
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        aug_prob: float = 0.0
    ) -> Dict[str, Any]:
        """
        Standard single-phase training.
        
        Args:
            epochs: Number of epochs to train
            model_name: Name for saving the model
            mixup_alpha: MixUp alpha (0.0 = disabled)
            cutmix_alpha: CutMix alpha (0.0 = disabled)
            aug_prob: Probability of applying augmentation
        
        Returns:
            Dictionary with training history and best accuracy
        """
        print("=" * 70)
        print(f"🔥 Training {model_name}")
        print("=" * 70)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = train_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                scaler=self.scaler,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                aug_prob=aug_prob,
                use_amp=self.use_amp
            )
            
            # Validate
            val_loss, val_acc = validate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device
            )
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_path = self.save_dir / f'{model_name}_best.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"💾 Saved new best model (Val Acc: {val_acc*100:.2f}%)")
            
            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"\n✅ Training complete!")
        print(f"   Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"   Model saved: {self.save_dir / f'{model_name}_best.pth'}")
        
        return {
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
    
    def train_two_phase(
        self,
        phase1_epochs: int,
        phase2_epochs: int,
        model_name: str = "model",
        phase1_lr: float = 1e-3,
        phase2_lr: float = 3e-4,
        phase2_mixup_alpha: float = 0.4,
        phase2_cutmix_alpha: float = 0.4,
        phase2_aug_prob: float = 0.5
    ) -> Dict[str, Any]:
        """
        Two-phase training for transfer learning models.
        
        Phase 1: Frozen backbone, train only classifier
        Phase 2: Unfreeze all, train with MixUp/CutMix
        
        Args:
            phase1_epochs: Number of epochs for Phase 1
            phase2_epochs: Number of epochs for Phase 2
            model_name: Name for saving the model
            phase1_lr: Learning rate for Phase 1
            phase2_lr: Learning rate for Phase 2
            phase2_mixup_alpha: MixUp alpha for Phase 2
            phase2_cutmix_alpha: CutMix alpha for Phase 2
            phase2_aug_prob: Augmentation probability for Phase 2
        
        Returns:
            Dictionary with training history and best accuracy
        """
        print("=" * 70)
        print(f"🔥 Two-Phase Training: {model_name}")
        print("=" * 70)
        
        # ================================================================
        # PHASE 1: Frozen Backbone
        # ================================================================
        print("\n" + "=" * 60)
        print("🔵 PHASE 1 — Training with Frozen Backbone")
        print("=" * 60)
        
        # Freeze backbone (assumes model has freeze_backbone method)
        if hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
        else:
            # Generic freeze (assume 'fc' or 'classifier' is the head)
            for name, param in self.model.named_parameters():
                if 'fc' in name or 'classifier' in name or 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("🔒 Backbone frozen (generic method)")
        
        # New optimizer for Phase 1 (only trainable params)
        phase1_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=phase1_lr
        )
        
        # New scheduler for Phase 1
        phase1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            phase1_optimizer,
            mode='max',
            patience=2,
            factor=0.5
        )
        
        # Train Phase 1 (no augmentation)
        for epoch in range(1, phase1_epochs + 1):
            print(f"\nEpoch {epoch}/{phase1_epochs} (Phase 1)")
            print("-" * 50)
            
            train_loss, train_acc = train_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                phase1_optimizer,
                self.device,
                scaler=self.scaler,
                mixup_alpha=0.0,
                cutmix_alpha=0.0,
                aug_prob=0.0,
                use_amp=self.use_amp
            )
            
            val_loss, val_acc = validate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device
            )
            
            phase1_scheduler.step(val_acc)
            current_lr = phase1_optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_path = self.save_dir / f'{model_name}_best.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"💾 Saved new best model (Val Acc: {val_acc*100:.2f}%)")
        
        # ================================================================
        # PHASE 2: Full Fine-Tuning
        # ================================================================
        print("\n" + "=" * 60)
        print("🔥 PHASE 2 — Unfreezing Backbone + Advanced Augmentations")
        print("=" * 60)
        
        # Unfreeze all parameters
        if hasattr(self.model, 'unfreeze_backbone'):
            self.model.unfreeze_backbone()
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            print("🔓 Backbone unfrozen (generic method)")
        
        # New optimizer for Phase 2 (all params)
        phase2_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=phase2_lr
        )
        
        # New scheduler for Phase 2
        phase2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            phase2_optimizer,
            mode='max',
            patience=2,
            factor=0.5
        )
        
        # Train Phase 2 (with augmentation)
        for epoch in range(1, phase2_epochs + 1):
            print(f"\nEpoch {epoch}/{phase2_epochs} (Phase 2)")
            print("-" * 50)
            
            train_loss, train_acc = train_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                phase2_optimizer,
                self.device,
                scaler=self.scaler,
                mixup_alpha=phase2_mixup_alpha,
                cutmix_alpha=phase2_cutmix_alpha,
                aug_prob=phase2_aug_prob,
                use_amp=self.use_amp
            )
            
            val_loss, val_acc = validate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device
            )
            
            phase2_scheduler.step(val_acc)
            current_lr = phase2_optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_path = self.save_dir / f'{model_name}_best.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"💾 Saved new best model (Val Acc: {val_acc*100:.2f}%)")
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered at Phase 2 epoch {epoch}")
                break
        
        print(f"\n✅ Training complete!")
        print(f"🏆 Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"   Model saved: {self.save_dir / f'{model_name}_best.pth'}")
        
        return {
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
    
    def save_checkpoint(self, epoch: int, model_name: str = "model"):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.save_dir / f'{model_name}_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        print(f"   Best val acc: {self.best_val_acc*100:.2f}%")


if __name__ == "__main__":
    print("Training orchestrator module loaded successfully!")
    print("\nFeatures:")
    print("  ✓ Standard single-phase training")
    print("  ✓ Two-phase training (frozen → full fine-tuning)")
    print("  ✓ MixUp/CutMix augmentation support")
    print("  ✓ Automatic Mixed Precision (AMP)")
    print("  ✓ Early stopping")
    print("  ✓ Learning rate scheduling")
    print("  ✓ Checkpoint saving/loading")
