"""Training loop orchestrator."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from .trainer import train_epoch, validate
from .early_stopping import EarlyStopping


class TrainingOrchestrator:
    """
    Orchestrates the training process.
    
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
        save_dir: str = "models/checkpoints"
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
        
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
    
    def train(self, epochs: int, model_name: str = "model") -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            model_name: Name for saving the model
        
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
                self.model, self.train_loader, self.criterion, 
                self.optimizer, self.device
            )
            
            # Validate
            val_loss, val_acc = validate(
                self.model, self.val_loader, self.criterion, self.device
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
    
    def save_checkpoint(self, epoch: int, model_name: str = "model"):
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch
            model_name: Name for the checkpoint
        """
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
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        print(f"   Best val acc: {self.best_val_acc*100:.2f}%")