"""
Training and validation functions with advanced augmentation support.

Supports:
- Standard training
- MixUp and CutMix augmentation
- Automatic Mixed Precision (AMP)
- Gradient clipping
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Tuple, Optional
from ..data_loading.augmentations import mixup_data, cutmix_data, mixup_criterion


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler: Optional[GradScaler] = None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    aug_prob: float = 0.0,
    gradient_clip: float = 1.0,
    use_amp: bool = True
) -> Tuple[float, float]:
    """
    Train model for one epoch with optional advanced augmentations.
    
    Args:
        model: PyTorch model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        scaler: GradScaler for AMP (optional, will create if use_amp=True)
        mixup_alpha: MixUp alpha parameter (0.0 = disabled)
        cutmix_alpha: CutMix alpha parameter (0.0 = disabled)
        aug_prob: Probability of applying MixUp/CutMix (0.0 = disabled)
        gradient_clip: Max norm for gradient clipping (default: 1.0)
        use_amp: Use Automatic Mixed Precision (default: True)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create scaler if not provided and AMP is enabled
    if use_amp and scaler is None:
        scaler = GradScaler()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Decide whether to use augmentation
        use_aug = (mixup_alpha > 0 or cutmix_alpha > 0) and (torch.rand(1).item() < aug_prob)
        
        if use_aug:
            # Randomly choose between MixUp and CutMix
            if torch.rand(1).item() < 0.5 and mixup_alpha > 0:
                # Apply MixUp
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                aug_type = "mixup"
            elif cutmix_alpha > 0:
                # Apply CutMix
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
                aug_type = "cutmix"
            else:
                aug_type = "none"
                use_aug = False
        else:
            aug_type = "none"
        
        # Forward pass with AMP
        if use_amp:
            with autocast():
                outputs = model(images)
                if use_aug:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            if use_aug:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item() * images.size(0)
        
        # For accuracy, use original labels (naive accuracy during mixed training)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": loss.item(),
            "acc": 100.0 * correct / total,
            "aug": aug_type
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_epoch_simple(
    model,
    loader,
    criterion,
    optimizer,
    device,
    gradient_clip: float = 1.0
) -> Tuple[float, float]:
    """
    Simple training epoch without augmentation or AMP.
    
    Args:
        model: PyTorch model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        gradient_clip: Max norm for gradient clipping (default: 1.0)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total


def validate(
    model,
    loader,
    criterion,
    device
) -> Tuple[float, float]:
    """
    Validate model.
    
    Args:
        model: PyTorch model
        loader: Validation data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / total, correct / total


def evaluate_model(
    model,
    test_loader,
    device,
    return_predictions: bool = True
):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use (cuda/cpu)
        return_predictions: Whether to return predictions and labels
    
    Returns:
        Dictionary with accuracy and optionally predictions/labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    
    result = {'accuracy': accuracy}
    
    if return_predictions:
        result['predictions'] = all_preds
        result['labels'] = all_labels
    
    return result


def test_trainer_functions():
    """Test trainer functions."""
    print("="*70)
    print("TRAINER MODULE TEST")
    print("="*70)
    
    # Create dummy model and data
    from torch import nn
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 7)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy dataloader
    from torch.utils.data import TensorDataset, DataLoader
    dummy_images = torch.randn(32, 3, 128, 128)
    dummy_labels = torch.randint(0, 7, (32,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=8)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n1. Testing simple training epoch:")
    print("-"*70)
    loss, acc = train_epoch_simple(model, loader, criterion, optimizer, device)
    print(f"Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
    print("✓ Simple training passed")
    
    print("\n2. Testing training with MixUp:")
    print("-"*70)
    scaler = GradScaler()
    loss, acc = train_epoch(
        model, loader, criterion, optimizer, device,
        scaler=scaler, mixup_alpha=0.4, aug_prob=0.5, use_amp=True
    )
    print(f"Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
    print("✓ MixUp training passed")
    
    print("\n3. Testing training with CutMix:")
    print("-"*70)
    loss, acc = train_epoch(
        model, loader, criterion, optimizer, device,
        scaler=scaler, cutmix_alpha=0.4, aug_prob=0.5, use_amp=True
    )
    print(f"Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
    print("✓ CutMix training passed")
    
    print("\n4. Testing validation:")
    print("-"*70)
    val_loss, val_acc = validate(model, loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")
    print("✓ Validation passed")
    
    print("\n5. Testing evaluation:")
    print("-"*70)
    results = evaluate_model(model, loader, device)
    print(f"Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Predictions shape: {len(results['predictions'])}")
    print("✓ Evaluation passed")
    
    print("\n" + "="*70)
    print("✅ ALL TRAINER TESTS PASSED!")
    print("="*70)


if __name__ == "__main__":
    test_trainer_functions()
