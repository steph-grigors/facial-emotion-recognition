"""
Advanced augmentation utilities for training.

Implements MixUp and CutMix augmentation techniques used in the
Enhanced CNN (ResNet50) training Phase 2.
"""

import torch
import numpy as np
from typing import Tuple


def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp augmentation to a batch.
    
    MixUp creates virtual training examples by mixing pairs of examples
    and their labels. Reference: https://arxiv.org/abs/1710.09412
    
    Args:
        x: Input images (batch_size, channels, height, width)
        y: Labels (batch_size,)
        alpha: Beta distribution parameter (default: 0.4)
    
    Returns:
        Tuple of:
        - mixed_x: Mixed images
        - y_a: First set of labels
        - y_b: Second set of labels
        - lam: Mixing coefficient (lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation to a batch.
    
    CutMix cuts and pastes patches between training images, with the
    ground truth labels proportionally mixed based on the area of patches.
    Reference: https://arxiv.org/abs/1905.04899
    
    Args:
        x: Input images (batch_size, channels, height, width)
        y: Labels (batch_size,)
        alpha: Beta distribution parameter (default: 0.4)
    
    Returns:
        Tuple of:
        - mixed_x: Mixed images
        - y_a: First set of labels
        - y_b: Second set of labels
        - lam: Mixing coefficient (adjusted by box area)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get image dimensions
    _, _, H, W = x.size()
    
    # Calculate box coordinates
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    # Uniform random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match the actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Calculate mixed loss for MixUp/CutMix.
    
    Loss is a weighted combination of losses for both labels.
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def test_augmentations():
    """Test MixUp and CutMix augmentations."""
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("AUGMENTATION UTILITIES TEST")
    print("="*70)
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 128)
    y = torch.tensor([0, 1, 2, 3])
    
    print(f"\nOriginal batch shape: {x.shape}")
    print(f"Original labels: {y.tolist()}")
    
    # Test MixUp
    print("\n" + "-"*70)
    print("Testing MixUp:")
    print("-"*70)
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    print(f"Mixed images shape: {mixed_x.shape}")
    print(f"Labels A: {y_a.tolist()}")
    print(f"Labels B: {y_b.tolist()}")
    print(f"Lambda: {lam:.3f}")
    print("✓ MixUp test passed")
    
    # Test CutMix
    print("\n" + "-"*70)
    print("Testing CutMix:")
    print("-"*70)
    cutmix_x, y_a, y_b, lam = cutmix_data(x, y, alpha=0.4)
    print(f"CutMix images shape: {cutmix_x.shape}")
    print(f"Labels A: {y_a.tolist()}")
    print(f"Labels B: {y_b.tolist()}")
    print(f"Lambda (adjusted): {lam:.3f}")
    print("✓ CutMix test passed")
    
    # Test loss calculation
    print("\n" + "-"*70)
    print("Testing mixed loss calculation:")
    print("-"*70)
    criterion = torch.nn.CrossEntropyLoss()
    pred = torch.randn(batch_size, 7)  # 7 classes
    
    normal_loss = criterion(pred, y)
    mixed_loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
    
    print(f"Normal loss: {normal_loss.item():.4f}")
    print(f"Mixed loss: {mixed_loss.item():.4f}")
    print("✓ Loss calculation test passed")
    
    print("\n" + "="*70)
    print("✅ All augmentation tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_augmentations()
