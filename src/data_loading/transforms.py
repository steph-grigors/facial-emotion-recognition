"""
Data transformation and augmentation pipelines for FER2013.

Provides separate transforms for:
- Baseline CNN: 224x224 RGB with standard augmentation
- Enhanced CNN (ResNet50): 128x128 RGB with advanced augmentation
- Validation/Testing: No augmentation
- Inference: Real-time prediction

All transforms use RGB (3 channels) and ImageNet normalization.
"""

from torchvision import transforms
from typing import Optional, List


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# BASELINE CNN TRANSFORMS (224x224 RGB)
# ============================================================================

def get_baseline_train_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Get training transforms for Baseline CNN.
    
    Used with the custom 3-layer CNN (65.81% accuracy).
    
    Augmentation:
    - Resize to 224x224
    - Random horizontal flip
    - Random rotation (±10 degrees)
    - Color jitter (brightness, contrast)
    - ImageNet normalization
    
    Args:
        image_size: Target image size (default: 224)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        
    Returns:
        Composed transform pipeline
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_baseline_val_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Get validation/test transforms for Baseline CNN.
    
    No augmentation - only preprocessing.
    
    Args:
        image_size: Target image size (default: 224)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        
    Returns:
        Composed transform pipeline
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


# ============================================================================
# ENHANCED CNN (ResNet50) TRANSFORMS (128x128 RGB)
# ============================================================================

def get_enhanced_train_transforms(
    image_size: int = 128,
    overshoot_size: int = 140,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Get training transforms for Enhanced CNN (ResNet50).
    
    Used with ResNet50 transfer learning (80.21% accuracy).
    
    Advanced augmentation strategy:
    - Overshoot resize (140x140)
    - Random resized crop to 128x128 (scale 0.8-1.0)
    - Random horizontal flip
    - Random rotation (±15 degrees)
    - Color jitter (brightness, contrast)
    - Additional random color jitter (30% probability)
    - Random grayscale (10% probability)
    - Random erasing (25% probability) - applied after tensor conversion
    - ImageNet normalization
    
    Args:
        image_size: Target image size after crop (default: 128)
        overshoot_size: Initial resize before crop (default: 140)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        
    Returns:
        Composed transform pipeline
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    return transforms.Compose([
        # PIL transforms
        transforms.Resize((overshoot_size, overshoot_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        ], p=0.3),
        transforms.RandomGrayscale(p=0.1),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Tensor normalization
        transforms.Normalize(mean=mean, std=std),
        
        # Tensor-only augmentation
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
    ])


def get_enhanced_val_transforms(
    image_size: int = 128,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Get validation/test transforms for Enhanced CNN (ResNet50).
    
    No augmentation - only preprocessing.
    
    Args:
        image_size: Target image size (default: 128)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        
    Returns:
        Composed transform pipeline
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

def get_train_transforms(
    model_type: str = 'baseline',
    **kwargs
) -> transforms.Compose:
    """
    Get training transforms based on model type.
    
    Args:
        model_type: 'baseline' or 'enhanced' (default: 'baseline')
        **kwargs: Additional arguments for specific transform function
        
    Returns:
        Appropriate transform pipeline
    """
    if model_type.lower() == 'baseline':
        return get_baseline_train_transforms(**kwargs)
    elif model_type.lower() == 'enhanced':
        return get_enhanced_train_transforms(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'baseline' or 'enhanced'")


def get_val_transforms(
    model_type: str = 'baseline',
    **kwargs
) -> transforms.Compose:
    """
    Get validation/test transforms based on model type.
    
    Args:
        model_type: 'baseline' or 'enhanced' (default: 'baseline')
        **kwargs: Additional arguments for specific transform function
        
    Returns:
        Appropriate transform pipeline
    """
    if model_type.lower() == 'baseline':
        return get_baseline_val_transforms(**kwargs)
    elif model_type.lower() == 'enhanced':
        return get_enhanced_val_transforms(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'baseline' or 'enhanced'")


def get_test_transforms(
    model_type: str = 'baseline',
    **kwargs
) -> transforms.Compose:
    """
    Get test transforms (same as validation).
    
    Args:
        model_type: 'baseline' or 'enhanced' (default: 'baseline')
        **kwargs: Additional arguments for specific transform function
        
    Returns:
        Appropriate transform pipeline
    """
    return get_val_transforms(model_type=model_type, **kwargs)


def get_inference_transforms(
    model_type: str = 'baseline',
    **kwargs
) -> transforms.Compose:
    """
    Get inference transforms for real-time predictions.
    
    Same as validation transforms - no augmentation.
    
    Args:
        model_type: 'baseline' or 'enhanced' (default: 'baseline')
        **kwargs: Additional arguments for specific transform function
        
    Returns:
        Appropriate transform pipeline
    """
    return get_val_transforms(model_type=model_type, **kwargs)


# ============================================================================
# CONFIG-BASED FACTORY
# ============================================================================

def get_transforms_from_config(
    config: dict, 
    split: str = 'train',
    model_type: str = 'baseline'
) -> transforms.Compose:
    """
    Get transforms based on configuration dictionary.
    
    Args:
        config: Configuration dictionary with keys:
                - image_size: int
                - mean: list (optional)
                - std: list (optional)
        split: Data split ('train', 'val', 'test', 'inference')
        model_type: 'baseline' or 'enhanced'
        
    Returns:
        Appropriate transform pipeline
    """
    image_size = config.get('image_size', 224 if model_type == 'baseline' else 128)
    mean = config.get('mean', IMAGENET_MEAN)
    std = config.get('std', IMAGENET_STD)
    
    if split == 'train':
        if model_type == 'baseline':
            return get_baseline_train_transforms(
                image_size=image_size, mean=mean, std=std
            )
        else:
            overshoot = config.get('overshoot_size', 140)
            return get_enhanced_train_transforms(
                image_size=image_size, 
                overshoot_size=overshoot,
                mean=mean, 
                std=std
            )
    elif split in ['val', 'test', 'inference']:
        if model_type == 'baseline':
            return get_baseline_val_transforms(
                image_size=image_size, mean=mean, std=std
            )
        else:
            return get_enhanced_val_transforms(
                image_size=image_size, mean=mean, std=std
            )
    else:
        raise ValueError(
            f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'inference'"
        )


# ============================================================================
# PRESETS
# ============================================================================

# Baseline CNN presets (224x224)
BASELINE_TRAIN_TRANSFORM = get_baseline_train_transforms()
BASELINE_VAL_TRANSFORM = get_baseline_val_transforms()

# Enhanced CNN presets (128x128)
ENHANCED_TRAIN_TRANSFORM = get_enhanced_train_transforms()
ENHANCED_VAL_TRANSFORM = get_enhanced_val_transforms()


if __name__ == '__main__':
    print("="*70)
    print("TRANSFORM PIPELINES TEST")
    print("="*70)
    
    print("\n1. BASELINE CNN TRANSFORMS (224x224 RGB):")
    print("-"*70)
    print("Training:")
    print(get_baseline_train_transforms())
    print("\nValidation:")
    print(get_baseline_val_transforms())
    
    print("\n" + "="*70)
    print("2. ENHANCED CNN TRANSFORMS (128x128 RGB):")
    print("-"*70)
    print("Training:")
    print(get_enhanced_train_transforms())
    print("\nValidation:")
    print(get_enhanced_val_transforms())
    
    print("\n" + "="*70)
    print("3. FACTORY FUNCTION TEST:")
    print("-"*70)
    
    # Test factory function
    baseline_train = get_train_transforms(model_type='baseline')
    enhanced_train = get_train_transforms(model_type='enhanced')
    
    print("✓ Baseline train transform created")
    print("✓ Enhanced train transform created")
    
    print("\n" + "="*70)
    print("4. KEY DIFFERENCES:")
    print("-"*70)
    print("Baseline CNN:")
    print("  - Input size: 224x224")
    print("  - Simple resize + augmentation")
    print("  - RGB (3 channels)")
    print("  - ImageNet normalization")
    print("\nEnhanced CNN (ResNet50):")
    print("  - Input size: 128x128 (from 140 overshoot + crop)")
    print("  - Advanced augmentation (RandomResizedCrop, RandomErasing)")
    print("  - RGB (3 channels)")
    print("  - ImageNet normalization")
    
    print("\n✅ All transform pipelines created successfully!")
