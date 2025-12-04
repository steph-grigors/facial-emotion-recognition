"""
Data transformation and augmentation pipelines for FER2013.

Provides separate transforms for:
- Training (with augmentation)
- Validation/Testing (no augmentation)
- Inference (real-time prediction)
"""

from torchvision import transforms
from typing import Optional


def get_train_transforms(
    image_size: int = 48,
    normalize: bool = True,
    mean: Optional[tuple] = None,
    std: Optional[tuple] = None
) -> transforms.Compose:
    """
    Get training data augmentation pipeline.
    
    Includes:
    - Random horizontal flip
    - Random rotation (±10 degrees)
    - Color jitter (brightness, contrast)
    - Grayscale conversion
    - Resize and normalization
    
    Args:
        image_size: Target image size (default: 48 for FER2013)
        normalize: Whether to normalize images (default: True)
        mean: Mean for normalization (default: [0.5] for grayscale)
        std: Std for normalization (default: [0.5] for grayscale)
        
    Returns:
        Composed transform pipeline
    """
    if mean is None:
        mean = [0.5]
    if std is None:
        std = [0.5]
    
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        
        # Data augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = 48,
    normalize: bool = True,
    mean: Optional[tuple] = None,
    std: Optional[tuple] = None
) -> transforms.Compose:
    """
    Get validation/test data transformation pipeline.
    
    No augmentation - only preprocessing:
    - Grayscale conversion
    - Resize and normalization
    
    Args:
        image_size: Target image size (default: 48 for FER2013)
        normalize: Whether to normalize images (default: True)
        mean: Mean for normalization (default: [0.5] for grayscale)
        std: Std for normalization (default: [0.5] for grayscale)
        
    Returns:
        Composed transform pipeline
    """
    if mean is None:
        mean = [0.5]
    if std is None:
        std = [0.5]
    
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_test_transforms(
    image_size: int = 48,
    normalize: bool = True,
    mean: Optional[tuple] = None,
    std: Optional[tuple] = None
) -> transforms.Compose:
    """
    Get test data transformation pipeline (same as validation).
    
    Args:
        image_size: Target image size (default: 48 for FER2013)
        normalize: Whether to normalize images (default: True)
        mean: Mean for normalization (default: [0.5] for grayscale)
        std: Std for normalization (default: [0.5] for grayscale)
        
    Returns:
        Composed transform pipeline
    """
    return get_val_transforms(image_size, normalize, mean, std)


def get_inference_transforms(
    image_size: int = 48,
    normalize: bool = True,
    mean: Optional[tuple] = None,
    std: Optional[tuple] = None
) -> transforms.Compose:
    """
    Get inference transformation pipeline for real-time predictions.
    
    Same as validation transforms - no augmentation.
    
    Args:
        image_size: Target image size (default: 48 for FER2013)
        normalize: Whether to normalize images (default: True)
        mean: Mean for normalization (default: [0.5] for grayscale)
        std: Std for normalization (default: [0.5] for grayscale)
        
    Returns:
        Composed transform pipeline
    """
    return get_val_transforms(image_size, normalize, mean, std)


def get_enhanced_train_transforms(
    image_size: int = 48,
    normalize: bool = True,
    mean: Optional[tuple] = None,
    std: Optional[tuple] = None
) -> transforms.Compose:
    """
    Get enhanced training augmentation pipeline with stronger augmentation.
    
    Used for the Enhanced CNN model (80.21% accuracy).
    
    Includes:
    - Random horizontal flip
    - Random rotation (±15 degrees, more than baseline)
    - Random affine (translation, scale)
    - Color jitter (brightness, contrast, saturation)
    - Random grayscale
    
    Args:
        image_size: Target image size (default: 48 for FER2013)
        normalize: Whether to normalize images (default: True)
        mean: Mean for normalization (default: [0.5] for grayscale)
        std: Std for normalization (default: [0.5] for grayscale)
        
    Returns:
        Composed transform pipeline
    """
    if mean is None:
        mean = [0.5]
    if std is None:
        std = [0.5]
    
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        
        # Stronger augmentation for enhanced model
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        ),
        
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_transforms_from_config(config: dict, split: str = 'train') -> transforms.Compose:
    """
    Get transforms based on configuration dictionary.
    
    Args:
        config: Configuration dictionary with keys:
                - image_size: int
                - normalize: bool
                - mean: list
                - std: list
                - augmentation_level: str ('baseline', 'enhanced')
        split: Data split ('train', 'val', 'test', 'inference')
        
    Returns:
        Appropriate transform pipeline
    """
    image_size = config.get('image_size', 48)
    normalize = config.get('normalize', True)
    mean = config.get('mean', [0.5])
    std = config.get('std', [0.5])
    aug_level = config.get('augmentation_level', 'baseline')
    
    if split == 'train':
        if aug_level == 'enhanced':
            return get_enhanced_train_transforms(image_size, normalize, mean, std)
        else:
            return get_train_transforms(image_size, normalize, mean, std)
    elif split == 'val':
        return get_val_transforms(image_size, normalize, mean, std)
    elif split == 'test':
        return get_test_transforms(image_size, normalize, mean, std)
    elif split == 'inference':
        return get_inference_transforms(image_size, normalize, mean, std)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'inference'")


# Default transforms for quick use
DEFAULT_TRAIN_TRANSFORM = get_train_transforms()
DEFAULT_VAL_TRANSFORM = get_val_transforms()
DEFAULT_TEST_TRANSFORM = get_test_transforms()
DEFAULT_INFERENCE_TRANSFORM = get_inference_transforms()


if __name__ == '__main__':
    # Example usage and testing
    print("Transform pipelines created successfully!")
    print("\n1. Training transforms (with augmentation):")
    train_transform = get_train_transforms()
    print(train_transform)
    
    print("\n2. Validation transforms (no augmentation):")
    val_transform = get_val_transforms()
    print(val_transform)
    
    print("\n3. Enhanced training transforms (stronger augmentation):")
    enhanced_transform = get_enhanced_train_transforms()
    print(enhanced_transform)
    
    print("\n4. Config-based transforms:")
    config = {
        'image_size': 48,
        'normalize': True,
        'mean': [0.5],
        'std': [0.5],
        'augmentation_level': 'enhanced'
    }
    config_transform = get_transforms_from_config(config, 'train')
    print(config_transform)