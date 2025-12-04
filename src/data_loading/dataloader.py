"""
DataLoader factory for FER2013 emotion recognition.

Provides functions to create PyTorch DataLoaders using ImageFolder
for train/val/test splits.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from typing import Tuple, Optional, Dict
import yaml

from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    get_enhanced_train_transforms,
    get_transforms_from_config
)


def create_dataloader(
    data_dir: str,
    transform,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, datasets.ImageFolder]:
    """
    Create a single DataLoader from a directory using ImageFolder.
    
    Args:
        data_dir: Path to data directory (should contain class subdirectories)
        transform: Torchvision transform pipeline
        batch_size: Batch size for DataLoader (default: 64)
        shuffle: Whether to shuffle data (default: True)
        num_workers: Number of workers for data loading (default: 2)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        
    Returns:
        Tuple of (DataLoader, ImageFolder dataset)
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Create ImageFolder dataset
    dataset = datasets.ImageFolder(
        root=str(data_dir),
        transform=transform
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )
    
    return dataloader, dataset


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    image_size: int = 48,
    augmentation_level: str = 'baseline',
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train, validation, and test DataLoaders.
    
    Expected directory structure:
    data_dir/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   └── ...
    ├── val/
    └── test/
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size for all DataLoaders (default: 64)
        image_size: Target image size (default: 48)
        augmentation_level: 'baseline' or 'enhanced' (default: 'baseline')
        num_workers: Number of workers for data loading (default: 2)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx dict)
    """
    data_dir = Path(data_dir)
    
    # Get transforms
    if augmentation_level == 'enhanced':
        train_transform = get_enhanced_train_transforms(image_size=image_size)
    else:
        train_transform = get_train_transforms(image_size=image_size)
    
    val_transform = get_val_transforms(image_size=image_size)
    test_transform = get_test_transforms(image_size=image_size)
    
    # Create DataLoaders
    train_loader, train_dataset = create_dataloader(
        data_dir / 'train',
        train_transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader, val_dataset = create_dataloader(
        data_dir / 'val',
        val_transform,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader, test_dataset = create_dataloader(
        data_dir / 'test',
        test_transform,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Get class to index mapping
    class_to_idx = train_dataset.class_to_idx
    
    print(f"✅ DataLoaders created successfully!")
    print(f"   Train: {len(train_dataset):,} images")
    print(f"   Val:   {len(val_dataset):,} images")
    print(f"   Test:  {len(test_dataset):,} images")
    print(f"   Classes: {list(class_to_idx.keys())}")
    print(f"   Batch size: {batch_size}")
    print(f"   Augmentation: {augmentation_level}")
    
    return train_loader, val_loader, test_loader, class_to_idx


def create_dataloaders_from_config(
    config_path: str,
    data_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create DataLoaders from a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        data_dir: Optional override for data directory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx dict)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract data config
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Get parameters
    if data_dir is None:
        data_dir = data_config.get('processed_dir', 'data/processed')
    
    batch_size = training_config.get('batch_size', 64)
    image_size = data_config.get('image_size', 48)
    num_workers = training_config.get('num_workers', 2)
    augmentation_level = training_config.get('augmentation_level', 'baseline')
    
    return create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        augmentation_level=augmentation_level,
        num_workers=num_workers
    )


def get_class_counts(dataloader: DataLoader) -> Dict[str, int]:
    """
    Get the count of samples for each class in a DataLoader.
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        Dictionary mapping class names to counts
    """
    dataset = dataloader.dataset
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Count samples per class
    class_counts = {class_name: 0 for class_name in class_to_idx.keys()}
    
    for _, label in dataset.samples:
        class_name = idx_to_class[label]
        class_counts[class_name] += 1
    
    return class_counts


def calculate_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    
    Useful for weighted loss functions.
    
    Args:
        train_loader: Training DataLoader
        
    Returns:
        Tensor of class weights (one per class)
    """
    class_counts = get_class_counts(train_loader)
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = []
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def print_dataloader_info(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader
) -> None:
    """
    Print detailed information about DataLoaders.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    """
    print("\n" + "="*70)
    print("DATALOADER INFORMATION")
    print("="*70)
    
    for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        dataset = loader.dataset
        print(f"\n{name} Set:")
        print(f"  Total samples: {len(dataset):,}")
        print(f"  Batch size: {loader.batch_size}")
        print(f"  Number of batches: {len(loader)}")
        print(f"  Shuffle: {loader.sampler is None or hasattr(loader.sampler, 'shuffle')}")
        
        # Class distribution
        class_counts = get_class_counts(loader)
        print(f"  Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / len(dataset)) * 100
            print(f"    {class_name:10s}: {count:5,} ({percentage:5.1f}%)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    # Example usage
    import sys
    
    # Test with dummy path (replace with actual path)
    data_dir = 'data/processed'
    
    try:
        # Create DataLoaders
        train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
            data_dir=data_dir,
            batch_size=64,
            augmentation_level='baseline'
        )
        
        # Print information
        print_dataloader_info(train_loader, val_loader, test_loader)
        
        # Calculate class weights
        class_weights = calculate_class_weights(train_loader)
        print(f"\nClass weights: {class_weights}")
        
        # Test loading a batch
        images, labels = next(iter(train_loader))
        print(f"\n✅ Successfully loaded a batch:")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Image range: [{images.min():.2f}, {images.max():.2f}]")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the data directory exists with train/val/test subdirectories")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()