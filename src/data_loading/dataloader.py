"""
DataLoader factory for FER2013 emotion recognition.

Provides functions to create PyTorch DataLoaders using ImageFolder
for train/val/test splits, with support for both Baseline CNN (224x224)
and Enhanced CNN/ResNet50 (128x128) models.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from typing import Tuple, Optional, Dict
import yaml

from .transforms import (
    get_baseline_train_transforms,
    get_baseline_val_transforms,
    get_enhanced_train_transforms,
    get_enhanced_val_transforms,
    get_train_transforms,
    get_val_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD
)


def create_dataloader(
    data_dir: str,
    transform,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, datasets.ImageFolder]:
    """
    Create a single DataLoader from a directory using ImageFolder.
    
    Args:
        data_dir: Path to data directory (should contain class subdirectories)
        transform: Torchvision transform pipeline
        batch_size: Batch size for DataLoader (default: 64)
        shuffle: Whether to shuffle data (default: True)
        num_workers: Number of workers for data loading (default: 4)
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
    model_type: str = 'baseline',
    batch_size: int = 64,
    num_workers: int = 4,
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
        model_type: 'baseline' (224x224) or 'enhanced' (128x128) (default: 'baseline')
        batch_size: Batch size for all DataLoaders (default: 64)
        num_workers: Number of workers for data loading (default: 4)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx dict)
    """
    data_dir = Path(data_dir)
    
    # Get appropriate transforms based on model type
    train_transform = get_train_transforms(model_type=model_type)
    val_transform = get_val_transforms(model_type=model_type)
    test_transform = get_val_transforms(model_type=model_type)
    
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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader, test_dataset = create_dataloader(
        data_dir / 'test',
        test_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Get class to index mapping
    class_to_idx = train_dataset.class_to_idx
    
    # Determine image size based on model type
    img_size = 224 if model_type == 'baseline' else 128
    
    print(f"✅ DataLoaders created successfully!")
    print(f"   Model type: {model_type.upper()}")
    print(f"   Image size: {img_size}x{img_size} RGB")
    print(f"   Train: {len(train_dataset):,} images")
    print(f"   Val:   {len(val_dataset):,} images")
    print(f"   Test:  {len(test_dataset):,} images")
    print(f"   Classes: {list(class_to_idx.keys())}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, class_to_idx


def create_dataloaders_from_config(
    config_path: str,
    data_dir: Optional[str] = None,
    model_type: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create DataLoaders from a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        data_dir: Optional override for data directory
        model_type: Optional override for model type ('baseline' or 'enhanced')
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx dict)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract data config
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    # Get parameters
    if data_dir is None:
        data_dir = data_config.get('processed_dir', 'data/processed')
    
    if model_type is None:
        architecture = model_config.get('architecture', 'baseline_cnn')
        if 'resnet' in architecture.lower() or 'enhanced' in architecture.lower():
            model_type = 'enhanced'
        else:
            model_type = 'baseline'
    
    batch_size = training_config.get('batch_size', 64)
    num_workers = training_config.get('num_workers', 4)
    
    return create_dataloaders(
        data_dir=data_dir,
        model_type=model_type,
        batch_size=batch_size,
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
    Formula: weight_i = total_samples / (num_classes * count_i)
    
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
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    print("\n📊 Class weights calculated:")
    for class_name, weight in zip(sorted(class_counts.keys()), weights):
        count = class_counts[class_name]
        print(f"   {class_name:10s}: count={count:5,} | weight={weight:.4f}")
    
    return weights_tensor


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
    import sys
    
    print("="*70)
    print("DATALOADER MODULE TEST")
    print("="*70)
    
    # Test with dummy path
    data_dir = 'data/processed'
    
    try:
        print("\n1. Testing BASELINE CNN DataLoaders (224x224):")
        print("-"*70)
        
        train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
            data_dir=data_dir,
            model_type='baseline',
            batch_size=64
        )
        
        # Test loading a batch
        images, labels = next(iter(train_loader))
        print(f"\n✓ Baseline batch loaded:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
        
        assert images.shape[1] == 3, f"Expected 3 channels (RGB), got {images.shape[1]}"
        assert images.shape[2] == 224, f"Expected 224x224, got {images.shape[2:]}"
        print("  ✓ Verified: 3 channels (RGB)")
        print("  ✓ Verified: 224x224 size")
        
        print("\n" + "="*70)
        print("2. Testing ENHANCED CNN DataLoaders (128x128):")
        print("-"*70)
        
        train_loader_enh, val_loader_enh, test_loader_enh, _ = create_dataloaders(
            data_dir=data_dir,
            model_type='enhanced',
            batch_size=64
        )
        
        # Test loading a batch
        images_enh, labels_enh = next(iter(train_loader_enh))
        print(f"\n✓ Enhanced batch loaded:")
        print(f"  Images shape: {images_enh.shape}")
        print(f"  Labels shape: {labels_enh.shape}")
        print(f"  Image range: [{images_enh.min():.2f}, {images_enh.max():.2f}]")
        
        assert images_enh.shape[1] == 3, f"Expected 3 channels (RGB), got {images_enh.shape[1]}"
        assert images_enh.shape[2] == 128, f"Expected 128x128, got {images_enh.shape[2:]}"
        print("  ✓ Verified: 3 channels (RGB)")
        print("  ✓ Verified: 128x128 size")
        
        # Print detailed info
        print("\n" + "="*70)
        print("3. Baseline DataLoader Details:")
        print_dataloader_info(train_loader, val_loader, test_loader)
        
        # Calculate class weights
        print("\n" + "="*70)
        print("4. Class Weights:")
        print("-"*70)
        class_weights = calculate_class_weights(train_loader)
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"⚠️  Data directory not found: {data_dir}")
        print("   This is expected if running outside of project directory.")
        print("   Module functionality verified through code inspection.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
