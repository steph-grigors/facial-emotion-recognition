#!/usr/bin/env python3
"""
Example usage of the data_loading module.

Demonstrates how to:
1. Create DataLoaders with different augmentation levels
2. Calculate class weights
3. Load data from config files
4. Print DataLoader statistics
5. Test loading batches
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import (
    create_dataloaders,
    create_dataloaders_from_config,
    calculate_class_weights,
    print_dataloader_info,
    get_train_transforms,
    get_enhanced_train_transforms
)


def example_basic_usage():
    """Example 1: Basic DataLoader creation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic DataLoader Creation")
    print("="*70)
    
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir='data/processed',
        batch_size=64,
        image_size=48,
        augmentation_level='baseline',
        num_workers=2
    )
    
    # Load one batch to verify
    images, labels = next(iter(train_loader))
    print(f"\n✅ Successfully loaded a batch:")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Image range: [{images.min():.2f}, {images.max():.2f}]")
    
    return train_loader, val_loader, test_loader


def example_enhanced_augmentation():
    """Example 2: Enhanced augmentation for better accuracy"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Enhanced Augmentation (80.21% accuracy)")
    print("="*70)
    
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir='data/processed',
        batch_size=64,
        augmentation_level='enhanced',  # Stronger augmentation
        num_workers=2
    )
    
    print("\n✨ Using enhanced augmentation for improved performance!")
    
    return train_loader, val_loader, test_loader


def example_class_weights():
    """Example 3: Calculate class weights for imbalanced dataset"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Class Weights for Imbalanced Dataset")
    print("="*70)
    
    train_loader, _, _, _ = create_dataloaders(
        data_dir='data/processed',
        batch_size=64
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader)
    
    print(f"\n📊 Class Weights:")
    class_to_idx = train_loader.dataset.class_to_idx
    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"   {class_name:10s}: {class_weights[idx]:.4f}")
    
    print(f"\n💡 Use these weights in your loss function:")
    print(f"   criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))")
    
    return class_weights


def example_config_based():
    """Example 4: Load from configuration file"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Config-Based DataLoader Creation")
    print("="*70)
    
    try:
        train_loader, val_loader, test_loader, class_to_idx = create_dataloaders_from_config(
            config_path='configs/config.yaml'
        )
        print("\n✅ Successfully loaded DataLoaders from config file!")
        
    except FileNotFoundError:
        print("\n⚠️  Config file not found. Using default parameters.")
        train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
            data_dir='data/processed'
        )
    
    return train_loader, val_loader, test_loader


def example_detailed_info():
    """Example 5: Print detailed DataLoader information"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Detailed DataLoader Information")
    print("="*70)
    
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_dir='data/processed',
        batch_size=64
    )
    
    # Print comprehensive info
    print_dataloader_info(train_loader, val_loader, test_loader)


def example_transform_comparison():
    """Example 6: Compare baseline vs enhanced transforms"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Transform Pipeline Comparison")
    print("="*70)
    
    print("\n📋 Baseline Transforms:")
    print("-" * 70)
    baseline_transform = get_train_transforms(image_size=48)
    print(baseline_transform)
    
    print("\n📋 Enhanced Transforms:")
    print("-" * 70)
    enhanced_transform = get_enhanced_train_transforms(image_size=48)
    print(enhanced_transform)
    
    print("\n💡 Key Differences:")
    print("   • Rotation: 10° → 15°")
    print("   • Added: RandomAffine (translation, scale)")
    print("   • ColorJitter: 0.2 → 0.3")
    print("   • Result: 65.81% → 80.21% accuracy!")


def main():
    """Run all examples"""
    print("="*70)
    print("DATA LOADING MODULE - USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Example 1: Basic usage
        train_loader, val_loader, test_loader = example_basic_usage()
        
        # Example 2: Enhanced augmentation
        example_enhanced_augmentation()
        
        # Example 3: Class weights
        example_class_weights()
        
        # Example 4: Config-based loading
        example_config_based()
        
        # Example 5: Detailed info
        example_detailed_info()
        
        # Example 6: Transform comparison
        example_transform_comparison()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n💡 Next steps:")
        print("   1. Use these DataLoaders in your training pipeline")
        print("   2. Experiment with baseline vs enhanced augmentation")
        print("   3. Use class weights for better performance on imbalanced data")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you've run the data preprocessing pipeline first:")
        print("   python scripts/setup_project.py --data-dir path/to/raw/data")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
