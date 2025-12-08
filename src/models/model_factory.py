"""
Model factory for creating emotion recognition models.

Handles:
- Baseline CNN (custom 3-layer CNN, 224x224, 65.81% accuracy)
- Enhanced CNN (ResNet50 transfer learning, 128x128, 80.21% accuracy)
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from .baseline_cnn import BaselineCNN
from .enhanced_cnn import EnhancedCNN


ModelType = Literal['baseline', 'enhanced', 'resnet50']


def create_model(
    model_type: ModelType = 'baseline',
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.5,
    device: Optional[str] = None
) -> nn.Module:
    """
    Factory function to create emotion recognition models.
    
    Args:
        model_type: Type of model ('baseline', 'enhanced', or 'resnet50')
                    'enhanced' and 'resnet50' are aliases
        num_classes: Number of emotion classes (default: 7)
        pretrained: Use pretrained weights (only for enhanced/resnet50)
        freeze_backbone: Freeze backbone initially (only for enhanced/resnet50)
        dropout_rate: Dropout rate (only for baseline)
        device: Device to move model to ('cuda' or 'cpu')
    
    Returns:
        PyTorch model
    """
    if model_type.lower() in ['baseline', 'baseline_cnn']:
        print("Creating Baseline CNN...")
        model = BaselineCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
    elif model_type.lower() in ['enhanced', 'enhanced_cnn', 'resnet50', 'resnet']:
        print("Creating Enhanced CNN (ResNet50)...")
        model = EnhancedCNN(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Use 'baseline' or 'enhanced'"
        )
    
    # Move to device if specified
    if device:
        model = model.to(device)
        print(f"Model moved to: {device}")
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    print(f"  Architecture: {info['model_name']}")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  Input: {info['input_size']}")
    
    return model


def load_model(
    checkpoint_path: str,
    model_type: ModelType = 'baseline',
    num_classes: int = 7,
    device: Optional[str] = None,
    strict: bool = True
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_type: Type of model ('baseline' or 'enhanced')
        num_classes: Number of emotion classes
        device: Device to load model to
        strict: Whether to strictly enforce state dict keys match
    
    Returns:
        Loaded PyTorch model
    """
    # Create model architecture
    if model_type.lower() in ['baseline', 'baseline_cnn']:
        model = BaselineCNN(num_classes=num_classes)
    elif model_type.lower() in ['enhanced', 'enhanced_cnn', 'resnet50']:
        # Load with unfrozen backbone for inference
        model = EnhancedCNN(
            num_classes=num_classes,
            pretrained=False,  # Don't need pretrained weights if loading checkpoint
            freeze_backbone=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Load checkpoint
    if device:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=strict)
    
    if device:
        model = model.to(device)
    
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    
    return model


def get_model_config(model_type: ModelType) -> dict:
    """
    Get recommended configuration for a model type.
    
    Args:
        model_type: Type of model ('baseline' or 'enhanced')
    
    Returns:
        Dictionary with recommended settings
    """
    if model_type.lower() in ['baseline', 'baseline_cnn']:
        return {
            'model_type': 'baseline',
            'input_size': 224,
            'input_channels': 3,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 50,
            'use_mixup': False,
            'use_cutmix': False,
            'two_phase_training': False,
            'expected_accuracy': 0.6581  # 65.81%
        }
    
    elif model_type.lower() in ['enhanced', 'enhanced_cnn', 'resnet50']:
        return {
            'model_type': 'enhanced',
            'input_size': 128,
            'input_channels': 3,
            'batch_size': 64,
            'phase1_lr': 1e-3,
            'phase2_lr': 3e-4,
            'phase1_epochs': 5,
            'phase2_epochs': 25,
            'use_mixup': True,
            'use_cutmix': True,
            'mixup_alpha': 0.4,
            'cutmix_alpha': 0.4,
            'aug_prob': 0.5,
            'two_phase_training': True,
            'expected_accuracy': 0.8021  # 80.21%
        }
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def test_model_factory():
    """Test model factory functions."""
    print("="*70)
    print("MODEL FACTORY TEST")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: Create Baseline CNN
    print("\n1. Creating Baseline CNN:")
    print("-"*70)
    baseline_model = create_model(
        model_type='baseline',
        num_classes=7,
        device=device
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = baseline_model(dummy_input)
    print(f"\n✓ Forward pass test:")
    print(f"  Input: {dummy_input.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == (2, 7), "Output shape mismatch"
    
    # Test 2: Create Enhanced CNN (frozen)
    print("\n" + "="*70)
    print("2. Creating Enhanced CNN (ResNet50) with frozen backbone:")
    print("-"*70)
    enhanced_model_frozen = create_model(
        model_type='enhanced',
        num_classes=7,
        pretrained=True,
        freeze_backbone=True,
        device=device
    )
    
    # Test forward pass
    dummy_input_enh = torch.randn(2, 3, 128, 128).to(device)
    output_enh = enhanced_model_frozen(dummy_input_enh)
    print(f"\n✓ Forward pass test:")
    print(f"  Input: {dummy_input_enh.shape}")
    print(f"  Output: {output_enh.shape}")
    assert output_enh.shape == (2, 7), "Output shape mismatch"
    
    # Test 3: Create Enhanced CNN (unfrozen)
    print("\n" + "="*70)
    print("3. Creating Enhanced CNN (ResNet50) with unfrozen backbone:")
    print("-"*70)
    enhanced_model_unfrozen = create_model(
        model_type='enhanced',
        num_classes=7,
        pretrained=True,
        freeze_backbone=False,
        device=device
    )
    
    # Test 4: Get model configs
    print("\n" + "="*70)
    print("4. Getting recommended configurations:")
    print("-"*70)
    
    print("\nBaseline CNN config:")
    baseline_config = get_model_config('baseline')
    for key, value in baseline_config.items():
        print(f"  {key}: {value}")
    
    print("\nEnhanced CNN config:")
    enhanced_config = get_model_config('enhanced')
    for key, value in enhanced_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ ALL MODEL FACTORY TESTS PASSED!")
    print("="*70)
    
    print("\nSummary:")
    print("  ✓ Baseline CNN: 224x224 RGB, custom architecture")
    print("  ✓ Enhanced CNN: 128x128 RGB, ResNet50 pretrained")
    print("  ✓ Both models support 7 emotion classes")
    print("  ✓ Forward passes validated")


if __name__ == "__main__":
    test_model_factory()
