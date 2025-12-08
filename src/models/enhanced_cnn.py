"""
Enhanced CNN with ResNet50 backbone for emotion recognition.

Uses transfer learning with ResNet50 pretrained on ImageNet.
Input: 128×128 RGB images
Output: 7 emotion classes
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


class EnhancedCNN(nn.Module):
    """
    Enhanced CNN using ResNet50 with transfer learning.
    
    Architecture:
    - Backbone: ResNet50 pretrained on ImageNet
    - Input: 128×128 RGB (3 channels)
    - Output: 7 emotion classes
    
    Args:
        num_classes: Number of output classes (default: 7)
        pretrained: Whether to load ImageNet pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone during training (default: True)
        dropout_rate: Dropout rate for classifier (default: 0.5)
    
    Example:
        >>> model = EnhancedCNN(num_classes=7, pretrained=True, freeze_backbone=True)
        >>> x = torch.randn(4, 3, 128, 128)
        >>> out = model(x)
        >>> print(out.shape)  # (4, 7)
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5
    ):
        super(EnhancedCNN, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # Load ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get input features for classifier
        num_features = self.backbone.fc.in_features
        
        # Replace classifier with custom one
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier."""
        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze only the classifier (fc layer)
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'EnhancedCNN (ResNet50)',
            'base_architecture': 'ResNet50',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'input_size': '128×128 RGB',
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with automatic key mapping for backward compatibility.
        
        Handles multiple formats:
        1. New format with dropout: 'backbone.fc.0.weight' (dropout), 'backbone.fc.1.weight' (linear)
        2. Old format with simple fc: 'backbone.fc.weight' (just linear)
        3. Raw ResNet50 format: 'fc.weight' (no backbone prefix)
        4. Raw ResNet50 with no prefix and simple fc: 'fc.weight'
        """
        # Step 1: Handle missing 'backbone.' prefix
        needs_backbone_prefix = False
        if 'conv1.weight' in state_dict:
            needs_backbone_prefix = True
        elif 'fc.weight' in state_dict and 'backbone.fc.weight' not in state_dict and 'backbone.fc.1.weight' not in state_dict:
            needs_backbone_prefix = True
        
        if needs_backbone_prefix:
            # Add 'backbone.' prefix to all keys
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f'backbone.{key}'
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        # Step 2: Handle simple Linear vs Sequential(Dropout, Linear) for fc layer
        # Check if we have simple fc (backbone.fc.weight) but model expects Sequential (backbone.fc.1.weight)
        has_simple_fc = 'backbone.fc.weight' in state_dict and 'backbone.fc.bias' in state_dict
        model_expects_sequential = any('backbone.fc.0' in k or 'backbone.fc.1' in k for k in self.state_dict().keys())
        
        if has_simple_fc and model_expects_sequential:
            # Convert simple Linear to Sequential format
            # backbone.fc.weight -> backbone.fc.1.weight (skip dropout layer 0)
            # backbone.fc.bias -> backbone.fc.1.bias
            fc_weight = state_dict.pop('backbone.fc.weight')
            fc_bias = state_dict.pop('backbone.fc.bias')
            state_dict['backbone.fc.1.weight'] = fc_weight
            state_dict['backbone.fc.1.bias'] = fc_bias
        
        # Now load with the corrected keys
        return super(EnhancedCNN, self).load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":
    """Test the model."""
    print("="*70)
    print("TESTING ENHANCED CNN (ResNet50)")
    print("="*70)
    
    # Test model creation
    print("\n1. Creating model...")
    model = EnhancedCNN(num_classes=7, pretrained=False, freeze_backbone=True)
    
    info = model.get_model_info()
    print(f"   ✅ Model created")
    print(f"   Architecture: {info['model_name']}")
    print(f"   Total parameters: {info['total_parameters']:,}")
    print(f"   Trainable parameters: {info['trainable_parameters']:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 128)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"   ✅ Forward pass successful")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 7), f"Wrong output shape: {output.shape}"
    
    # Test freeze/unfreeze
    print("\n3. Testing freeze/unfreeze...")
    
    model.freeze_backbone()
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Frozen trainable params: {trainable_frozen:,}")
    
    model.unfreeze_backbone()
    trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Unfrozen trainable params: {trainable_unfrozen:,}")
    
    assert trainable_frozen < trainable_unfrozen, "Unfreeze didn't work!"
    print(f"   ✅ Freeze/unfreeze working correctly")
    
    # Test loading MULTIPLE formats
    print("\n4. Testing state dict loading (all formats)...")
    
    # Format 1: Raw ResNet50 with simple Linear fc
    print("   Testing Format 1: Raw ResNet50 (fc.weight, fc.bias)...")
    format1_state_dict = {}
    for name, param in model.state_dict().items():
        # Remove 'backbone.' prefix and convert Sequential fc to simple Linear
        if name.startswith('backbone.fc.1.'):
            # backbone.fc.1.weight -> fc.weight
            new_key = name.replace('backbone.fc.1.', 'fc.')
        elif name.startswith('backbone.fc.0.'):
            # Skip dropout layer
            continue
        elif name.startswith('backbone.'):
            new_key = name.replace('backbone.', '')
        else:
            new_key = name
        format1_state_dict[new_key] = param.clone()
    
    model_test1 = EnhancedCNN(num_classes=7, pretrained=False, freeze_backbone=False)
    model_test1.load_state_dict(format1_state_dict, strict=True)
    print(f"   ✅ Format 1 loaded successfully")
    
    # Format 2: With backbone prefix but simple Linear fc
    print("   Testing Format 2: backbone.fc.weight, backbone.fc.bias...")
    format2_state_dict = {}
    for name, param in model.state_dict().items():
        # Convert Sequential fc to simple Linear but keep backbone prefix
        if name.startswith('backbone.fc.1.'):
            new_key = name.replace('backbone.fc.1.', 'backbone.fc.')
        elif name.startswith('backbone.fc.0.'):
            continue
        else:
            new_key = name
        format2_state_dict[new_key] = param.clone()
    
    model_test2 = EnhancedCNN(num_classes=7, pretrained=False, freeze_backbone=False)
    model_test2.load_state_dict(format2_state_dict, strict=True)
    print(f"   ✅ Format 2 loaded successfully")
    
    # Format 3: Full new format with Sequential
    print("   Testing Format 3: backbone.fc.0.*, backbone.fc.1.*...")
    format3_state_dict = model.state_dict()
    model_test3 = EnhancedCNN(num_classes=7, pretrained=False, freeze_backbone=False)
    model_test3.load_state_dict(format3_state_dict, strict=True)
    print(f"   ✅ Format 3 loaded successfully")
    
    # Test forward passes are identical
    print("\n5. Testing output consistency across formats...")
    x_test = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out1 = model_test1(x_test)
        out2 = model_test2(x_test)
        out3 = model_test3(x_test)
    
    print(f"   Output shapes: {out1.shape}, {out2.shape}, {out3.shape}")
    assert out1.shape == out2.shape == out3.shape == (2, 7)
    print(f"   ✅ All formats produce correct output shape")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nKey Features:")
    print("  ✓ ResNet50 backbone with ImageNet pretraining")
    print("  ✓ 128×128 RGB input")
    print("  ✓ Freeze/unfreeze for two-phase training")
    print("  ✓ Handles 3 different checkpoint formats:")
    print("    - Raw ResNet50 (no prefix, simple fc)")
    print("    - With backbone prefix (simple fc)")
    print("    - Full new format (Sequential fc with dropout)")
    print("  ✓ Fully backward compatible")