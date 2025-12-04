"""
Transfer learning models for facial emotion recognition.

This module provides wrappers for pretrained models (ResNet, EfficientNet)
adapted for emotion recognition tasks.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


class TransferLearningModel(nn.Module):
    """
    Transfer learning wrapper for pretrained models.
    
    Supports:
        - ResNet (resnet18, resnet34, resnet50)
        - EfficientNet (efficientnet_b0, efficientnet_b1, efficientnet_b2)
    
    Args:
        model_name (str): Name of the pretrained model
        num_classes (int): Number of output classes (default: 7)
        pretrained (bool): Whether to use pretrained weights (default: True)
        freeze_base (bool): Whether to freeze base model layers (default: False)
    """
    
    def __init__(
        self, 
        model_name: Literal['resnet18', 'resnet34', 'resnet50', 
                           'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'],
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_base: bool = False
    ):
        super(TransferLearningModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_base = freeze_base
        
        # Load pretrained model
        if model_name.startswith('resnet'):
            self.base_model, self.classifier = self._load_resnet(model_name)
        elif model_name.startswith('efficientnet'):
            self.base_model, self.classifier = self._load_efficientnet(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Optionally freeze base model
        if freeze_base:
            self._freeze_base_model()
    
    def _load_resnet(self, model_name: str):
        """Load ResNet model and modify for grayscale input."""
        # Get the appropriate ResNet model
        if model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if self.pretrained else None)
        elif model_name == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1' if self.pretrained else None)
        elif model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2' if self.pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet variant: {model_name}")
        
        # Modify first conv layer for grayscale (1 channel instead of 3)
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # If pretrained, average the weights from 3 channels to 1
        if self.pretrained:
            with torch.no_grad():
                model.conv1.weight = nn.Parameter(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )
        
        # Replace classifier
        num_features = model.fc.in_features
        classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Remove original fc layer
        model.fc = nn.Identity()
        
        return model, classifier
    
    def _load_efficientnet(self, model_name: str):
        """Load EfficientNet model and modify for grayscale input."""
        # Get the appropriate EfficientNet model
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1' if self.pretrained else None)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(weights='IMAGENET1K_V1' if self.pretrained else None)
        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(weights='IMAGENET1K_V1' if self.pretrained else None)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
        
        # Modify first conv layer for grayscale
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # If pretrained, average the weights from 3 channels to 1
        if self.pretrained:
            with torch.no_grad():
                model.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Replace classifier
        num_features = model.classifier[1].in_features
        classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Remove original classifier
        model.classifier = nn.Identity()
        
        return model, classifier
    
    def _freeze_base_model(self):
        """Freeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 48, 48)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        features = self.base_model(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self) -> dict:
        """
        Get model architecture information.
        
        Returns:
            dict: Dictionary containing model parameters and architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'freeze_base': self.freeze_base,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'input_size': '48x48',
            'architecture': f'{self.model_name} + Custom Classifier'
        }


def test_transfer_learning():
    """Test function to verify the transfer learning models."""
    print("Testing ResNet50...")
    model_resnet = TransferLearningModel('resnet50', num_classes=7, pretrained=False)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 48, 48)
    
    # Forward pass
    output = model_resnet(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nResNet50 Info:")
    for key, value in model_resnet.get_model_info().items():
        print(f"  {key}: {value}")
    
    assert output.shape == (batch_size, 7), f"Expected output shape (4, 7), got {output.shape}"
    print("✓ ResNet50 test passed!")
    
    print("\n" + "="*50 + "\n")
    
    print("Testing EfficientNet-B0...")
    model_efficientnet = TransferLearningModel('efficientnet_b0', num_classes=7, pretrained=False)
    
    # Forward pass
    output = model_efficientnet(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nEfficientNet-B0 Info:")
    for key, value in model_efficientnet.get_model_info().items():
        print(f"  {key}: {value}")
    
    assert output.shape == (batch_size, 7), f"Expected output shape (4, 7), got {output.shape}"
    print("✓ EfficientNet-B0 test passed!")


if __name__ == "__main__":
    test_transfer_learning()