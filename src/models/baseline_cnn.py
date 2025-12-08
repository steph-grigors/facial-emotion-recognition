"""
Baseline CNN architecture for facial emotion recognition.

This module implements the baseline convolutional neural network that achieved
65.81% accuracy on the FER2013 dataset.

Architecture matches the working notebook:
- Input: 224x224 RGB (3 channels)
- 3 Convolutional blocks (32, 64, 128 filters)
- Adaptive pooling to 7x7
- 2 Fully connected layers
- Output: 7 emotion classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline CNN architecture for emotion recognition.
    
    Architecture:
        - 3 Convolutional blocks with increasing filters (32, 64, 128)
        - Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool
        - Adaptive pooling to ensure 7x7 spatial dimensions
        - 2 Fully connected layers with dropout
        - Output: 7 emotion classes
    
    Args:
        num_classes (int): Number of emotion classes (default: 7)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.5):
        super(BaselineCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Adaptive pooling to get consistent 7x7 output size
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Classifier layers
        # Input: 128 channels * 7 * 7 = 6272
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_model_info(self) -> dict:
        """
        Get model architecture information.
        
        Returns:
            dict: Dictionary containing model parameters and architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BaselineCNN',
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': '224x224 RGB',
            'input_channels': 3,
            'architecture': '3 Conv blocks + Adaptive Pooling + 2 FC layers',
            'fc_input_size': 128 * 7 * 7
        }


def test_baseline_cnn():
    """Test function to verify the model architecture."""
    model = BaselineCNN(num_classes=7)
    
    # Create dummy input (224x224 RGB)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    
    print("="*70)
    print("BASELINE CNN ARCHITECTURE TEST")
    print("="*70)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nModel Info:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Test with different input sizes to verify adaptive pooling
    print("\n" + "-"*70)
    print("Testing Adaptive Pooling with different input sizes:")
    print("-"*70)
    
    test_sizes = [224, 256, 300]
    for size in test_sizes:
        test_input = torch.randn(1, 3, size, size)
        test_output = model(test_input)
        print(f"  Input: {size}x{size} -> Output: {test_output.shape} ✓")
    
    assert output.shape == (batch_size, 7), f"Expected output shape (4, 7), got {output.shape}"
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_baseline_cnn()
