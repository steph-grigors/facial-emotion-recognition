"""
Baseline CNN architecture for facial emotion recognition.

This module implements the baseline convolutional neural network that achieved
65.81% accuracy on the FER2013 dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline CNN architecture for emotion recognition.
    
    Architecture:
        - 3 Convolutional blocks with increasing filters (32, 64, 128)
        - Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
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
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.25)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.25)
        
        # Calculate the size after convolutions
        # Input: 48x48 -> after pool1: 24x24 -> after pool2: 12x12 -> after pool3: 6x6
        self.fc_input_size = 128 * 6 * 6  # 4608
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 48, 48)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        
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
            'input_size': '48x48',
            'architecture': '3 Conv blocks + 2 FC layers'
        }


def test_baseline_cnn():
    """Test function to verify the model architecture."""
    model = BaselineCNN(num_classes=7)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 48, 48)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nModel Info:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")
    
    assert output.shape == (batch_size, 7), f"Expected output shape (4, 7), got {output.shape}"
    print("\n✓ Model test passed!")


if __name__ == "__main__":
    test_baseline_cnn()