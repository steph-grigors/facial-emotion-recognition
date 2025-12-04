"""
Enhanced CNN architecture for facial emotion recognition.

This module implements an improved CNN with additional features like
residual connections, deeper architecture, and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv -> BatchNorm -> ReLU -> Dropout.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of convolving kernel
        padding (int): Padding added to input
        dropout_rate (float): Dropout probability
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1, dropout_rate: float = 0.25):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Args:
        channels (int): Number of channels
        dropout_rate (float): Dropout probability
    """
    
    def __init__(self, channels: int, dropout_rate: float = 0.25):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += residual
        out = F.relu(out)
        out = self.dropout2(out)
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important regions.
    """
    
    def __init__(self):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        attention = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        # Multiply attention weights with input
        return x * attention


class EnhancedCNN(nn.Module):
    """
    Enhanced CNN architecture with residual connections and attention.
    
    Architecture:
        - 4 Convolutional blocks with increasing filters (32, 64, 128, 256)
        - Residual blocks after each conv block
        - Spatial attention mechanism
        - 3 Fully connected layers with dropout
        - Output: 7 emotion classes
    
    Args:
        num_classes (int): Number of emotion classes (default: 7)
        dropout_rate (float): Dropout probability (default: 0.5)
        use_attention (bool): Whether to use spatial attention (default: True)
    """
    
    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.5, 
                 use_attention: bool = True):
        super(EnhancedCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # First block: 1 -> 32
        self.conv_block1 = ConvBlock(1, 32, dropout_rate=0.25)
        self.residual1 = ResidualBlock(32, dropout_rate=0.25)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block: 32 -> 64
        self.conv_block2 = ConvBlock(32, 64, dropout_rate=0.25)
        self.residual2 = ResidualBlock(64, dropout_rate=0.25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third block: 64 -> 128
        self.conv_block3 = ConvBlock(64, 128, dropout_rate=0.3)
        self.residual3 = ResidualBlock(128, dropout_rate=0.3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth block: 128 -> 256
        self.conv_block4 = ConvBlock(128, 256, dropout_rate=0.3)
        self.residual4 = ResidualBlock(256, dropout_rate=0.3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Spatial attention
        if use_attention:
            self.attention = SpatialAttention()
        
        # Calculate the size after convolutions
        # Input: 48x48 -> 24x24 -> 12x12 -> 6x6 -> 3x3
        self.fc_input_size = 256 * 3 * 3  # 2304
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 48, 48)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # First block
        x = self.conv_block1(x)
        x = self.residual1(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv_block2(x)
        x = self.residual2(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv_block3(x)
        x = self.residual3(x)
        x = self.pool3(x)
        
        # Fourth block
        x = self.conv_block4(x)
        x = self.residual4(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
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
            'model_name': 'EnhancedCNN',
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'use_attention': self.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': '48x48',
            'architecture': '4 Conv blocks + Residual + Attention + 3 FC layers'
        }


def test_enhanced_cnn():
    """Test function to verify the model architecture."""
    model = EnhancedCNN(num_classes=7, use_attention=True)
    
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
    test_enhanced_cnn()