"""
Model factory for creating models from configuration.

This module provides a factory pattern to instantiate models based on
configuration parameters.
"""

from typing import Dict, Any
import torch.nn as nn

from .baseline_cnn import BaselineCNN
from .enhanced_cnn import EnhancedCNN
from .transfer_learning import TransferLearningModel


class ModelFactory:
    """
    Factory class for creating model instances.
    
    Supports:
        - baseline_cnn: Simple baseline CNN
        - enhanced_cnn: Enhanced CNN with residual blocks and attention
        - resnet18, resnet34, resnet50: ResNet variants
        - efficientnet_b0, efficientnet_b1, efficientnet_b2: EfficientNet variants
    """
    
    SUPPORTED_MODELS = [
        'baseline_cnn',
        'enhanced_cnn',
        'resnet18',
        'resnet34', 
        'resnet50',
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b2',
    ]
    
    @staticmethod
    def create_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create a model instance based on name and configuration.
        
        Args:
            model_name (str): Name of the model to create
            config (dict): Configuration dictionary with model parameters
        
        Returns:
            nn.Module: Instantiated model
        
        Raises:
            ValueError: If model_name is not supported
        
        Examples:
            >>> config = {'num_classes': 7, 'dropout_rate': 0.5}
            >>> model = ModelFactory.create_model('baseline_cnn', config)
            
            >>> config = {'num_classes': 7, 'pretrained': True, 'freeze_base': False}
            >>> model = ModelFactory.create_model('resnet50', config)
        """
        model_name = model_name.lower()
        
        if model_name not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Supported models: {ModelFactory.SUPPORTED_MODELS}"
            )
        
        # Get common parameters
        num_classes = config.get('num_classes', 7)
        
        # Create model based on type
        if model_name == 'baseline_cnn':
            dropout_rate = config.get('dropout_rate', 0.5)
            return BaselineCNN(
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
        
        elif model_name == 'enhanced_cnn':
            dropout_rate = config.get('dropout_rate', 0.5)
            use_attention = config.get('use_attention', True)
            return EnhancedCNN(
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                use_attention=use_attention
            )
        
        elif model_name in ['resnet18', 'resnet34', 'resnet50', 
                           'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
            pretrained = config.get('pretrained', True)
            freeze_base = config.get('freeze_base', False)
            return TransferLearningModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_base=freeze_base
            )
    
    @staticmethod
    def list_models() -> list:
        """
        Get list of all supported models.
        
        Returns:
            list: List of supported model names
        """
        return ModelFactory.SUPPORTED_MODELS.copy()
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name (str): Name of the model
        
        Returns:
            dict: Information about the model architecture and parameters
        """
        model_name = model_name.lower()
        
        info = {
            'baseline_cnn': {
                'description': 'Baseline CNN with 3 conv blocks',
                'parameters': ['num_classes', 'dropout_rate'],
                'input_size': '48x48',
                'grayscale': True,
                'pretrained': False
            },
            'enhanced_cnn': {
                'description': 'Enhanced CNN with residual blocks and attention',
                'parameters': ['num_classes', 'dropout_rate', 'use_attention'],
                'input_size': '48x48',
                'grayscale': True,
                'pretrained': False
            },
            'resnet18': {
                'description': 'ResNet-18 adapted for grayscale',
                'parameters': ['num_classes', 'pretrained', 'freeze_base'],
                'input_size': '48x48',
                'grayscale': True,
                'pretrained': True
            },
            'resnet34': {
                'description': 'ResNet-34 adapted for grayscale',
                'parameters': ['num_classes', 'pretrained', 'freeze_base'],
                'input_size': '48x48',
                'grayscale': True,
                'pretrained': True
            },
            'resnet50': {
                'description': 'ResNet-50 adapted for grayscale',
                'parameters': ['num_classes', 'pretrained', 'freeze_base'],
                'input_size': '48x48',
                'grayscale': True,
                'pretrained': True
            },
            'efficientnet_b0': {
                'description': 'EfficientNet-B0 adapted for grayscale',
                'parameters': ['num_classes', 'pretrained', 'freeze_base'],
                'input_size': '48x48',
                'grayscale': True,
                'pretrained': True
            },
            'efficientnet_b1': {
                'description': 'EfficientNet-B1 adapted for grayscale',
                'parameters': ['num_classes', 'pretrained', 'freeze_base'],
                'input_size': '48x48',
                'grayscale': True,
                'pretrained': True
            },
            'efficientnet_b2': {
                'description': 'EfficientNet-B2 adapted for grayscale',
                'parameters': ['num_classes', 'pretrained', 'freeze_base'],
                'input_size': '48x48',
                'grayscale': True,
                'pretrained': True
            },
        }
        
        return info.get(model_name, {})


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Convenience function to create a model.
    
    Args:
        model_name (str): Name of the model to create
        **kwargs: Model configuration parameters
    
    Returns:
        nn.Module: Instantiated model
    
    Examples:
        >>> model = create_model('baseline_cnn', num_classes=7, dropout_rate=0.5)
        >>> model = create_model('resnet50', num_classes=7, pretrained=True)
    """
    return ModelFactory.create_model(model_name, kwargs)


def print_all_models_info():
    """Print information about all supported models."""
    print("=" * 80)
    print("SUPPORTED MODELS")
    print("=" * 80)
    
    for model_name in ModelFactory.list_models():
        info = ModelFactory.get_model_info(model_name)
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"Description: {info.get('description', 'N/A')}")
        print(f"Parameters: {', '.join(info.get('parameters', []))}")
        print(f"Input Size: {info.get('input_size', 'N/A')}")
        print(f"Grayscale: {info.get('grayscale', 'N/A')}")
        print(f"Pretrained Available: {info.get('pretrained', 'N/A')}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Print all models info
    print_all_models_info()
    
    # Test creating different models
    print("\n" + "=" * 80)
    print("TESTING MODEL CREATION")
    print("=" * 80)
    
    # Test baseline CNN
    print("\nCreating Baseline CNN...")
    model1 = create_model('baseline_cnn', num_classes=7, dropout_rate=0.5)
    print(f"✓ {model1.get_model_info()['model_name']} created successfully")
    print(f"  Total parameters: {model1.get_model_info()['total_parameters']:,}")
    
    # Test enhanced CNN
    print("\nCreating Enhanced CNN...")
    model2 = create_model('enhanced_cnn', num_classes=7, use_attention=True)
    print(f"✓ {model2.get_model_info()['model_name']} created successfully")
    print(f"  Total parameters: {model2.get_model_info()['total_parameters']:,}")
    
    # Test transfer learning
    print("\nCreating ResNet50 (no pretrained weights for testing)...")
    model3 = create_model('resnet50', num_classes=7, pretrained=False)
    print(f"✓ {model3.get_model_info()['model_name']} created successfully")
    print(f"  Total parameters: {model3.get_model_info()['total_parameters']:,}")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)