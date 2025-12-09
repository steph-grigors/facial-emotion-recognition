"""
Scripts module for Facial Emotion Recognition application.

This module contains utility scripts for model training, data processing,
and project setup for the facial emotion recognition system.
"""

__version__ = "1.0.0"
__author__ = "Stéphan Georis"

# Training scripts
from .train import train_baseline_cnn, train_enhanced_cnn, evaluate_model_on_test
from .train_model import main as train_model_main

# Utility scripts
from .inspect_model import inspect_model, detect_architecture
from .quick_inference import main as quick_inference_main

# Setup and testing scripts
from .setup_project import main as setup_project_main
from .test_data_loading import (
    example_basic_usage,
    example_enhanced_augmentation,
    example_class_weights,
    example_config_based,
    example_detailed_info,
    example_transform_comparison
)

__all__ = [
    # Training functions
    'train_baseline_cnn',
    'train_enhanced_cnn',
    'evaluate_model_on_test',
    'train_model_main',
    
    # Utility functions
    'inspect_model',
    'detect_architecture',
    'quick_inference_main',
    
    # Setup functions
    'setup_project_main',
    
    # Data loading examples
    'example_basic_usage',
    'example_enhanced_augmentation',
    'example_class_weights',
    'example_config_based',
    'example_detailed_info',
    'example_transform_comparison',
]