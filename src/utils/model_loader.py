"""
Model loading utilities with automatic cache detection.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


def find_model_checkpoint(
    model_name: str,
    models_dir: str = 'models',
    checkpoints_dir: str = 'models/checkpoints'
) -> Optional[Path]:
    """
    Find the best available checkpoint for a model.
    
    Search priority:
    1. models/{model_name}_best.pth (final trained model)
    2. models/checkpoints/{model_name}_best.pth (best checkpoint)
    3. models/{model_name}.pth (any saved model)
    
    Args:
        model_name: Name of the model (e.g., 'enhanced_cnn', 'baseline_cnn')
        models_dir: Directory for final models
        checkpoints_dir: Directory for checkpoints
    
    Returns:
        Path to checkpoint if found, None otherwise
    
    Example:
        >>> path = find_model_checkpoint('enhanced_cnn')
        >>> if path:
        ...     print(f"Found: {path}")
    """
    search_paths = [
        Path(models_dir) / f'{model_name}_best.pth',
        Path(checkpoints_dir) / f'{model_name}_best.pth',
        Path(models_dir) / f'{model_name}.pth',
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def detect_model_architecture(checkpoint_path: str) -> str:
    """
    Detect model architecture from checkpoint by analyzing layer names.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Model name string (e.g., 'resnet50', 'enhanced_cnn', 'baseline_cnn')
    
    Example:
        >>> arch = detect_model_architecture('models/best_model.pth')
        >>> print(f"Detected: {arch}")
    """
    import torch
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    keys = list(state_dict.keys())
    
    # Check for ResNet
    if any('layer1' in k and 'downsample' in k for k in keys):
        # Count layers to determine ResNet variant
        layer_counts = sum(1 for k in keys if k.startswith('layer'))
        if layer_counts > 200:
            return 'resnet50'
        elif layer_counts > 100:
            return 'resnet34'
        else:
            return 'resnet18'
    
    # Check for EnhancedCNN
    elif any('conv_block' in k for k in keys):
        return 'enhanced_cnn'
    
    # Check for BaselineCNN
    elif any('conv1.0.weight' in k for k in keys) and not any('features' in k for k in keys):
        return 'baseline_cnn'
    
    # Check for EfficientNet
    elif any('_conv_stem' in k or 'blocks' in k for k in keys):
        if 'efficientnet' in str(keys).lower():
            return 'efficientnet_b0'  # Default to b0
        return 'efficientnet_b0'
    
    # Default fallback
    print(f"⚠️  Could not auto-detect architecture. Defaulting to enhanced_cnn")
    return 'enhanced_cnn'


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True
) -> torch.nn.Module:
    """
    Load model weights from checkpoint.
    
    Handles both wrapped (TransferLearningModel) and unwrapped (raw ResNet) models.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        strict: Whether to strictly enforce key matching
    
    Returns:
        Model with loaded weights
    
    Example:
        >>> model = create_model('enhanced_cnn')
        >>> model = load_model_from_checkpoint(model, 'models/enhanced_cnn_best.pth', device)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Full checkpoint with optimizer, etc.
            state_dict = checkpoint['model_state_dict']
            
            # Print additional info if available
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if 'val_accuracy' in checkpoint:
                print(f"   Val Accuracy: {checkpoint['val_accuracy']*100:.2f}%")
            if 'test_accuracy' in checkpoint:
                print(f"   Test Accuracy: {checkpoint['test_accuracy']*100:.2f}%")
        elif 'state_dict' in checkpoint:
            # Another common format
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's a state dict wrapped in a dict
            state_dict = checkpoint
    else:
        # Direct state dict
        state_dict = checkpoint
    
    # Check if this is a raw ResNet/torchvision model being loaded into TransferLearningModel
    from src.models.transfer_learning import TransferLearningModel
    if isinstance(model, TransferLearningModel):
        # Check if state_dict has raw keys (without 'base_model.' prefix)
        has_base_model_prefix = any(k.startswith('base_model.') for k in state_dict.keys())
        has_raw_resnet_keys = any(k.startswith('layer') and not k.startswith('base_model.') for k in state_dict.keys())
        
        if has_raw_resnet_keys and not has_base_model_prefix:
            print("   ⚠️  Detected raw ResNet checkpoint (not wrapped in TransferLearningModel)")
            print("   Converting to wrapped format...")
            
            # Separate base_model and classifier keys
            new_state_dict = {}
            classifier_keys = ['fc.weight', 'fc.bias']
            
            for key, value in state_dict.items():
                if key in classifier_keys:
                    # Map fc.weight -> classifier.4.weight, fc.bias -> classifier.4.bias
                    new_key = key.replace('fc.', 'classifier.4.')
                    new_state_dict[new_key] = value
                else:
                    # Add base_model. prefix
                    new_state_dict[f'base_model.{key}'] = value
            
            state_dict = new_state_dict
            print("   ✅ Converted to wrapped format")
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        if not strict:
            print(f"   ⚠️  Warning: {e}")
            print("   Continuing with strict=False")
        else:
            raise
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    
    return model


def load_or_create_model(
    model_name: str,
    model_params: Dict[str, Any],
    device: torch.device,
    models_dir: str = 'models',
    checkpoints_dir: str = 'models/checkpoints',
    force_new: bool = False
) -> tuple:
    """
    Load model from checkpoint if exists, otherwise create new model.
    
    Args:
        model_name: Name of the model (e.g., 'enhanced_cnn')
        model_params: Parameters for model creation (num_classes, dropout_rate, etc.)
        device: Device to load model on
        models_dir: Directory for final models
        checkpoints_dir: Directory for checkpoints
        force_new: If True, always create new model (ignore cache)
    
    Returns:
        Tuple of (model, checkpoint_path, is_pretrained)
        - model: Model instance
        - checkpoint_path: Path to loaded checkpoint (None if new)
        - is_pretrained: True if loaded from checkpoint
    
    Example:
        >>> model, ckpt_path, is_pretrained = load_or_create_model(
        ...     'enhanced_cnn',
        ...     {'num_classes': 7, 'dropout_rate': 0.5},
        ...     device,
        ...     force_new=False
        ... )
        >>> if is_pretrained:
        ...     print(f"Loaded from {ckpt_path}")
        ... else:
        ...     print("Created new model")
    """
    from src.models import create_model
    
    checkpoint_path = None
    is_pretrained = False
    
    # Try to find existing checkpoint
    if not force_new:
        checkpoint_path = find_model_checkpoint(model_name, models_dir, checkpoints_dir)
    
    # Create model
    model = create_model(model_name, **model_params)
    
    # Load weights if checkpoint exists
    if checkpoint_path and not force_new:
        print(f"\n📦 Found existing checkpoint for '{model_name}'")
        model = load_model_from_checkpoint(model, checkpoint_path, device)
        is_pretrained = True
    else:
        if force_new:
            print(f"\n🆕 Creating new '{model_name}' model (force_new=True)")
        else:
            print(f"\n🆕 Creating new '{model_name}' model (no checkpoint found)")
        model = model.to(device)
        is_pretrained = False
    
    return model, checkpoint_path, is_pretrained


def get_best_model_for_inference(
    models_dir: str = 'models',
    checkpoints_dir: str = 'models/checkpoints',
    preferred_models: list = None
) -> Optional[Path]:
    """
    Find the best available model for inference.
    
    Search priority (unless preferred_models specified):
    1. models/best_model.pth (your 80% model)
    2. models/enhanced_cnn_best.pth
    3. models/baseline_cnn_best.pth
    4. Any *_best.pth in models/
    
    Args:
        models_dir: Directory for final models
        checkpoints_dir: Directory for checkpoints
        preferred_models: List of model names in priority order
    
    Returns:
        Path to best model, or None if no model found
    
    Example:
        >>> best_model = get_best_model_for_inference()
        >>> if best_model:
        ...     print(f"Using: {best_model}")
    """
    models_path = Path(models_dir)
    
    if preferred_models is None:
        preferred_models = [
            'best_model',
            'enhanced_cnn_best',
            'baseline_cnn_best',
        ]
    
    # Check preferred models first
    for model_name in preferred_models:
        path = models_path / f'{model_name}.pth'
        if path.exists():
            print(f"✅ Found best model: {path}")
            return path
    
    # Fall back to any *_best.pth
    best_models = list(models_path.glob('*_best.pth'))
    if best_models:
        path = best_models[0]
        print(f"✅ Found model: {path}")
        return path
    
    # Check checkpoints directory
    checkpoints_path = Path(checkpoints_dir)
    if checkpoints_path.exists():
        best_checkpoints = list(checkpoints_path.glob('*_best.pth'))
        if best_checkpoints:
            path = best_checkpoints[0]
            print(f"✅ Found checkpoint: {path}")
            return path
    
    print("⚠️  No trained model found")
    return None


def list_available_models(
    models_dir: str = 'models',
    checkpoints_dir: str = 'models/checkpoints'
) -> Dict[str, list]:
    """
    List all available trained models.
    
    Args:
        models_dir: Directory for final models
        checkpoints_dir: Directory for checkpoints
    
    Returns:
        Dictionary with 'models' and 'checkpoints' lists
    
    Example:
        >>> available = list_available_models()
        >>> print(f"Final models: {available['models']}")
        >>> print(f"Checkpoints: {available['checkpoints']}")
    """
    models_path = Path(models_dir)
    checkpoints_path = Path(checkpoints_dir)
    
    available = {
        'models': [],
        'checkpoints': []
    }
    
    # List final models
    if models_path.exists():
        available['models'] = [
            str(p) for p in models_path.glob('*.pth')
        ]
    
    # List checkpoints
    if checkpoints_path.exists():
        available['checkpoints'] = [
            str(p) for p in checkpoints_path.glob('*.pth')
        ]
    
    return available