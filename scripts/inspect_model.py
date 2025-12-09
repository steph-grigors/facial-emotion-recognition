"""
Model checkpoint inspector - detects architecture and shows details.

This tool helps you identify what kind of model is in a .pth file.

Usage:
    python inspect_model.py <path_to_model.pth>

Example:
    python inspect_model.py models/best_model.pth
    python inspect_model.py models/checkpoints/enhanced_cnn_resnet50_best.pth
"""

import torch
import sys
from pathlib import Path


def detect_architecture(state_dict, keys):
    """Detect model architecture from state dict keys."""
    
    # Check for ResNet (transfer learning)
    if any('layer1' in k and 'downsample' in k for k in keys):
        arch_type = "ResNet (Transfer Learning)"
        
        # Detect which ResNet variant
        if 'layer4.2.conv3.weight' in keys:
            variant = "ResNet50 or deeper"
        elif 'layer4.1.conv2.weight' in keys:
            variant = "ResNet34"
        else:
            variant = "ResNet18"
        
        # Check output classes
        if 'fc.weight' in keys:
            num_classes = state_dict['fc.weight'].shape[0]
        else:
            num_classes = "Unknown"
        
        return arch_type, variant, num_classes
    
    # Check for EfficientNet
    elif any('_blocks' in k or 'efficientnet' in k.lower() for k in keys):
        arch_type = "EfficientNet (Transfer Learning)"
        variant = "EfficientNet-B0/B1/B2"
        
        # Check classifier
        classifier_keys = [k for k in keys if 'classifier' in k]
        if classifier_keys:
            num_classes = state_dict[classifier_keys[0]].shape[0] if 'weight' in classifier_keys[0] else "Unknown"
        else:
            num_classes = "Unknown"
        
        return arch_type, variant, num_classes
    
    # Check for VGG
    elif 'classifier.6.weight' in keys or ('classifier.weight' in keys and 'features.0.weight' in keys):
        arch_type = "VGG (Transfer Learning)"
        variant = "VGG16 or VGG19"
        
        if 'classifier.6.weight' in keys:
            num_classes = state_dict['classifier.6.weight'].shape[0]
        else:
            num_classes = "Unknown"
        
        return arch_type, variant, num_classes
    
    # Check for NEW Baseline CNN (224×224 RGB)
    elif all(k in keys for k in ['conv1.0.weight', 'conv2.0.weight', 'conv3.0.weight', 'fc1.weight', 'fc2.weight']):
        # Check if it's the new RGB version
        first_conv_shape = state_dict['conv1.0.weight'].shape
        
        if first_conv_shape[1] == 3:  # RGB (3 channels)
            arch_type = "Baseline CNN (Custom - NEW RGB)"
            variant = "224×224 RGB input, 3-layer CNN"
            num_classes = state_dict['fc2.weight'].shape[0]
        else:  # Old grayscale
            arch_type = "Baseline CNN (Custom - OLD Grayscale)"
            variant = "48×48 grayscale input (OBSOLETE)"
            num_classes = state_dict['fc2.weight'].shape[0]
        
        return arch_type, variant, num_classes
    
    # Check for Enhanced CNN with attention/residual blocks
    elif any('conv_block' in k or 'residual' in k for k in keys):
        arch_type = "Enhanced CNN (Custom)"
        
        features = []
        if any('attention' in k for k in keys):
            features.append("attention")
        if any('residual' in k for k in keys):
            features.append("residual blocks")
        
        variant = "With " + ", ".join(features) if features else "Basic version"
        
        # Find classifier
        classifier_keys = [k for k in keys if k.startswith('fc') or k.startswith('classifier')]
        if classifier_keys:
            for k in classifier_keys:
                if 'weight' in k:
                    num_classes = state_dict[k].shape[0]
                    break
            else:
                num_classes = "Unknown"
        else:
            num_classes = "Unknown"
        
        return arch_type, variant, num_classes
    
    # Unknown architecture
    else:
        arch_type = "Unknown Architecture"
        
        # Try to gather some info
        conv_keys = [k for k in keys if 'conv' in k.lower() and 'weight' in k]
        fc_keys = [k for k in keys if ('fc' in k or 'classifier' in k) and 'weight' in k]
        
        variant = f"{len(conv_keys)} conv layers, {len(fc_keys)} FC layers"
        
        # Try to find output dimension
        if fc_keys:
            num_classes = state_dict[fc_keys[-1]].shape[0]
        else:
            num_classes = "Unknown"
        
        return arch_type, variant, num_classes


def inspect_model(model_path):
    """Inspect model checkpoint and display information."""
    
    print(f"\n{'='*70}")
    print(f"INSPECTING: {model_path}")
    print(f"{'='*70}\n")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                checkpoint_type = "Full checkpoint (with metadata)"
                
                # Show metadata
                print(f"📦 Checkpoint Type: {checkpoint_type}")
                if 'epoch' in checkpoint:
                    print(f"   Epoch: {checkpoint['epoch']}")
                if 'val_accuracy' in checkpoint:
                    print(f"   Val Accuracy: {checkpoint['val_accuracy']*100:.2f}%")
                if 'train_loss' in checkpoint:
                    print(f"   Train Loss: {checkpoint['train_loss']:.4f}")
            else:
                state_dict = checkpoint
                checkpoint_type = "State dict only"
                print(f"📦 Checkpoint Type: {checkpoint_type}")
        else:
            state_dict = checkpoint
            checkpoint_type = "Raw tensor dict"
            print(f"📦 Checkpoint Type: {checkpoint_type}")
        
        # Get keys
        keys = list(state_dict.keys())
        
        print(f"\n🔑 Total Keys: {len(keys)}")
        
        # Show first few keys
        print(f"\nFirst 10 keys:")
        for i, key in enumerate(keys[:10], 1):
            shape = state_dict[key].shape
            print(f"   {i}. {key:40s} → {shape}")
        
        if len(keys) > 10:
            print(f"   ... and {len(keys) - 10} more keys")
        
        # Detect architecture
        print(f"\n🔍 Architecture Detection:")
        arch_type, variant, num_classes = detect_architecture(state_dict, keys)
        
        print(f"   Architecture: {arch_type}")
        print(f"   Variant: {variant}")
        print(f"   Output Classes: {num_classes}")
        
        # Check input shape (from first conv layer)
        first_conv = None
        for key in keys:
            if 'conv' in key.lower() and 'weight' in key and '0' in key:
                first_conv = key
                break
        
        if first_conv:
            shape = state_dict[first_conv].shape
            input_channels = shape[1]
            print(f"   Input Channels: {input_channels} ({'RGB' if input_channels == 3 else 'Grayscale' if input_channels == 1 else 'Unknown'})")
            
            # Suggest input size
            if "ResNet50" in arch_type or "Enhanced CNN" in variant:
                print(f"   Suggested Input: 128×128 RGB")
            elif "Baseline CNN" in arch_type and input_channels == 3:
                print(f"   Suggested Input: 224×224 RGB")
            elif input_channels == 1:
                print(f"   Suggested Input: 48×48 Grayscale (OLD architecture)")
        
        # Parameter count
        total_params = sum(p.numel() for p in state_dict.values())
        trainable_params = total_params  # Assume all trainable
        
        print(f"\n📊 Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        # File size
        import os
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"\n💾 File Size: {size_mb:.2f} MB")
        
        # Recommendations
        print(f"\n{'='*70}")
        print("USAGE RECOMMENDATION:")
        print(f"{'='*70}\n")
        
        if "ResNet" in arch_type:
            print("Use with the updated inference module:")
            print("```python")
            print("from src.inference import create_predictor")
            print("")
            print("predictor = create_predictor(")
            print(f"    '{model_path}',")
            print("    model_type='enhanced'  # For ResNet50")
            print(")")
            print("")
            print("prediction, confidence = predictor.predict('image.jpg')")
            print("```")
        
        elif "Baseline CNN" in arch_type:
            if "RGB" in variant:
                print("Use with the updated inference module:")
                print("```python")
                print("from src.inference import create_predictor")
                print("")
                print("predictor = create_predictor(")
                print(f"    '{model_path}',")
                print("    model_type='baseline'  # For Baseline CNN")
                print(")")
                print("")
                print("prediction, confidence = predictor.predict('image.jpg')")
                print("```")
            else:
                print("⚠️  WARNING: This appears to be an OLD model (48×48 grayscale)")
                print("The current codebase uses RGB architecture.")
                print("Please train a new model with: python scripts/train.py")
        
        else:
            print("Load manually:")
            print("```python")
            print("from src.models import create_model")
            print("")
            print("model = create_model('enhanced', num_classes=7)  # Adjust as needed")
            print(f"model.load_state_dict(torch.load('{model_path}'))")
            print("```")
        
        print()
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_model.pth>")
        print("\nExample:")
        print("  python inspect_model.py models/best_model.pth")
        print("  python inspect_model.py models/checkpoints/enhanced_cnn_resnet50_best.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    inspect_model(model_path)