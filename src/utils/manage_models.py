"""
Model management utility.

List, inspect, and manage trained models.
"""

import argparse
from pathlib import Path
from src.utils import list_available_models, get_best_model_for_inference

def list_models():
    """List all available models."""
    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70)
    
    available = list_available_models()
    
    print("\n📁 Final Models (models/):")
    if available['models']:
        for i, model_path in enumerate(available['models'], 1):
            path = Path(model_path)
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   {i}. {path.name} ({size_mb:.1f} MB)")
    else:
        print("   (none)")
    
    print("\n📁 Checkpoints (models/checkpoints/):")
    if available['checkpoints']:
        for i, model_path in enumerate(available['checkpoints'], 1):
            path = Path(model_path)
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   {i}. {path.name} ({size_mb:.1f} MB)")
    else:
        print("   (none)")
    
    print("\n✨ Best Model for Inference:")
    best = get_best_model_for_inference()
    if best:
        size_mb = best.stat().st_size / (1024 * 1024)
        print(f"   {best} ({size_mb:.1f} MB)")
    else:
        print("   (none found)")
    
    print("\n" + "="*70)


def inspect_model(model_path: str):
    """Inspect a model checkpoint."""
    import torch
    from src.models import create_model
    
    path = Path(model_path)
    
    if not path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("\n" + "="*70)
    print(f"MODEL INSPECTION: {path.name}")
    print("="*70)
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location='cpu')
    
    print("\n📦 Checkpoint Contents:")
    if isinstance(checkpoint, dict):
        print("   Type: Dictionary")
        print(f"   Keys: {list(checkpoint.keys())}")
        
        # Print metadata if available
        if 'epoch' in checkpoint:
            print(f"\n📊 Training Info:")
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'train_loss' in checkpoint:
            print(f"   Train Loss: {checkpoint['train_loss']:.4f}")
        if 'train_accuracy' in checkpoint:
            print(f"   Train Acc: {checkpoint['train_accuracy']*100:.2f}%")
        if 'val_loss' in checkpoint:
            print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
        if 'val_accuracy' in checkpoint:
            print(f"   Val Acc: {checkpoint['val_accuracy']*100:.2f}%")
        if 'test_accuracy' in checkpoint:
            print(f"   Test Acc: {checkpoint['test_accuracy']*100:.2f}%")
        
        # Count parameters
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        print("   Type: State Dict")
        state_dict = checkpoint
    
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n🔢 Parameters:")
    print(f"   Total: {total_params:,}")
    
    # File size
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n💾 File Size:")
    print(f"   {size_mb:.2f} MB")
    
    print("\n" + "="*70)


def set_best_model(model_path: str):
    """Copy a model to best_model.pth for inference."""
    import shutil
    
    source = Path(model_path)
    dest = Path('models/best_model.pth')
    
    if not source.exists():
        print(f"❌ Source model not found: {model_path}")
        return
    
    print(f"\n📋 Setting best model for inference:")
    print(f"   Source: {source}")
    print(f"   Destination: {dest}")
    
    response = input("\nContinue? [y/N]: ")
    if response.lower() == 'y':
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)
        print(f"✅ Best model set: {dest}")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(description='Manage trained models')
    parser.add_argument('command', choices=['list', 'inspect', 'set-best'],
                       help='Command to execute')
    parser.add_argument('--model', type=str, help='Model path (for inspect/set-best)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_models()
    
    elif args.command == 'inspect':
        if not args.model:
            print("❌ Please specify --model <path>")
            return
        inspect_model(args.model)
    
    elif args.command == 'set-best':
        if not args.model:
            print("❌ Please specify --model <path>")
            return
        set_best_model(args.model)


if __name__ == '__main__':
    main()