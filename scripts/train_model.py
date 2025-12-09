"""
Training script using existing modules with model caching support.

Usage:
    # Train with auto-load (checks for existing models)
    python train.py --model enhanced_cnn --epochs 50
    
    # Force retrain from scratch
    python train.py --model enhanced_cnn --epochs 50 --force-new
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils import load_or_create_model, get_device, set_seed, setup_paths, load_config
from src.training import TrainingOrchestrator


def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    parser.add_argument('--model', type=str, default='enhanced_cnn',
                        choices=['baseline_cnn', 'enhanced_cnn', 'resnet18', 'resnet34', 
                                'resnet50', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--force-new', action='store_true', 
                        help='Force train new model (ignore cached checkpoint)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    config = load_config(args.config)
    paths = setup_paths(config)
    
    print("\n" + "="*70)
    print("FACIAL EMOTION RECOGNITION - TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Force new: {args.force_new}")
    print(f"  Device: {device}")
    
    # Model parameters
    model_params = {
        'num_classes': config['data']['num_classes'],
        'dropout_rate': 0.5
    }
    if args.model == 'enhanced_cnn':
        model_params['use_attention'] = True
    
    # Load or create model (uses existing TrainingOrchestrator's load_checkpoint if found)
    model, checkpoint_path, is_pretrained = load_or_create_model(
        model_name=args.model,
        model_params=model_params,
        device=device,
        models_dir=str(paths['models_dir']),
        checkpoints_dir=str(paths['checkpoints_dir']),
        force_new=args.force_new
    )
    
    if is_pretrained and not args.force_new:
        print(f"\n⚠️  Model loaded from checkpoint: {checkpoint_path}")
        print(f"   To train from scratch, use --force-new flag")
        
        response = input("\nContinue training from this checkpoint? [y/N]: ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(tuple(config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['data']['normalize_mean'],
            std=config['data']['normalize_std']
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(tuple(config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['data']['normalize_mean'],
            std=config['data']['normalize_std']
        )
    ])
    
    # Load data
    print(f"\n📊 Loading data from {paths['data_dir']}...")
    train_dataset = datasets.ImageFolder(root=paths['data_dir'] / 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=paths['data_dir'] / 'val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
    )
    
    # Training orchestrator (use existing module!)
    print(f"\n🚀 Initializing training...")
    orchestrator = TrainingOrchestrator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=10,
        save_dir=str(paths['checkpoints_dir'])
    )
    
    # If resuming from checkpoint, load optimizer/scheduler state
    if is_pretrained and checkpoint_path:
        try:
            # Try to load full checkpoint (not just weights)
            full_checkpoint_path = paths['checkpoints_dir'] / f"{args.model}_checkpoint.pth"
            if full_checkpoint_path.exists():
                print(f"Loading optimizer/scheduler state...")
                orchestrator.load_checkpoint(str(full_checkpoint_path))
        except Exception as e:
            print(f"Could not load full checkpoint state: {e}")
            print("Starting with fresh optimizer/scheduler")
    
    # Train (uses existing TrainingOrchestrator.train method)
    results = orchestrator.train(epochs=args.epochs, model_name=args.model)
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {results['best_val_acc']*100:.2f}%")
    print(f"   Model saved to: {paths['checkpoints_dir']}/{args.model}_best.pth")
    
    # Optionally copy to final models directory
    import shutil
    final_model_path = paths['models_dir'] / f"{args.model}_best.pth"
    checkpoint_file = paths['checkpoints_dir'] / f"{args.model}_best.pth"
    
    if checkpoint_file.exists():
        shutil.copy(checkpoint_file, final_model_path)
        print(f"   Also copied to: {final_model_path}")
        print(f"\n💡 To use this model for inference, run:")
        print(f"   python manage_models.py set-best --model {final_model_path}")


if __name__ == '__main__':
    main()