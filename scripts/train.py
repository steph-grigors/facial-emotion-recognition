"""
Complete training script for Facial Emotion Recognition.

Demonstrates training both:
- Baseline CNN (224x224 RGB, standard training)
- Enhanced CNN (128x128 RGB, two-phase training with MixUp/CutMix)
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import our modules
from src.models import create_model, get_model_config
from src.data_loading import create_dataloaders, calculate_class_weights, print_dataloader_info
from src.training import TrainingOrchestrator


def train_baseline_cnn(
    data_dir: str = 'data/processed',
    save_dir: str = 'models/checkpoints',
    device: str = 'cuda'
):
    """
    Train Baseline CNN (65.81% accuracy target).
    
    Architecture: Custom 3-layer CNN
    Input: 224x224 RGB
    Training: Single-phase, standard augmentation
    """
    print("\n" + "="*70)
    print("TRAINING BASELINE CNN")
    print("="*70)
    
    # Get recommended config
    config = get_model_config('baseline')
    print("\nUsing configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print("\n" + "-"*70)
    model = create_model(
        model_type='baseline',
        num_classes=7,
        device=device
    )
    
    # Create dataloaders
    print("\n" + "-"*70)
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir=data_dir,
        model_type='baseline',
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    # Print dataloader info
    print_dataloader_info(train_loader, val_loader, test_loader)
    
    # Calculate class weights (optional)
    class_weights = calculate_class_weights(train_loader)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Optionally use weighted loss:
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        min_lr=1e-6
    )
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=15,
        save_dir=save_dir,
        use_amp=True
    )
    
    # Train (standard single-phase)
    results = orchestrator.train(
        epochs=config['epochs'],
        model_name='baseline_cnn',
        mixup_alpha=0.0,  # No MixUp for baseline
        cutmix_alpha=0.0,  # No CutMix for baseline
        aug_prob=0.0
    )
    
    return results, model


def train_enhanced_cnn(
    data_dir: str = 'data/processed',
    save_dir: str = 'models/checkpoints',
    device: str = 'cuda'
):
    """
    Train Enhanced CNN / ResNet50 (80.21% accuracy target).
    
    Architecture: ResNet50 pretrained on ImageNet
    Input: 128x128 RGB
    Training: Two-phase (frozen → full fine-tuning) with MixUp/CutMix
    """
    print("\n" + "="*70)
    print("TRAINING ENHANCED CNN (ResNet50)")
    print("="*70)
    
    # Get recommended config
    config = get_model_config('enhanced')
    print("\nUsing configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model (with frozen backbone initially)
    print("\n" + "-"*70)
    model = create_model(
        model_type='enhanced',
        num_classes=7,
        pretrained=True,
        freeze_backbone=True,  # Start frozen for Phase 1
        device=device
    )
    
    # Create dataloaders
    print("\n" + "-"*70)
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir=data_dir,
        model_type='enhanced',
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    # Print dataloader info
    print_dataloader_info(train_loader, val_loader, test_loader)
    
    # Calculate class weights (optional)
    class_weights = calculate_class_weights(train_loader)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Optionally use weighted loss:
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Initial optimizer (will be recreated in orchestrator for each phase)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['phase1_lr']
    )
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=None,  # Scheduler created per-phase in train_two_phase
        early_stopping_patience=10,
        save_dir=save_dir,
        use_amp=True
    )
    
    # Train (two-phase)
    results = orchestrator.train_two_phase(
        phase1_epochs=config['phase1_epochs'],
        phase2_epochs=config['phase2_epochs'],
        model_name='enhanced_cnn_resnet50',
        phase1_lr=config['phase1_lr'],
        phase2_lr=config['phase2_lr'],
        phase2_mixup_alpha=config['mixup_alpha'],
        phase2_cutmix_alpha=config['cutmix_alpha'],
        phase2_aug_prob=config['aug_prob']
    )
    
    return results, model


def evaluate_model_on_test(
    model,
    test_loader,
    device,
    model_name: str = "model"
):
    """Evaluate model on test set."""
    from src.training import evaluate_model, compute_metrics, plot_confusion_matrix
    
    print("\n" + "="*70)
    print(f"EVALUATING {model_name} ON TEST SET")
    print("="*70)
    
    results = evaluate_model(model, test_loader, device)
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {results['accuracy']*100:.2f}%")
    
    # Get class names
    class_names = list(test_loader.dataset.class_to_idx.keys())
    
    # Compute detailed metrics
    metrics = compute_metrics(
        results['labels'],
        results['predictions'],
        class_names=class_names
    )
    
    print(f"\n{metrics['report']}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        class_names=class_names,
        save_path=f'results/{model_name}_confusion_matrix.png'
    )
    
    return results


def main():
    """Main training function."""
    # Configuration
    DATA_DIR = 'data/processed'
    SAVE_DIR = 'models/checkpoints'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("FACIAL EMOTION RECOGNITION - TRAINING")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Save directory: {SAVE_DIR}")
    
    # Create directories
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Choose which model to train
    print("\n" + "="*70)
    print("SELECT MODEL TO TRAIN")
    print("="*70)
    print("1. Baseline CNN (224x224, ~65% accuracy)")
    print("2. Enhanced CNN / ResNet50 (128x128, ~80% accuracy)")
    print("3. Both models")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1' or choice == '3':
        # Train Baseline CNN
        baseline_results, baseline_model = train_baseline_cnn(
            data_dir=DATA_DIR,
            save_dir=SAVE_DIR,
            device=DEVICE
        )
        
        # Evaluate on test set
        _, _, test_loader, _ = create_dataloaders(
            data_dir=DATA_DIR,
            model_type='baseline',
            batch_size=64
        )
        evaluate_model_on_test(
            baseline_model,
            test_loader,
            DEVICE,
            model_name="baseline_cnn"
        )
    
    if choice == '2' or choice == '3':
        # Train Enhanced CNN
        enhanced_results, enhanced_model = train_enhanced_cnn(
            data_dir=DATA_DIR,
            save_dir=SAVE_DIR,
            device=DEVICE
        )
        
        # Evaluate on test set
        _, _, test_loader_enh, _ = create_dataloaders(
            data_dir=DATA_DIR,
            model_type='enhanced',
            batch_size=64
        )
        evaluate_model_on_test(
            enhanced_model,
            test_loader_enh,
            DEVICE,
            model_name="enhanced_cnn_resnet50"
        )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
