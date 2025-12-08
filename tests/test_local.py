"""
Local testing script - Test all modules before deployment.

This script tests:
1. Models module (Baseline: 224x224 RGB, Enhanced: 128x128 RGB)
2. Training module with correct data shapes
3. Inference module (NOT IMPLEMENTED - requires inference module updates)
4. Utils module (optional)

Run this to verify everything works!
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

print("="*70)
print("FACIAL EMOTION RECOGNITION - LOCAL TEST")
print("="*70)

# Test 1: Models Module
print("\n[1/4] Testing Models Module...")
try:
    from src.models import create_model
    
    # Test Baseline CNN (224x224 RGB)
    print("   Testing Baseline CNN (224x224 RGB)...")
    model_baseline = create_model('baseline', num_classes=7, dropout_rate=0.5)
    
    info = model_baseline.get_model_info()
    print(f"   ✅ Created {info['model_name']} with {info['total_parameters']:,} parameters")
    
    # Test forward pass - BASELINE USES 224x224 RGB
    dummy_input_baseline = torch.randn(2, 3, 224, 224)  # RGB!
    output_baseline = model_baseline(dummy_input_baseline)
    assert output_baseline.shape == (2, 7), f"Wrong output shape: {output_baseline.shape}"
    print(f"   ✅ Baseline forward pass: {dummy_input_baseline.shape} → {output_baseline.shape}")
    
    # Test Enhanced CNN (128x128 RGB - ResNet50)
    print("   Testing Enhanced CNN (128x128 RGB - ResNet50)...")
    model_enhanced = create_model(
        'enhanced', 
        num_classes=7, 
        pretrained=False,  # Don't download weights for testing
        freeze_backbone=False
    )
    
    info2 = model_enhanced.get_model_info()
    print(f"   ✅ Created {info2['model_name']} with {info2['total_parameters']:,} parameters")
    
    # Test forward pass - ENHANCED USES 128x128 RGB
    dummy_input_enhanced = torch.randn(2, 3, 128, 128)  # RGB!
    output_enhanced = model_enhanced(dummy_input_enhanced)
    assert output_enhanced.shape == (2, 7), f"Wrong output shape: {output_enhanced.shape}"
    print(f"   ✅ Enhanced forward pass: {dummy_input_enhanced.shape} → {output_enhanced.shape}")
    
    print("   ✅ Models module working!")
    
except Exception as e:
    print(f"   ❌ Models module failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test 2: Training Module
print("\n[2/4] Testing Training Module...")
try:
    from src.training import train_epoch, validate, EarlyStopping
    from src.training import compute_metrics, plot_confusion_matrix
    from torch.utils.data import TensorDataset, DataLoader
    
    print("   Testing with Baseline CNN (224x224 RGB)...")
    
    # Create dummy data - BASELINE: 224x224 RGB
    dummy_images = torch.randn(100, 3, 224, 224)  # RGB!
    dummy_labels = torch.randint(0, 7, (100,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Test training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('baseline', num_classes=7)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test one epoch (simple version without augmentation)
    from src.training import train_epoch_simple
    train_loss, train_acc = train_epoch_simple(model, loader, criterion, optimizer, device)
    print(f"   ✅ Training epoch: loss={train_loss:.4f}, acc={train_acc*100:.2f}%")
    
    val_loss, val_acc = validate(model, loader, criterion, device)
    print(f"   ✅ Validation: loss={val_loss:.4f}, acc={val_acc*100:.2f}%")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=5)
    early_stopping(val_loss)
    print(f"   ✅ Early stopping initialized")
    
    # Test metrics
    y_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 1, 1, 2]
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    metrics = compute_metrics(y_true, y_pred, class_names)
    print(f"   ✅ Metrics computed: accuracy={metrics['accuracy']*100:.1f}%")
    
    print("   ✅ Training module working!")
    
except Exception as e:
    print(f"   ❌ Training module failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test 3: Data Loading Module
print("\n[3/4] Testing Data Loading Module...")
try:
    from src.data_loading import get_train_transforms, get_val_transforms
    from src.data_loading import IMAGENET_MEAN, IMAGENET_STD
    
    # Test Baseline transforms (224x224)
    print("   Testing Baseline transforms (224x224 RGB)...")
    baseline_train_tf = get_train_transforms(model_type='baseline')
    baseline_val_tf = get_val_transforms(model_type='baseline')
    print(f"   ✅ Baseline transforms created")
    
    # Test Enhanced transforms (128x128)
    print("   Testing Enhanced transforms (128x128 RGB)...")
    enhanced_train_tf = get_train_transforms(model_type='enhanced')
    enhanced_val_tf = get_val_transforms(model_type='enhanced')
    print(f"   ✅ Enhanced transforms created")
    
    # Verify normalization values
    print(f"   ✅ ImageNet normalization: mean={IMAGENET_MEAN}, std={IMAGENET_STD}")
    
    # Test augmentations
    from src.data_loading import mixup_data, cutmix_data
    x = torch.randn(4, 3, 128, 128)
    y = torch.tensor([0, 1, 2, 3])
    
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    print(f"   ✅ MixUp augmentation working (lambda={lam:.3f})")
    
    cut_x, y_a, y_b, lam = cutmix_data(x, y, alpha=0.4)
    print(f"   ✅ CutMix augmentation working (lambda={lam:.3f})")
    
    print("   ✅ Data loading module working!")
    
except Exception as e:
    print(f"   ❌ Data loading module failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test 4: Utils Module (Optional)
print("\n[4/4] Testing Utils Module (Optional)...")
try:
    # Try to import utils functions
    try:
        from src.utils import get_device, set_seed
        
        device = get_device()
        print(f"   ✅ Device detection: {device}")
        
        set_seed(42)
        print(f"   ✅ Seed setting working")
        
        print("   ✅ Utils module working!")
    except ImportError:
        print("   ⚠️  Utils module not found (optional, skipping)")
    
except Exception as e:
    print(f"   ⚠️  Utils module failed (optional): {e}")


# Summary
print("\n" + "="*70)
print("✅ ALL CORE TESTS PASSED!")
print("="*70)

print("\n📋 Test Summary:")
print("   ✅ Models: Baseline (224x224 RGB) & Enhanced (128x128 RGB, ResNet50)")
print("   ✅ Training: train_epoch, validate, early stopping, metrics")
print("   ✅ Data Loading: transforms (RGB + ImageNet norm), MixUp, CutMix")
print("   ✅ Utils: device detection, seed setting (optional)")

print("\n🎯 Key Specifications:")
print("   • Baseline CNN:")
print("     - Input: 224×224 RGB (3 channels)")
print("     - Architecture: Custom 3-layer CNN")
print("     - Parameters: ~3.3M")
print("     - Target accuracy: ~65.81%")
print("")
print("   • Enhanced CNN (ResNet50):")
print("     - Input: 128×128 RGB (3 channels)")
print("     - Architecture: ResNet50 pretrained on ImageNet")
print("     - Parameters: ~23.5M")
print("     - Target accuracy: ~80.21%")
print("     - Training: Two-phase (frozen → unfrozen)")

print("\n⚠️  Important Notes:")
print("   • NO grayscale conversion - all models use RGB")
print("   • ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
print("   • Different image sizes for different models (224 vs 128)")
print("   • Enhanced CNN requires two-phase training for best results")

print("\n🚀 Next Steps:")
print("   1. Train models: python scripts/train.py")
print("   2. Test with real data: python tests/test_data_loaders.py")
print("   3. Run inference (requires inference module updates)")
print("   4. Deploy to production")

print("\n" + "="*70)
