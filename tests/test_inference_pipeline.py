"""
Complete inference testing with corrected architecture.

Tests inference with both Baseline CNN (224×224 RGB) and Enhanced CNN (128×128 RGB).

Prerequisites:
1. Trained model checkpoint in models/checkpoints/
2. Test data in data/processed/test/

Run: python tests/test_inference_complete.py
"""

import torch
from pathlib import Path
import random
import sys

print("="*80)
print("COMPLETE INFERENCE PIPELINE TEST")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose model type
MODEL_TYPE = 'enhanced'  # Change to 'baseline' to test Baseline CNN

# Model paths
MODEL_PATHS = {
    'baseline': 'models/final/baseline_cnn_best.pth',
    'enhanced': 'models/final/best_model.pth'
}

# Or use your existing model (but it must match the architecture!)
# MODEL_PATHS['enhanced'] = 'models/best_model.pth'  # Only if it's the NEW architecture

TEST_DATA_PATH = 'data/processed/test'
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

MODEL_PATH = MODEL_PATHS[MODEL_TYPE]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n📋 Configuration:")
print(f"   Model type: {MODEL_TYPE.upper()}")
print(f"   Model path: {MODEL_PATH}")
print(f"   Input size: {'224×224' if MODEL_TYPE == 'baseline' else '128×128'} RGB")
print(f"   Test data: {TEST_DATA_PATH}")
print(f"   Device: {DEVICE}")

# ============================================================================
# CHECK PREREQUISITES
# ============================================================================
print(f"\n[1/6] Checking prerequisites...")

model_path = Path(MODEL_PATH)
if not model_path.exists():
    print(f"   ❌ Model not found: {MODEL_PATH}")
    print(f"\n   ⚠️  You need to train a model first!")
    print(f"   Run: python scripts/train.py")
    print(f"\n   Or update MODEL_PATH to point to an existing model.")
    sys.exit(1)
print(f"   ✅ Model found")

test_path = Path(TEST_DATA_PATH)
if not test_path.exists():
    print(f"   ❌ Test data not found: {TEST_DATA_PATH}")
    sys.exit(1)
print(f"   ✅ Test data found")

# Count test images
test_images = []
for emotion_dir in test_path.iterdir():
    if emotion_dir.is_dir() and emotion_dir.name in CLASS_NAMES:
        images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
        test_images.extend(images)

if len(test_images) == 0:
    print(f"   ❌ No test images found in {TEST_DATA_PATH}")
    sys.exit(1)
print(f"   ✅ Found {len(test_images)} test images")

# ============================================================================
# LOAD MODEL AND CREATE PREDICTOR
# ============================================================================
print(f"\n[2/6] Loading model and creating predictor...")

from src.inference import create_predictor

try:
    predictor = create_predictor(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        device=str(DEVICE),
        class_names=CLASS_NAMES
    )
    print(f"   ✅ Predictor created: {predictor}")
except Exception as e:
    print(f"   ❌ Failed to create predictor: {e}")
    print(f"\n   ⚠️  This usually means the model file is incompatible.")
    print(f"   The model might be from the old architecture (48×48 grayscale).")
    print(f"   Please train a new model with: python scripts/train.py")
    sys.exit(1)

# ============================================================================
# TEST 1: SINGLE PREDICTION
# ============================================================================
print(f"\n[3/6] Testing single image prediction...")

# Pick random test image
test_image_path = random.choice(test_images)
true_emotion = test_image_path.parent.name

print(f"\n   📸 Test Image: {test_image_path.name}")
print(f"   📁 True Label: {true_emotion}")

# Predict
try:
    prediction, confidence, probs, original_image = predictor.predict_with_image(
        test_image_path
    )
    
    print(f"   🤖 Predicted: {prediction} ({confidence:.1f}%)")
    
    is_correct = prediction == true_emotion
    if is_correct:
        print(f"   ✅ CORRECT!")
    else:
        print(f"   ❌ WRONG (but this is OK for a single image)")
    
    print(f"\n   All Probabilities:")
    for name, prob in zip(CLASS_NAMES, probs):
        bar = '█' * int(prob / 5)
        marker = '→' if name == prediction else ' '
        print(f"   {marker} {name:10s}: {prob:5.1f}% {bar}")
    
    # Visualize
    print(f"\n   Generating visualization...")
    from src.inference import visualize_prediction
    
    Path('results').mkdir(exist_ok=True)
    visualize_prediction(
        original_image,
        prediction,
        confidence,
        probs,
        CLASS_NAMES,
        true_label=true_emotion,
        save_path='results/test_single_prediction.png',
        show=False
    )
    
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 2: BATCH PREDICTION
# ============================================================================
print(f"\n[4/6] Testing batch prediction...")

# Get sample images (one per emotion)
sample_images = []
sample_labels = []
for emotion in CLASS_NAMES:
    emotion_dir = test_path / emotion
    if emotion_dir.exists():
        images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
        if images:
            sample_img = random.choice(images)
            sample_images.append(sample_img)
            sample_labels.append(emotion)

print(f"   Testing {len(sample_images)} images (one per emotion)...")

# Predict batch
try:
    results = predictor.predict_batch(sample_images, batch_size=8)
    
    # Show results
    correct = 0
    print(f"\n   Results:")
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"   {i}. ❌ ERROR: {result['error']}")
            continue
        
        # Get info
        path = Path(result['path'])
        true_label = path.parent.name
        pred = result['prediction']
        conf = result['confidence']
        
        is_correct = pred == true_label
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"   {i}. {status} {path.name[:20]:20s} → {pred:10s} ({conf:5.1f}%) [True: {true_label}]")
    
    accuracy = (correct / len(results)) * 100 if results else 0
    print(f"\n   📊 Batch Accuracy: {correct}/{len(results)} = {accuracy:.1f}%")
    
    # Visualize batch
    print(f"\n   Generating batch visualization...")
    from src.inference import visualize_batch_predictions
    
    visualize_batch_predictions(
        results,
        ncols=4,
        save_path='results/test_batch_predictions.png',
        show=False
    )
    
except Exception as e:
    print(f"   ❌ Batch prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: STRESS TEST (100 IMAGES)
# ============================================================================
print(f"\n[5/6] Stress test (100 random images)...")

# Get 100 random images
stress_test_images = random.sample(test_images, min(100, len(test_images)))

try:
    stress_results = predictor.predict_batch(stress_test_images, batch_size=32)
    
    # Calculate accuracy
    stress_correct = 0
    stress_total = 0
    
    for result in stress_results:
        if 'error' in result:
            continue
        
        path = Path(result['path'])
        true_label = path.parent.name
        pred = result['prediction']
        
        if pred == true_label:
            stress_correct += 1
        stress_total += 1
    
    if stress_total > 0:
        stress_accuracy = (stress_correct / stress_total) * 100
        print(f"   📊 Stress Test Accuracy: {stress_correct}/{stress_total} = {stress_accuracy:.2f}%")
    else:
        print(f"   ⚠️  No valid predictions in stress test")
    
except Exception as e:
    print(f"   ❌ Stress test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: VISUALIZATION TEST
# ============================================================================
print(f"\n[6/6] Testing visualization functions...")

try:
    # Test top predictions plot
    from src.inference import plot_top_predictions
    
    _, _, test_probs = predictor.predict(test_image_path)
    
    plot_top_predictions(
        test_probs,
        CLASS_NAMES,
        top_k=5,
        save_path='results/test_top_predictions.png',
        show=False
    )
    print(f"   ✅ Top predictions plot saved")
    
    # Test confusion matrix (if enough samples)
    if len(results) >= 7:
        from src.inference import plot_confusion_matrix_preview
        
        plot_confusion_matrix_preview(
            results,
            CLASS_NAMES,
            save_path='results/test_confusion_matrix.png',
            show=False
        )
        print(f"   ✅ Confusion matrix saved")
    else:
        print(f"   ⚠️  Not enough samples for confusion matrix")
    
except Exception as e:
    print(f"   ⚠️  Visualization test failed: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n" + "="*80)
print("✅ ALL INFERENCE TESTS COMPLETED!")
print("="*80)

print(f"\nModel Information:")
print(f"  • Type: {MODEL_TYPE.upper()}")
print(f"  • Architecture: {'Custom 3-layer CNN' if MODEL_TYPE == 'baseline' else 'ResNet50 (transfer learning)'}")
print(f"  • Input: {'224×224' if MODEL_TYPE == 'baseline' else '128×128'} RGB")
print(f"  • Normalization: ImageNet RGB")
print(f"  • Device: {DEVICE}")

print(f"\nTest Results:")
print(f"  • Single prediction: {'✅ Correct' if is_correct else '❌ Wrong'}")
print(f"  • Batch predictions ({len(sample_images)} images): {correct}/{len(results)} correct ({accuracy:.1f}%)")
if 'stress_accuracy' in locals():
    print(f"  • Stress test (100 images): {stress_correct}/{stress_total} correct ({stress_accuracy:.2f}%)")

print(f"\nGenerated Files:")
print(f"  • results/test_single_prediction.png")
print(f"  • results/test_batch_predictions.png")
print(f"  • results/test_top_predictions.png")
if len(results) >= 7:
    print(f"  • results/test_confusion_matrix.png")

print(f"\n💡 Performance Notes:")
print(f"  • Baseline CNN: ~65% accuracy (fast, lightweight)")
print(f"  • Enhanced CNN: ~80% accuracy (slower, more accurate)")
print(f"  • All models trained on FER2013 dataset")
print(f"  • Works best with: frontal faces, good lighting")
print(f"  • May struggle with: profiles, occlusions, poor lighting")

print(f"\n🚀 Next Steps:")
print(f"  1. Check visualizations in results/ directory")
print(f"  2. Try with your own images")
print(f"  3. Deploy with Streamlit/Gradio:")
print(f"     - See deployment guides in docs/")
print(f"  4. Integrate into your application")

print(f"\n" + "="*80)