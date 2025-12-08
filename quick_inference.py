"""
Quick inference example using updated modules.

This is the simplest way to use the emotion recognition model.

Prerequisites:
- Trained model in models/checkpoints/
- Image to predict (or use default test image)

Usage:
    python quick_inference.py                    # Use default test image
    python quick_inference.py path/to/image.jpg  # Use your image
"""

from src.inference import create_predictor
from pathlib import Path
import sys

print("="*70)
print("QUICK INFERENCE EXAMPLE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model path - update this to your trained model
MODEL_PATH = 'models/final/best_model.pth' 
MODEL_TYPE = 'enhanced'  # or 'baseline'

# Alternative: Use baseline model
# MODEL_PATH = 'models/checkpoints/baseline_cnn_best.pth'
# MODEL_TYPE = 'baseline'

# ============================================================================
# SETUP
# ============================================================================

print(f"\n📋 Setup:")
print(f"   Model: {MODEL_PATH}")
print(f"   Type: {MODEL_TYPE.upper()}")

# Check if model exists
if not Path(MODEL_PATH).exists():
    print(f"\n❌ Model not found: {MODEL_PATH}")
    print(f"\n💡 Please train a model first:")
    print(f"   python scripts/train.py")
    print(f"\n   Or update MODEL_PATH in this script to point to your model.")
    sys.exit(1)

# Create predictor (one line!)
print(f"\n🔄 Loading model...")
try:
    predictor = create_predictor(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE
    )
except Exception as e:
    print(f"\n❌ Failed to load model: {e}")
    print(f"\n💡 The model might be from the old architecture.")
    print(f"   Train a new model with: python scripts/train.py")
    sys.exit(1)

# ============================================================================
# GET IMAGE
# ============================================================================

# Get image path from command line or use default
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Default test image
    image_path = 'data/processed/test/happy/happy_12785.jpg'
    
    # Try to find any test image if default doesn't exist
    if not Path(image_path).exists():
        test_dir = Path('data/processed/test')
        if test_dir.exists():
            # Find first available image
            for emotion in predictor.class_names:
                emotion_dir = test_dir / emotion
                if emotion_dir.exists():
                    images = list(emotion_dir.glob('*.jpg'))
                    if images:
                        image_path = str(images[0])
                        break

# Check if image exists
if not Path(image_path).exists():
    print(f"\n❌ Image not found: {image_path}")
    print(f"\n💡 Usage:")
    print(f"   python quick_inference.py <path_to_image>")
    sys.exit(1)

print(f"   Image: {image_path}")

# ============================================================================
# PREDICT
# ============================================================================

print(f"\n🔍 Predicting...")

try:
    prediction, confidence, probs = predictor.predict(image_path)
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}\n")

print(f"🎯 Prediction: {prediction.upper()}")
print(f"   Confidence: {confidence:.1f}%")

print(f"\n📊 All Probabilities:")
for name, prob in zip(predictor.class_names, probs):
    # Create simple text bar
    bar_length = int(prob / 2)  # Scale to 0-50 chars
    bar = '█' * bar_length
    
    # Highlight prediction
    marker = '→' if name == prediction else ' '
    
    print(f"   {marker} {name:10s}: {prob:5.1f}% {bar}")

# ============================================================================
# VISUALIZATION (OPTIONAL)
# ============================================================================

try:
    from src.inference import visualize_prediction
    
    # Get image with prediction
    _, _, _, original_image = predictor.predict_with_image(image_path)
    
    # Try to get true label from path (if image is from test set)
    true_label = None
    if '/test/' in str(image_path):
        true_label = Path(image_path).parent.name
    
    # Save visualization
    output_path = 'results/quick_inference_result.png'
    Path('results').mkdir(exist_ok=True)
    
    visualize_prediction(
        original_image,
        prediction,
        confidence,
        probs,
        predictor.class_names,
        true_label=true_label,
        save_path=output_path,
        show=False
    )
    
    print(f"\n📊 Visualization saved: {output_path}")

except Exception as e:
    print(f"\n⚠️  Visualization failed (non-critical): {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"✅ Prediction complete!")
print(f"{'='*70}\n")

print(f"💡 Try with your own image:")
print(f"   python quick_inference.py path/to/your/image.jpg")

print()
