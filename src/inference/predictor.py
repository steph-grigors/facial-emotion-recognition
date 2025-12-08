"""
Emotion prediction functions with support for both Baseline and Enhanced models.

This module works with:
- Baseline CNN: 224×224 RGB
- Enhanced CNN (ResNet50): 128×128 RGB

Both use ImageNet normalization.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
from torchvision import transforms


def predict_emotion(
    image: Union[str, Path, Image.Image],
    model: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    transform: transforms.Compose
) -> Tuple[str, float, np.ndarray, Image.Image]:
    """
    Predict emotion from a single image.
    
    Args:
        image: Path to image file or PIL Image
        model: Trained model
        device: CPU or GPU device
        class_names: List of emotion labels
        transform: Image transforms (use get_inference_transform() helper)
    
    Returns:
        Tuple of (prediction, confidence, probabilities, original_image)
        - prediction: Predicted class name
        - confidence: Confidence score (0-100)
        - probabilities: All class probabilities (0-100)
        - original_image: Original PIL image (RGB)
    
    Example:
        >>> from src.data_loading import get_inference_transforms
        >>> transform = get_inference_transforms(model_type='enhanced')
        >>> prediction, confidence, probs, img = predict_emotion(
        ...     'test.jpg', model, device, class_names, transform
        ... )
        >>> print(f"Predicted: {prediction} ({confidence:.1f}%)")
    """
    # Load image and ensure RGB
    if isinstance(image, (str, Path)):
        original_image = Image.open(image).convert('RGB')
    else:
        # Ensure RGB even if passed as PIL Image
        original_image = image.convert('RGB') if image.mode != 'RGB' else image
    
    # Transform and predict
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    prediction = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    probs = probabilities.cpu().numpy()[0] * 100
    
    return prediction, confidence_score, probs, original_image


def predict_batch(
    images: List[Union[str, Path, Image.Image]],
    model: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    transform: transforms.Compose,
    batch_size: int = 32
) -> List[Dict]:
    """
    Predict emotions for multiple images with batched inference.
    
    Args:
        images: List of image paths or PIL Images
        model: Trained model
        device: CPU or GPU device
        class_names: List of emotion labels
        transform: Image transforms (use get_inference_transform() helper)
        batch_size: Batch size for inference (default: 32)
    
    Returns:
        List of dictionaries with prediction results
    
    Example:
        >>> from src.data_loading import get_inference_transforms
        >>> transform = get_inference_transforms(model_type='baseline')
        >>> results = predict_batch(
        ...     ['img1.jpg', 'img2.jpg'], model, device, class_names, transform
        ... )
        >>> for result in results:
        ...     print(f"{result['path']}: {result['prediction']}")
    """
    results = []
    
    # Process in batches for efficiency
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        
        batch_tensors = []
        batch_originals = []
        batch_paths = []
        
        for image in batch_images:
            try:
                # Load and convert to RGB
                if isinstance(image, (str, Path)):
                    original_image = Image.open(image).convert('RGB')
                    image_path = str(image)
                else:
                    original_image = image.convert('RGB') if image.mode != 'RGB' else image
                    image_path = 'PIL Image'
                
                # Transform
                image_tensor = transform(original_image)
                
                batch_tensors.append(image_tensor)
                batch_originals.append(original_image)
                batch_paths.append(image_path)
                
            except Exception as e:
                results.append({
                    'path': str(image) if isinstance(image, (str, Path)) else 'PIL Image',
                    'error': str(e)
                })
        
        # Skip if no valid images in batch
        if not batch_tensors:
            continue
        
        # Batch predict
        try:
            batch_tensor = torch.stack(batch_tensors).to(device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probabilities, 1)
            
            # Extract results
            for idx, (pred_idx, conf, original_img, path) in enumerate(
                zip(predicted_indices, confidences, batch_originals, batch_paths)
            ):
                prediction = class_names[pred_idx.item()]
                confidence_score = conf.item() * 100
                probs = probabilities[idx].cpu().numpy() * 100
                
                results.append({
                    'path': path,
                    'prediction': prediction,
                    'confidence': confidence_score,
                    'probabilities': probs,
                    'original_image': original_img
                })
                
        except Exception as e:
            # If batch fails, fall back to individual processing
            for original_img, path in zip(batch_originals, batch_paths):
                try:
                    pred, conf, probs, _ = predict_emotion(
                        original_img, model, device, class_names, transform
                    )
                    results.append({
                        'path': path,
                        'prediction': pred,
                        'confidence': conf,
                        'probabilities': probs,
                        'original_image': original_img
                    })
                except Exception as e2:
                    results.append({
                        'path': path,
                        'error': str(e2)
                    })
    
    return results


class EmotionPredictor:
    """
    High-level emotion predictor class with support for both model types.
    
    Args:
        model: Trained model (Baseline or Enhanced)
        device: CPU or GPU device
        class_names: List of emotion labels
        transform: Image transforms (use get_inference_transform() helper)
        model_type: 'baseline' or 'enhanced' (for informational purposes)
    
    Example:
        >>> from src.models import load_model
        >>> from src.data_loading import get_inference_transforms
        >>> 
        >>> # For Enhanced CNN (ResNet50)
        >>> model = load_model('models/checkpoints/enhanced_cnn_resnet50_best.pth', 
        ...                    model_type='enhanced', device='cuda')
        >>> transform = get_inference_transforms(model_type='enhanced')
        >>> predictor = EmotionPredictor(model, 'cuda', class_names, transform, 'enhanced')
        >>> 
        >>> # Predict
        >>> prediction, confidence = predictor.predict('test.jpg')
        >>> print(f"Emotion: {prediction} ({confidence:.1f}%)")
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Union[str, torch.device],
        class_names: List[str],
        transform: transforms.Compose,
        model_type: Optional[str] = None
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.class_names = class_names
        self.transform = transform
        self.model_type = model_type
        self.model.eval()
    
    def predict(
        self, 
        image: Union[str, Path, Image.Image]
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from image.
        
        Args:
            image: Path to image or PIL Image (will be converted to RGB)
        
        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        prediction, confidence, probs, _ = predict_emotion(
            image, self.model, self.device, self.class_names, self.transform
        )
        return prediction, confidence, probs
    
    def predict_with_image(
        self, 
        image: Union[str, Path, Image.Image]
    ) -> Tuple[str, float, np.ndarray, Image.Image]:
        """
        Predict emotion and return original image.
        
        Args:
            image: Path to image or PIL Image (will be converted to RGB)
        
        Returns:
            Tuple of (prediction, confidence, probabilities, original_image)
        """
        return predict_emotion(
            image, self.model, self.device, self.class_names, self.transform
        )
    
    def predict_batch(
        self, 
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict emotions for multiple images.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for inference
        
        Returns:
            List of dictionaries with prediction results
        """
        return predict_batch(
            images, self.model, self.device, self.class_names, 
            self.transform, batch_size
        )
    
    def __repr__(self):
        model_info = f"model_type='{self.model_type}'" if self.model_type else "model_type=Unknown"
        return f"EmotionPredictor({model_info}, device='{self.device}', classes={len(self.class_names)})"


# ============================================================================
# HELPER FUNCTIONS FOR EASY SETUP
# ============================================================================

def create_predictor(
    model_path: str,
    model_type: str = 'enhanced',
    device: Optional[str] = None,
    class_names: Optional[List[str]] = None
) -> EmotionPredictor:
    """
    Convenient function to create a predictor from a model checkpoint.
    
    This automatically loads the model and sets up the correct transforms.
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        model_type: 'baseline' or 'enhanced' (default: 'enhanced')
        device: 'cuda' or 'cpu' (default: auto-detect)
        class_names: List of emotion labels (default: FER2013 emotions)
    
    Returns:
        EmotionPredictor ready for inference
    
    Example:
        >>> # Quick setup for Enhanced CNN
        >>> predictor = create_predictor(
        ...     'models/checkpoints/enhanced_cnn_resnet50_best.pth',
        ...     model_type='enhanced'
        ... )
        >>> prediction, confidence = predictor.predict('image.jpg')
        >>> 
        >>> # Quick setup for Baseline CNN
        >>> predictor = create_predictor(
        ...     'models/checkpoints/baseline_cnn_best.pth',
        ...     model_type='baseline'
        ... )
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Default class names
    if class_names is None:
        class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Load model
    from src.models import load_model
    model = load_model(
        checkpoint_path=model_path,
        model_type=model_type,
        num_classes=len(class_names),
        device=device
    )
    
    # Get appropriate transform
    from src.data_loading import get_inference_transforms
    transform = get_inference_transforms(model_type=model_type)
    
    # Create predictor
    predictor = EmotionPredictor(
        model=model,
        device=device,
        class_names=class_names,
        transform=transform,
        model_type=model_type
    )
    
    print(f"✅ Predictor created successfully!")
    print(f"   Model: {model_type.upper()}")
    print(f"   Input: {'224×224' if model_type == 'baseline' else '128×128'} RGB")
    print(f"   Device: {device}")
    print(f"   Classes: {class_names}")
    
    return predictor


if __name__ == "__main__":
    """Test inference module."""
    print("="*70)
    print("INFERENCE MODULE TEST")
    print("="*70)
    
    # Test with dummy data
    from src.models import create_model
    from src.data_loading import get_inference_transforms
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Test Baseline
    print("\n1. Testing Baseline CNN (224×224 RGB)...")
    model_baseline = create_model('baseline', num_classes=7, device=device)
    transform_baseline = get_inference_transforms(model_type='baseline')
    
    # Create dummy RGB image
    dummy_img = Image.new('RGB', (300, 300), color=(128, 128, 128))
    
    pred, conf, probs, img = predict_emotion(
        dummy_img, model_baseline, device, class_names, transform_baseline
    )
    print(f"   ✅ Prediction: {pred} ({conf:.1f}%)")
    assert len(probs) == 7, "Wrong number of probabilities"
    print(f"   ✅ All probabilities: {probs.sum():.1f}% (should be ~100)")
    
    # Test Enhanced
    print("\n2. Testing Enhanced CNN (128×128 RGB)...")
    model_enhanced = create_model('enhanced', num_classes=7, pretrained=False, device=device)
    transform_enhanced = get_inference_transforms(model_type='enhanced')
    
    pred, conf, probs, img = predict_emotion(
        dummy_img, model_enhanced, device, class_names, transform_enhanced
    )
    print(f"   ✅ Prediction: {pred} ({conf:.1f}%)")
    
    # Test predictor class
    print("\n3. Testing EmotionPredictor class...")
    predictor = EmotionPredictor(
        model_enhanced, device, class_names, transform_enhanced, 'enhanced'
    )
    pred2, conf2, probs2 = predictor.predict(dummy_img)
    print(f"   ✅ EmotionPredictor: {pred2} ({conf2:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ ALL INFERENCE TESTS PASSED!")
    print("="*70)
    print("\nKey Features:")
    print("  ✓ Works with both Baseline (224×224) and Enhanced (128×128)")
    print("  ✓ All inputs automatically converted to RGB")
    print("  ✓ Uses ImageNet normalization")
    print("  ✓ Batched inference support")
    print("  ✓ Helper function: create_predictor() for easy setup")