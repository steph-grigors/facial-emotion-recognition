"""
Inference module for emotion prediction.

Supports both Baseline CNN (224×224 RGB) and Enhanced CNN (128×128 RGB).
All models use ImageNet normalization.

Quick Start:
    >>> from src.inference import create_predictor
    >>> 
    >>> # Create predictor (auto-loads model and sets up transforms)
    >>> predictor = create_predictor(
    ...     'models/checkpoints/enhanced_cnn_resnet50_best.pth',
    ...     model_type='enhanced'
    ... )
    >>> 
    >>> # Predict single image
    >>> prediction, confidence = predictor.predict('image.jpg')
    >>> print(f"{prediction}: {confidence:.1f}%")
    >>> 
    >>> # Predict batch
    >>> results = predictor.predict_batch(['img1.jpg', 'img2.jpg'])
    >>> 
    >>> # Visualize
    >>> from src.inference import visualize_batch_predictions
    >>> visualize_batch_predictions(results)

Advanced Usage:
    >>> from src.inference import EmotionPredictor, predict_emotion
    >>> from src.models import load_model
    >>> from src.data_loading import get_inference_transforms
    >>> 
    >>> # Manual setup with more control
    >>> model = load_model('model.pth', model_type='baseline', device='cuda')
    >>> transform = get_inference_transforms(model_type='baseline')
    >>> class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    >>> 
    >>> predictor = EmotionPredictor(model, 'cuda', class_names, transform)
    >>> prediction, confidence, probs = predictor.predict('image.jpg')
"""

from .predictor import (
    EmotionPredictor,
    predict_emotion,
    predict_batch,
    create_predictor  # Convenient helper function
)

from .visualizer import (
    visualize_prediction,
    visualize_batch_predictions,
    plot_top_predictions,
    plot_confusion_matrix_preview
)

__all__ = [
    # Predictor classes and functions
    'EmotionPredictor',
    'predict_emotion',
    'predict_batch',
    'create_predictor',
    
    # Visualization functions
    'visualize_prediction',
    'visualize_batch_predictions',
    'plot_top_predictions',
    'plot_confusion_matrix_preview',
]

__version__ = '2.0.0'  # Updated for RGB architecture