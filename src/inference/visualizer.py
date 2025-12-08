"""
Visualization utilities for predictions.

Works with both Baseline CNN (224×224 RGB) and Enhanced CNN (128×128 RGB).
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict
from PIL import Image
from pathlib import Path


def visualize_prediction(
    original_image: Image.Image,
    prediction: str,
    confidence: float,
    probabilities: np.ndarray,
    class_names: List[str],
    true_label: Optional[str] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize single prediction with probability distribution.
    
    Args:
        original_image: PIL Image (RGB)
        prediction: Predicted class name
        confidence: Confidence score (0-100)
        probabilities: All class probabilities (0-100)
        class_names: List of emotion labels
        true_label: True label (optional, for showing correctness)
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot (default: True)
    
    Example:
        >>> from src.inference import create_predictor
        >>> predictor = create_predictor('model.pth', model_type='enhanced')
        >>> pred, conf, probs, img = predictor.predict_with_image('test.jpg')
        >>> visualize_prediction(img, pred, conf, probs, predictor.class_names)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Show image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    
    # Title with prediction
    if true_label:
        is_correct = prediction == true_label
        color = 'green' if is_correct else 'red'
        symbol = '✓' if is_correct else '✗'
        title = f'{symbol} True: {true_label}\nPredicted: {prediction}\nConfidence: {confidence:.1f}%'
    else:
        color = 'black'
        title = f'Predicted: {prediction}\nConfidence: {confidence:.1f}%'
    
    axes[0].set_title(title, fontsize=14, fontweight='bold', color=color)
    
    # Show probability distribution
    colors = ['coral' if class_names[i] == prediction else 'skyblue' 
              for i in range(len(class_names))]
    axes[1].barh(class_names, probabilities, color=colors, edgecolor='navy', alpha=0.8)
    axes[1].set_xlabel('Probability (%)', fontsize=12)
    axes[1].set_title('Emotion Probability Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 100)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        axes[1].text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch_predictions(
    results: List[Dict],
    ncols: int = 4,
    figsize: tuple = (16, 8),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize predictions for multiple images in a grid.
    
    Args:
        results: List of prediction results from predict_batch
        ncols: Number of columns in grid
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot (default: True)
    
    Example:
        >>> from src.inference import create_predictor
        >>> predictor = create_predictor('model.pth', model_type='enhanced')
        >>> results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
        >>> visualize_batch_predictions(results, ncols=3)
    """
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    
    if not valid_results:
        print("⚠️  No valid predictions to visualize")
        return
    
    n_images = len(valid_results)
    nrows = (n_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(valid_results):
        if idx >= len(axes):
            break
        
        # Display image
        axes[idx].imshow(result['original_image'])
        axes[idx].axis('off')
        
        # Extract info
        prediction = result['prediction']
        confidence = result['confidence']
        path_str = str(result['path'])
        
        # Try to extract true label from path
        true_label = _extract_true_label_from_path(path_str)
        
        # Title with prediction
        if true_label:
            is_correct = prediction == true_label
            color = 'green' if is_correct else 'red'
            symbol = '✓' if is_correct else '✗'
            
            # Get filename for display
            filename = Path(path_str).name if path_str != 'PIL Image' else 'PIL'
            title = f'{symbol} {filename}\nTrue: {true_label}\nPred: {prediction} ({confidence:.1f}%)'
        else:
            color = 'black'
            filename = Path(path_str).name if path_str != 'PIL Image' else 'PIL'
            title = f'{filename}\nPred: {prediction}\n({confidence:.1f}%)'
        
        axes[idx].set_title(title, fontsize=9, color=color, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    # Add summary info
    if valid_results and 'probabilities' in valid_results[0]:
        total = len(results)
        valid = len(valid_results)
        errors = len(error_results)
        
        # Count correct predictions if true labels available
        correct = sum(1 for r in valid_results 
                     if _extract_true_label_from_path(str(r['path'])) == r['prediction'])
        
        if correct > 0:
            accuracy = (correct / valid) * 100
            title_text = f'Emotion Predictions - {valid}/{total} valid | Accuracy: {correct}/{valid} ({accuracy:.1f}%)'
        else:
            title_text = f'Emotion Predictions - {valid}/{total} images'
        
        if errors > 0:
            title_text += f' | {errors} errors'
    else:
        title_text = 'Emotion Predictions'
    
    plt.suptitle(title_text, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved batch visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print errors if any
    if error_results:
        print(f"\n⚠️  {len(error_results)} images had errors:")
        for err in error_results[:5]:  # Show first 5
            print(f"   - {err['path']}: {err['error']}")
        if len(error_results) > 5:
            print(f"   ... and {len(error_results) - 5} more")


def plot_top_predictions(
    probabilities: np.ndarray,
    class_names: List[str],
    top_k: int = 3,
    figsize: tuple = (8, 4),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot top-k predictions with color-coded confidence.
    
    Args:
        probabilities: Class probabilities (0-100)
        class_names: List of emotion labels
        top_k: Number of top predictions to show (default: 3)
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot (default: True)
    
    Example:
        >>> from src.inference import create_predictor
        >>> predictor = create_predictor('model.pth', model_type='enhanced')
        >>> _, _, probs = predictor.predict('test.jpg')
        >>> plot_top_predictions(probs, predictor.class_names, top_k=5)
    """
    # Get top-k indices
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_probs = probabilities[top_indices]
    top_names = [class_names[i] for i in top_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color based on confidence (green = high, yellow = medium, red = low)
    colors = plt.cm.RdYlGn(top_probs / 100)
    bars = ax.barh(range(len(top_names)), top_probs, color=colors, edgecolor='navy', alpha=0.8)
    
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=11)
    ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Emotion Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved top predictions plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix_preview(
    results: List[Dict],
    class_names: List[str],
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot confusion matrix from batch prediction results.
    
    Only works if true labels can be extracted from paths.
    
    Args:
        results: List of prediction results from predict_batch
        class_names: List of emotion labels
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot (default: True)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Extract predictions and true labels
    y_true = []
    y_pred = []
    
    for result in results:
        if 'error' in result:
            continue
        
        true_label = _extract_true_label_from_path(str(result['path']))
        if true_label and true_label in class_names:
            y_true.append(class_names.index(true_label))
            y_pred.append(class_names.index(result['prediction']))
    
    if len(y_true) < 2:
        print("⚠️  Not enough labeled predictions for confusion matrix")
        return
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}, ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix ({len(y_true)} samples)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_true_label_from_path(path_str: str) -> Optional[str]:
    """
    Extract emotion label from file path.
    
    Assumes structure like: .../emotion/image.jpg
    """
    if path_str == 'PIL Image':
        return None
    
    try:
        import os
        parent_dir = os.path.basename(os.path.dirname(path_str))
        
        # Check if parent directory is an emotion label
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        if parent_dir.lower() in emotions:
            return parent_dir.lower()
    except:
        pass
    
    return None


if __name__ == "__main__":
    """Test visualization module."""
    print("="*70)
    print("VISUALIZATION MODULE TEST")
    print("="*70)
    
    # Create dummy data
    dummy_img = Image.new('RGB', (224, 224), color=(150, 150, 150))
    prediction = 'happy'
    confidence = 85.3
    probabilities = np.array([5.2, 2.1, 3.8, 85.3, 1.5, 1.2, 0.9])
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Test single prediction visualization
    print("\n1. Testing single prediction visualization...")
    visualize_prediction(
        dummy_img, prediction, confidence, probabilities, class_names,
        true_label='happy', show=False
    )
    print("   ✅ Single prediction visualization works")
    
    # Test top predictions
    print("\n2. Testing top predictions plot...")
    plot_top_predictions(probabilities, class_names, top_k=3, show=False)
    print("   ✅ Top predictions plot works")
    
    # Test batch visualization
    print("\n3. Testing batch visualization...")
    dummy_results = [
        {
            'path': 'test/happy/img1.jpg',
            'prediction': 'happy',
            'confidence': 85.3,
            'probabilities': probabilities,
            'original_image': dummy_img
        },
        {
            'path': 'test/sad/img2.jpg',
            'prediction': 'sad',
            'confidence': 72.1,
            'probabilities': np.array([8.2, 3.1, 5.8, 10.3, 2.5, 72.1, 1.0]),
            'original_image': dummy_img
        },
    ]
    visualize_batch_predictions(dummy_results, ncols=2, show=False)
    print("   ✅ Batch visualization works")
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATION TESTS PASSED!")
    print("="*70)
    print("\nKey Features:")
    print("  ✓ Single prediction with probability distribution")
    print("  ✓ Batch predictions with accuracy tracking")
    print("  ✓ Top-k predictions with color coding")
    print("  ✓ Confusion matrix from batch results")
    print("  ✓ Automatic true label extraction from paths")
    print("  ✓ Error handling and reporting")