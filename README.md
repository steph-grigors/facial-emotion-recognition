# 🎭 Facial Emotion Recognition

A deep learning application for real-time facial emotion recognition using PyTorch and Streamlit. Recognizes 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.


## 🎯 Overview

Deep learning-powered facial emotion recognition with an interactive web interface. Upload images or use your webcam to detect emotions in real-time.

**Live Demo:** [https://cnn-facial-emotion-recognition.streamlit.app/]

## ✨ Features

- **Real-time Detection**: Webcam support for live emotion recognition
- **Multiple Input Methods**: Upload images or use camera
- **Confidence Scores**: View prediction probabilities for all emotions
- **Interactive UI**: Clean, modern Streamlit interface
- **Production Ready**: Trained ResNet50 model with 80%+ accuracy

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/steph-grigors/facial-emotion-recognition.git
cd facial-emotion-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Makefile Commands

Quick reference for available commands:

**Setup & Installation:**
```bash
make install          # Install dependencies
make clean            # Clean generated files and caches
```

**Data Pipeline:**
```bash
make download-data    # Download FER dataset from Kaggle
make preprocess       # Run complete data preprocessing pipeline
```

**Training:**
```bash
make train            # Train Enhanced CNN (default)
make train-baseline   # Train Baseline CNN
make train-enhanced   # Train Enhanced CNN (ResNet50)
```

**Inference & Testing:**
```bash
make inference        # Run quick inference on test image
make inspect          # Inspect model architecture
```

**Web Application:**
```bash
make app              # Run Streamlit web app
```
Visit `http://localhost:8501` in your browser.

## 📁 Project Structure
```
facial-emotion-recognition/
├── app.py                 # Streamlit application
├── src/                   # Source code modules
│   ├── models.py         # Model architectures
│   ├── inference.py      # Prediction logic
│   ├── data_loading.py   # Data utilities
│   └── training.py       # Training orchestrator
├── models/               # Trained model files
│   └── final/
│       └── best_model.pth
├── scripts/              # Training & utility scripts
├── data/                 # Dataset (not in repo)
└── requirements.txt      # Dependencies
```

## 🧠 Model

**Architecture:** ResNet50 (Transfer Learning)
- Pretrained on ImageNet
- Fine-tuned on FER dataset
- Input: 128×128 RGB images
- Output: 7 emotion classes

**Performance:**
- Validation Accuracy: ~80%
- Real-time inference: <100ms per image


## 🎨 Usage Examples

### Web Interface
1. Launch the Streamlit app
2. Choose input method (Upload or Camera)
3. View emotion predictions with confidence scores

### Programmatic Usage
```python
from src.inference import create_predictor

# Load model
predictor = create_predictor(
    model_path='models/final/best_model.pth',
    model_type='enhanced'
)

# Predict emotion
emotion, confidence, probabilities = predictor.predict('image.jpg')
print(f"Emotion: {emotion} ({confidence:.1f}%)")
```

## 📊 Dataset

Uses the [Facial Emotion Recognition Dataset](https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset/data?suggestionBundleId=1420) with 7 emotion categories:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+
- Streamlit 1.28+
- OpenCV
- Pillow

See `requirements.txt` for complete list.


## 📧 Contact

**Stéphan Grigorescu**
- Portfolio: [stephan-gs.work](https://stephan-gs.work)
- GitHub: [@steph-grigors](https://github.com/steph-grigors)
- LinkedIn: [@stéphan-grs](https://www.linkedin.com/in/stéphan-grs)


---

⭐ Star this repo if you find it useful!