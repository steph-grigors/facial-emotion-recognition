# 🎭 Facial Emotion Recognition

A deep learning project for recognizing facial emotions using state-of-the-art computer vision techniques. This project demonstrates end-to-end machine learning workflow from data preprocessing to model deployment.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements multiple deep learning architectures for facial emotion recognition, including:
- Custom CNN architectures
- Transfer learning with ResNet50
- EfficientNet models
- Ensemble methods

The system can classify facial expressions into 7 categories: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**.

## ✨ Features

- **Multiple Model Architectures**: Baseline CNN, ResNet50, EfficientNet, and ensemble models
- **Advanced Training Techniques**:
  - Transfer learning
  - Mixed precision training
  - Learning rate scheduling
  - Early stopping
  - Data augmentation
- **Experiment Tracking**: Integration with TensorBoard, Weights & Biases, and MLflow
- **Model Interpretability**: Grad-CAM visualizations for model explainability
- **Production Ready**: REST API with FastAPI and interactive demo with Streamlit
- **Comprehensive Testing**: Unit tests and integration tests
- **Docker Support**: Containerized deployment

## 📊 Dataset

This project uses the [Facial Emotion Recognition Dataset](https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset) from Kaggle.

**Dataset Statistics:**
- Number of classes: 7
- Training images: TBD
- Validation images: TBD
- Test images: TBD
- Image size: 224x224 (resized)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Kaggle API credentials

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/facial-emotion-recognition.git
cd facial-emotion-recognition
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
make install
# or
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. **Create project structure**
```bash
make setup
```

6. **Download dataset**
```bash
make data
```

## 📁 Project Structure

```
facial-emotion-recognition/
├── data/                  # Dataset directory
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architectures
│   ├── training/        # Training loops and utilities
│   ├── utils/           # Helper functions
│   └── inference/       # Inference and prediction
├── configs/             # Configuration files
├── scripts/             # Training and evaluation scripts
├── tests/               # Unit tests
├── app/                 # Web application (Streamlit & FastAPI)
├── results/             # Model outputs and visualizations
└── models/              # Saved models
```

For detailed structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 💻 Usage

### Training

**Train with default configuration:**
```bash
make train
```

**Train specific models:**
```bash
make train-baseline      # Baseline CNN
make train-resnet        # ResNet50
make train-efficientnet  # EfficientNet
```

**Train with custom config:**
```bash
python scripts/train.py --config configs/config.yaml --model-config configs/model_configs/resnet50.yaml
```

### Evaluation

```bash
make evaluate
# or
python scripts/evaluate.py --model-path models/final/best_model.pth
```

### Inference

```bash
python scripts/predict.py --image path/to/image.jpg --model-path models/final/best_model.pth
```

### Interactive Demo

**Streamlit App:**
```bash
make streamlit
# or
streamlit run app/streamlit_app.py
```

**FastAPI Server:**
```bash
make api
# or
uvicorn app.api.main:app --reload
```

## 🧠 Models

### Baseline CNN
- Custom architecture with 4 convolutional blocks
- ~2M parameters
- Baseline for comparison

### ResNet50 (Transfer Learning)
- Pretrained on ImageNet
- Fine-tuned on emotion dataset
- ~23M parameters

### EfficientNet-B0
- State-of-the-art efficient architecture
- Best accuracy/efficiency trade-off
- ~5M parameters

### Ensemble
- Combines predictions from multiple models
- Improved robustness and accuracy

## 📈 Results

| Model | Val Accuracy | Test Accuracy | Parameters | Inference Time |
|-------|--------------|---------------|------------|----------------|
| Baseline CNN | TBD | TBD | 2M | TBD ms |
| ResNet50 | TBD | TBD | 23M | TBD ms |
| EfficientNet-B0 | TBD | TBD | 5M | TBD ms |
| Ensemble | TBD | TBD | - | TBD ms |

*Results to be updated after training*

### Confusion Matrix
![Confusion Matrix](results/confusion_matrices/confusion_matrix.png)

### Training Curves
![Training Curves](results/plots/training_curves.png)

## 🎨 Demo

### Streamlit Interface
[Add screenshot here]

### API Usage

```python
import requests

# Predict emotion from image
files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
print(response.json())
```

## 🛠️ Development

### Running Tests
```bash
make test
```

### Code Quality
```bash
make lint    # Run linters
make format  # Format code
```

### Docker

**Build image:**
```bash
make docker-build
```

**Run container:**
```bash
make docker-run
```

## 📝 To-Do

- [ ] Complete baseline model training
- [ ] Implement ResNet50 transfer learning
- [ ] Add EfficientNet experiments
- [ ] Create ensemble model
- [ ] Build Streamlit demo
- [ ] Deploy FastAPI endpoint
- [ ] Write comprehensive tests
- [ ] Add model interpretation visualizations
- [ ] Create Docker deployment
- [ ] Write technical blog post

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset from [Kaggle](https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset)
- Pretrained models from PyTorch
- Inspired by recent advances in emotion recognition research

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/facial-emotion-recognition](https://github.com/yourusername/facial-emotion-recognition)

---

⭐ If you found this project helpful, please consider giving it a star!
