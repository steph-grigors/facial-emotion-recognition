# Makefile for Facial Emotion Recognition Project
# Provides convenient commands for common tasks

.PHONY: help setup install install-dev clean data train evaluate test docker-build docker-run lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup          - Create directory structure"
	@echo "  make install        - Install dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make clean          - Clean generated files"
	@echo "  make data           - Download and prepare dataset"
	@echo "  make train          - Train the model"
	@echo "  make evaluate       - Evaluate trained model"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make notebook       - Start Jupyter notebook"
	@echo "  make streamlit      - Run Streamlit app"
	@echo "  make api            - Run FastAPI server"

# Setup project structure
setup:
	@echo "Creating directory structure..."
	mkdir -p data/raw data/processed data/augmented
	mkdir -p notebooks
	mkdir -p src/data src/models src/training src/utils src/inference
	mkdir -p configs/model_configs configs/training_configs
	mkdir -p experiments/runs
	mkdir -p models/checkpoints models/final
	mkdir -p results/metrics results/plots results/confusion_matrices results/reports
	mkdir -p tests
	mkdir -p scripts
	mkdir -p app/api app/static
	mkdir -p docker
	mkdir -p logs
	@echo "Creating __init__.py files..."
	touch src/__init__.py
	touch src/data/__init__.py
	touch src/models/__init__.py
	touch src/training/__init__.py
	touch src/utils/__init__.py
	touch src/inference/__init__.py
	touch tests/__init__.py
	@echo "Directory structure created!"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
	@echo "Cleaned!"

# Download and prepare data
data:
	@echo "Downloading dataset from Kaggle..."
	python scripts/download_data.py
	@echo "Preprocessing data..."
	python scripts/preprocess_data.py

# Train model
train:
	python scripts/train.py --config configs/config.yaml

# Train with specific model config
train-baseline:
	python scripts/train.py --config configs/config.yaml --model-config configs/model_configs/baseline_cnn.yaml

train-resnet:
	python scripts/train.py --config configs/config.yaml --model-config configs/model_configs/resnet50.yaml

train-efficientnet:
	python scripts/train.py --config configs/config.yaml --model-config configs/model_configs/efficientnet.yaml

# Evaluate model
evaluate:
	python scripts/evaluate.py --model-path models/final/best_model.pth

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Run linters
lint:
	@echo "Running flake8..."
	flake8 src/ tests/ scripts/
	@echo "Running pylint..."
	pylint src/ tests/ scripts/
	@echo "Running mypy..."
	mypy src/

# Format code
format:
	@echo "Formatting with black..."
	black src/ tests/ scripts/
	@echo "Sorting imports with isort..."
	isort src/ tests/ scripts/

# Build Docker image
docker-build:
	docker build -t facial-emotion-recognition:latest -f docker/Dockerfile .

# Run Docker container
docker-run:
	docker-compose -f docker/docker-compose.yml up

# Start Jupyter notebook
notebook:
	jupyter notebook notebooks/

# Run Streamlit app
streamlit:
	streamlit run app/streamlit_app.py

# Run FastAPI server
api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# TensorBoard
tensorboard:
	tensorboard --logdir experiments/runs

# Export model
export:
	python scripts/export_model.py --model-path models/final/best_model.pth --output-path models/final/model.onnx
