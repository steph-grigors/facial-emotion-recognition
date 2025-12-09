# ========================================================================
# Makefile for Facial Emotion Recognition Project
# ========================================================================
# Quick commands for common tasks
# Usage: make <command>
# ========================================================================

.PHONY: help install clean preprocess train inference app test

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# ========================================================================
# HELP
# ========================================================================

help:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Facial Emotion Recognition - Available Commands$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup & Installation:$(NC)"
	@echo "  make install          - Install dependencies"
	@echo "  make clean            - Clean generated files and caches"
	@echo ""
	@echo "$(YELLOW)Data Pipeline:$(NC)"
	@echo "  make preprocess       - Run complete data preprocessing pipeline"
	@echo "  make download-data    - Download FER dataset from Kaggle"
	@echo ""
	@echo "$(YELLOW)Training:$(NC)"
	@echo "  make train            - Train Enhanced CNN (default)"
	@echo "  make train-baseline   - Train Baseline CNN"
	@echo "  make train-enhanced   - Train Enhanced CNN (ResNet50)"
	@echo ""
	@echo "$(YELLOW)Inference & Testing:$(NC)"
	@echo "  make inference        - Run quick inference on test image"
	@echo "  make inspect          - Inspect model architecture"
	@echo "  make test             - Run unit tests"
	@echo ""
	@echo "$(YELLOW)Web Application:$(NC)"
	@echo "  make app              - Run Streamlit web app"
	@echo "  make app-dark         - Run dark theme Streamlit app"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(NC)"
	@echo "  make tensorboard      - Launch TensorBoard"
	@echo ""
	@echo "$(BLUE)========================================================================$(NC)"

# ========================================================================
# SETUP & INSTALLATION
# ========================================================================

install:
	@echo "$(GREEN)Installing dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete!$(NC)"

clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "$(GREEN)✓ Cleaned!$(NC)"

# ========================================================================
# DATA PIPELINE
# ========================================================================

download-data:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Downloading FER Dataset from Kaggle$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Make sure you have:$(NC)"
	@echo "  1. Kaggle account"
	@echo "  2. kaggle.json in ~/.kaggle/"
	@echo "  3. Kaggle CLI installed: pip install kaggle"
	@echo ""
	@read -p "Press Enter to continue or Ctrl+C to cancel..." _
	@echo ""
	@echo "$(GREEN)Downloading dataset...$(NC)"
	kaggle datasets download -d fahadullaha/facial-emotion-recognition-dataset
	@echo "$(GREEN)Extracting dataset...$(NC)"
	unzip -q facial-emotion-recognition-dataset.zip -d data/raw/
	rm facial-emotion-recognition-dataset.zip
	@echo "$(GREEN)✓ Dataset downloaded to data/raw/$(NC)"

preprocess:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Running Data Preprocessing Pipeline$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	@echo ""
	@python -c "from src.data.data_splitter import DataSplitter; \
		from src.data.file_organizer import FileOrganizer; \
		import pandas as pd; \
		from pathlib import Path; \
		print('[1/2] Creating stratified splits...'); \
		splitter = DataSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42); \
		df = splitter.load_from_directory('data/raw/train'); \
		train_df, val_df, test_df = splitter.split_data(df); \
		splitter.verify_splits(); \
		Path('results/metrics').mkdir(parents=True, exist_ok=True); \
		splitter.save_splits('results/metrics'); \
		print('\n[2/2] Organizing files...'); \
		train_df = pd.read_csv('results/metrics/train_split.csv'); \
		val_df = pd.read_csv('results/metrics/val_split.csv'); \
		test_df = pd.read_csv('results/metrics/test_split.csv'); \
		organizer = FileOrganizer(output_dir='data/processed', operation='copy', create_dirs=True); \
		organizer.organize_splits(train_df, val_df, test_df); \
		organizer.verify_file_counts(); \
		print('\n✓ Preprocessing complete! Data ready in data/processed/')"

# ========================================================================
# TRAINING
# ========================================================================

train: train-enhanced

train-baseline:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Training Baseline CNN$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	python scripts/train_model.py --config configs/models/baseline_cnn_config.yaml

train-enhanced:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Training Enhanced CNN (ResNet50)$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	python scripts/train_model.py --config configs/models/enhanced_cnn_config.yaml

# ========================================================================
# INFERENCE & TESTING
# ========================================================================

inference:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Running Quick Inference$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	@if [ -z "$(IMAGE)" ]; then \
		echo "$(YELLOW)Usage: make inference IMAGE=path/to/image.jpg$(NC)"; \
		echo "$(YELLOW)Example: make inference IMAGE=data/test/happy/img1.jpg$(NC)"; \
		exit 1; \
	fi
	python scripts/quick_inference.py $(IMAGE)

inspect:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Inspecting Model Architecture$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	python scripts/inspect_model.py models/final/best_model.pth


# ========================================================================
# WEB APPLICATION
# ========================================================================

app:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Launching Streamlit App (Dark Theme)$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Opening at: http://localhost:8501$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(NC)"
	@echo ""
	streamlit run app.py


# ========================================================================
# QUICK WORKFLOWS
# ========================================================================

# Complete setup from scratch
setup-from-scratch: install download-data preprocess
	@echo "$(GREEN)========================================================================$(NC)"
	@echo "$(GREEN)  ✓ Complete setup finished!$(NC)"
	@echo "$(GREEN)========================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  make train          - Train the model"
	@echo "  make app            - Run the web app"


# Full pipeline: preprocess → train → inference
pipeline: preprocess train-enhanced
	@echo "$(GREEN)========================================================================$(NC)"
	@echo "$(GREEN)  ✓ Training pipeline complete!$(NC)"
	@echo "$(GREEN)========================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Model saved to: models/final/best_model.pth$(NC)"
	@echo "$(YELLOW)Run the app with: make app$(NC)"

# ========================================================================
# DEVELOPMENT HELPERS
# ========================================================================

# Check if dataset is downloaded
check-data:
	@if [ -d "data/raw" ]; then \
		echo "$(GREEN)✓ Raw data found$(NC)"; \
	else \
		echo "$(RED)✗ Raw data not found. Run: make download-data$(NC)"; \
	fi
	@if [ -d "data/processed/train" ]; then \
		echo "$(GREEN)✓ Processed data found$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Processed data not found. Run: make preprocess$(NC)"; \
	fi

# Check if model exists
check-model:
	@if [ -f "models/final/best_model.pth" ]; then \
		echo "$(GREEN)✓ Trained model found$(NC)"; \
		python scripts/inspect_model.py models/final/best_model.pth; \
	else \
		echo "$(RED)✗ Model not found. Run: make train$(NC)"; \
	fi

# Show project status
status: check-data check-model
	@echo ""
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Project Status$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"

# ========================================================================
# EXAMPLES
# ========================================================================

examples:
	@echo "$(BLUE)========================================================================$(NC)"
	@echo "$(GREEN)  Example Workflows$(NC)"
	@echo "$(BLUE)========================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)1. Complete setup from scratch:$(NC)"
	@echo "   make setup-from-scratch"
	@echo ""
	@echo "$(YELLOW)2. Train model:$(NC)"
	@echo "   make train              # Enhanced CNN (default)"
	@echo "   make train-baseline     # Baseline CNN"
	@echo ""
	@echo "$(YELLOW)3. Test inference:$(NC)"
	@echo "   make inference IMAGE=path/to/image.jpg"
	@echo ""
	@echo "$(YELLOW)4. Run web app:$(NC)"
	@echo "   make app                # Dark theme (default)"
	@echo "   make app-light          # Light theme"
	@echo ""
	@echo "$(YELLOW)5. Full pipeline:$(NC)"
	@echo "   make pipeline           # Preprocess + Train"
	@echo ""
