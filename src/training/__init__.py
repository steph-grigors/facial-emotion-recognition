"""Training utilities and orchestration."""

from .trainer import train_epoch, validate, evaluate_model, train_epoch_simple
from .orchestrator import TrainingOrchestrator
from .early_stopping import EarlyStopping
from .metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_per_class_accuracy
)

__all__ = [
    # Trainer functions
    'train_epoch',
    'train_epoch_simple',
    'validate',
    'evaluate_model',
    
    # Orchestrator
    'TrainingOrchestrator',
    
    # Early stopping
    'EarlyStopping',
    
    # Metrics
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_training_history',
    'plot_per_class_accuracy',
]