"""Training module for facial emotion recognition."""

from .trainer import train_epoch, validate, evaluate_model
from .early_stopping import EarlyStopping
from .metrics import (
    compute_metrics, 
    plot_confusion_matrix, 
    plot_training_history,
    plot_per_class_accuracy
)
from .orchestrator import TrainingOrchestrator

__all__ = [
    'train_epoch',
    'validate',
    'evaluate_model',
    'EarlyStopping',
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_training_history',
    'plot_per_class_accuracy',
    'TrainingOrchestrator',
]