"""Training and validation functions."""

import torch
from tqdm import tqdm
from typing import Tuple


def train_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device) -> Tuple[float, float]:
    """
    Validate model.
    
    Args:
        model: PyTorch model
        loader: Validation data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), correct / total


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use (cuda/cpu)
    
    Returns:
        Dictionary with accuracy, predictions, and true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }