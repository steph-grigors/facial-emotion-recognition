"""Early stopping implementation."""


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience (int): How many epochs to wait after last improvement
        min_delta (float): Minimum change to qualify as improvement
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        """
        Check if training should stop.
        
        Args:
            val_loss (float): Current validation loss
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False