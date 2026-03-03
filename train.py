"""
Training loop for Transformer model.
Phase 4: Training with proper early stopping and validation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm
import random
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def create_balanced_sampler(df, label_col='label', pos_weight=1.0):
    """
    Create sample weights for negative sampling.
    
    Negative sampling ratio typically 1 positive : 5 negatives.
    """
    pos_count = (df[label_col] == 1).sum()
    neg_count = (df[label_col] == 0).sum()
    
    # Calculate target ratio
    neg_sample_ratio = 5  # 1:5 positive:negative
    target_neg_per_epoch = pos_count * neg_sample_ratio
    
    if target_neg_per_epoch < neg_count:
        # Subsample negatives
        pos_indices = df[df[label_col] == 1].index.tolist()
        neg_indices = df[df[label_col] == 0].index.tolist()
        
        sampled_neg = np.random.choice(neg_indices, int(target_neg_per_epoch), replace=False)
        keep_indices = pos_indices + list(sampled_neg)
        
        return keep_indices
    else:
        # Keep all
        return df.index.tolist()


class TrainingConfig:
    """Training configuration."""
    
    def __init__(
        self,
        batch_size: int = 1024,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        early_stopping_patience: int = 3,
        early_stopping_metric: str = 'ndcg_at_k',
        checkpoint_dir: str = 'checkpoints',
        save_checkpoints: bool = True
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device: str,
    description: str = "Training"
) -> float:
    """Train one epoch and return average loss."""
    
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc=description, leave=False):
        cart = batch['cart'].to(device)
        candidate = batch['candidate'].to(device)
        user_features = batch['user_features'].to(device)
        rest_features = batch['rest_features'].to(device)
        context_features = batch['context_features'].to(device)
        labels = batch['label'].to(device).view(-1, 1)  # Reshape to [batch_size, 1]
        
        # Forward pass
        predictions = model(cart, candidate, user_features, rest_features, context_features)
        
        # Loss
        loss = criterion(predictions, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(
    model,
    dataloader,
    device: str,
    criterion = None,
    description: str = "Validation"
) -> Dict[str, float]:
    """Validate model and compute metrics."""
    
    from eval import evaluate_model
    
    model.eval()
    total_loss = 0.0
    
    if criterion is not None:
        with torch.no_grad():
            for batch in dataloader:
                cart = batch['cart'].to(device)
                candidate = batch['candidate'].to(device)
                user_features = batch['user_features'].to(device)
                rest_features = batch['rest_features'].to(device)
                context_features = batch['context_features'].to(device)
                labels = batch['label'].to(device).view(-1, 1)  # Reshape to [batch_size, 1]
                
                predictions = model(cart, candidate, user_features, rest_features, context_features)
                loss = criterion(predictions, labels)
                total_loss += loss.item()
    
    # Evaluate ranking metrics
    metrics = evaluate_model(model, dataloader, device, k=8, per_cart=True)
    
    if criterion is not None:
        metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    config: TrainingConfig,
    device: str
) -> Tuple[dict, list]:
    """
    Train model with early stopping.
    
    Returns:
        (best_metrics, training_history)
    """
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    criterion = nn.BCELoss()
    
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    training_history = []
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / 'best_model.pt'
    last_checkpoint_path = checkpoint_dir / 'last_model.pt'
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Dropout: {config.dropout}")
    print()
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            description=f"  Train"
        )
        
        # Validation
        val_metrics = validate(
            model, val_loader, device, criterion,
            description=f"  Val"
        )
        val_loss = val_metrics.get('loss', 0.0)
        
        # Logging
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Precision@8: {val_metrics.get('precision_at_k', 0):.4f}")
        print(f"  Val NDCG@8: {val_metrics.get('ndcg_at_k', 0):.4f}")
        print(f"  Val AUC: {val_metrics.get('auc', 0):.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **val_metrics
        })
        
        # Early stopping
        current_metric = val_metrics.get(config.early_stopping_metric, 0)
        
        if best_metrics is None or current_metric > best_metrics.get(config.early_stopping_metric, 0):
            best_metrics = val_metrics
            best_epoch = epoch + 1
            patience_counter = 0

            if config.save_checkpoints:
                torch.save(
                    {
                        'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_metrics': best_metrics,
                        'training_config': {
                            'batch_size': config.batch_size,
                            'num_epochs': config.num_epochs,
                            'learning_rate': config.learning_rate,
                            'weight_decay': config.weight_decay,
                            'dropout': config.dropout,
                            'early_stopping_patience': config.early_stopping_patience,
                            'early_stopping_metric': config.early_stopping_metric,
                        },
                    },
                    best_checkpoint_path,
                )
                print(f"  Saved best checkpoint: {best_checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  [No improvement for {patience_counter} epochs]")
        
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    print(f"\nBest model at epoch {best_epoch}")
    print(f"Best Val NDCG@8: {best_metrics.get('ndcg_at_k', 0):.4f}")

    if config.save_checkpoints:
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'last_metrics': training_history[-1] if training_history else {},
            },
            last_checkpoint_path,
        )
        print(f"Saved last checkpoint: {last_checkpoint_path}")

        if best_checkpoint_path.exists():
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint for final evaluation: {best_checkpoint_path}")
    
    return best_metrics, training_history
