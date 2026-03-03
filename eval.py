"""
Ranking evaluation metrics.
Phase 5: Proper ranking evaluation (Precision@K, Recall@K, NDCG@K, AUC).
"""

import numpy as np
import torch
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score
import pandas as pd


# =====================
# RANKING METRICS
# =====================

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Discounted Cumulative Gain at k.
    
    Args:
        relevances: Binary relevance scores in ranking order.
        k: Cutoff position.
    
    Returns:
        DCG@k score.
    """
    relevances = np.asarray(relevances)[:k]
    gains = relevances / np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains)


def idcg_at_k(num_relevant: int, k: int) -> float:
    """
    Ideal DCG at k (all relevant items ranked first).
    """
    if num_relevant == 0:
        return 0.0
    max_relevances = np.ones(min(num_relevant, k))
    return dcg_at_k(max_relevances, k)


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Normalized DCG at k.
    """
    dcg = dcg_at_k(relevances, k)
    idcg = idcg_at_k(np.sum(relevances), k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(relevances: np.ndarray, k: int) -> float:
    """Precision at k."""
    return np.sum(relevances[:k]) / k if k > 0 else 0.0


def recall_at_k(relevances: np.ndarray, k: int) -> float:
    """Recall at k."""
    total_relevant = np.sum(relevances)
    if total_relevant == 0:
        return 0.0
    return np.sum(relevances[:k]) / total_relevant


# =====================
# BATCH EVALUATION
# =====================

def evaluate_ranking_batch(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int = 8
) -> Dict[str, float]:
    """
    Evaluate ranking metrics on a batch.
    
    Args:
        predictions: Model predictions (batch_size,)
        labels: Ground truth labels (batch_size,)
        k: Cutoff for ranking metrics
    
    Returns:
        Dict with Precision@k, Recall@k, NDCG@k, AUC
    """
    
    predictions = predictions.cpu().detach().numpy().flatten()
    labels = labels.cpu().detach().numpy().flatten().astype(int)
    
    # Sort by predictions (descending)
    sorted_indices = np.argsort(-predictions)
    relevances = labels[sorted_indices]
    
    metrics = {
        'precision_at_k': precision_at_k(relevances, k),
        'recall_at_k': recall_at_k(relevances, k),
        'ndcg_at_k': ndcg_at_k(relevances, k),
        'auc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    }
    
    return metrics


# =====================
# PER-CART EVALUATION
# =====================

def evaluate_per_cart(
    df_with_predictions: pd.DataFrame,
    order_id_col: str = "cart_state_id",
    pred_col: str = "pred_score",
    label_col: str = "label",
    k: int = 8
) -> Dict[str, float]:
    """
    Evaluate ranking metrics per cart-state (per cart_state_id).
    
    Each row in df_with_predictions contains:
    - cart_state_id: Unique cart-state identifier (order_id + step)
    - pred_score: Model prediction
    - label: Ground truth label
    
    For each unique cart_state_id:
    - Rank candidates by pred_score
    - Calculate ranking metrics
    - Average across all cart states
    
    Args:
        df_with_predictions: DataFrame with predictions
        order_id_col: Column name for cart-state identifier
        pred_col: Column name for predictions
        label_col: Column name for labels
        k: Cutoff position
    
    Returns:
        Dict with aggregated metrics
    """
    
    # Fallback: if requested column doesn't exist, try alternatives
    if order_id_col not in df_with_predictions.columns:
        if "cart_state_id" in df_with_predictions.columns:
            order_id_col = "cart_state_id"
        elif "order_id" in df_with_predictions.columns:
            order_id_col = "order_id"
        else:
            raise ValueError(f"Neither '{order_id_col}' nor 'order_id' found in DataFrame columns")
    
    metrics_list = []
    group_sizes = []
    
    for cart_id, group in df_with_predictions.groupby(order_id_col):
        # Rank by predictions (descending)
        sorted_group = group.sort_values(pred_col, ascending=False)
        relevances = sorted_group[label_col].values.astype(int)
        group_sizes.append(len(group))
        
        # Calculate metrics for this cart state
        metrics_list.append({
            'precision_at_k': precision_at_k(relevances, k),
            'recall_at_k': recall_at_k(relevances, k),
            'ndcg_at_k': ndcg_at_k(relevances, k),
            'mrr': (1.0 / (np.where(relevances == 1)[0][0] + 1)) if np.any(relevances == 1) else 0.0
        })
    
    if not metrics_list:
        return {
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'ndcg_at_k': 0.0,
            'mrr': 0.0
        }
    
    # Sanity check: print group size statistics
    group_sizes = np.array(group_sizes)
    print(f"  [Eval Sanity] Groups: {len(group_sizes)}, "
          f"Avg size: {group_sizes.mean():.1f}, "
          f"Min: {group_sizes.min()}, Max: {group_sizes.max()}, "
          f"Median: {np.median(group_sizes):.0f}")
    
    # Average across all cart states
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    # Also compute AUC across all predictions
    if label_col in df_with_predictions.columns and pred_col in df_with_predictions.columns:
        labels = df_with_predictions[label_col].values.astype(int)
        preds = df_with_predictions[pred_col].values
        
        if len(np.unique(labels)) > 1:
            avg_metrics['auc'] = roc_auc_score(labels, preds)
        else:
            avg_metrics['auc'] = 0.0
    
    return avg_metrics


# =====================
# FULL EVALUATION
# =====================

def evaluate_model(
    model,
    dataloader,
    device: str,
    k: int = 8,
    per_cart: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on a full dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with batches
        device: 'cpu' or 'cuda'
        k: Ranking cutoff
        per_cart: If True, evaluate per cart; else aggregate all
    
    Returns:
        Average metrics
    """
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_cart_state_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            cart = batch['cart'].to(device)
            candidate = batch['candidate'].to(device)
            user_features = batch['user_features'].to(device)
            rest_features = batch['rest_features'].to(device)
            context_features = batch['context_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            predictions = model(cart, candidate, user_features, rest_features, context_features)
            
            all_predictions.append(predictions.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten().astype(int))
            
            # Collect cart_state_id for per-cart-state evaluation
            if 'cart_state_id' in batch:
                all_cart_state_ids.extend(batch['cart_state_id'])
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    if per_cart and all_cart_state_ids:
        # Per-cart-state evaluation
        eval_df = pd.DataFrame({
            'cart_state_id': all_cart_state_ids,
            'pred_score': all_predictions,
            'label': all_labels
        })
        metrics = evaluate_per_cart(eval_df, order_id_col='cart_state_id', k=k)
    else:
        # Aggregate evaluation (fallback)
        sorted_indices = np.argsort(-all_predictions)
        relevances = all_labels[sorted_indices]
        
        metrics = {
            'precision_at_k': precision_at_k(relevances, k),
            'recall_at_k': recall_at_k(relevances, k),
            'ndcg_at_k': ndcg_at_k(relevances, k),
            'auc': roc_auc_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else 0.0
        }
    
    return metrics


# =====================
# METRICS FORMATTING
# =====================

def format_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format results dictionary as a nice table.
    
    Args:
        results: Dict like {model_name: {metric: value}}
    
    Returns:
        Formatted string table
    """
    
    # Collect all metrics
    metrics = set()
    for model_results in results.values():
        metrics.update(model_results.keys())
    
    metrics = sorted(list(metrics))
    
    # Build table
    header = ["Model"] + metrics
    rows = []
    
    for model_name in sorted(results.keys()):
        row = [model_name]
        for metric in metrics:
            value = results[model_name].get(metric, 0.0)
            row.append(f"{value:.4f}")
        rows.append(row)
    
    # ASCII table
    col_widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    
    table = ""
    # Header
    table += " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(header)) + "\n"
    table += "-" * (sum(col_widths) + 3 * (len(header) - 1)) + "\n"
    
    # Rows
    for row in rows:
        table += " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row)) + "\n"
    
    return table
