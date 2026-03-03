"""
Ablation study: Evaluate impact of removing components.
Phase 6: Ablation study with model variants.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import pandas as pd

from model import TransformerRecommenderAblation
from train import train_epoch, validate, TrainingConfig, get_device
from eval import evaluate_model, format_metrics_table


def run_ablation_study(
    train_loader,
    val_loader,
    test_loader,
    config_dict: dict,
    train_config: TrainingConfig,
    device: str
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study on transformer model.
    
    Variants:
    1. Full Model (baseline)
    2. No Sequence (shuffled order)
    3. No User Features
    4. No Context Features
    """
    
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)
    
    variants = [
        ('Full Model', 'full'),
        ('No Sequence', 'no_sequence'),
        ('No User Features', 'no_user_features'),
        ('No Context Features', 'no_context_features'),
    ]
    
    results = {}
    
    for variant_name, ablation_type in variants:
        print(f"\n{variant_name} (ablation_type={ablation_type})")
        print("-" * 40)
        
        # Create model
        model = TransformerRecommenderAblation(
            config_dict,
            ablation_type=ablation_type,
            dropout=train_config.dropout
        ).to(device)
        
        # Train
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )
        criterion = nn.BCELoss()
        
        best_val_ndcg = 0.0
        
        for epoch in range(train_config.num_epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                description=f"  [{variant_name}] Epoch {epoch+1}"
            )
            
            val_metrics = validate(
                model, val_loader, device, criterion
            )
            
            val_ndcg = val_metrics.get('ndcg_at_k', 0)
            
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
            
            if epoch % (train_config.num_epochs // 2) == 0:
                print(f"  Epoch {epoch+1}: Val NDCG@8={val_ndcg:.4f}")
        
        # Final evaluation on test set
        test_metrics = evaluate_model(
            model, test_loader, device, k=8, per_cart=True
        )
        
        results[variant_name] = {
            'Precision@8': test_metrics.get('precision_at_k', 0),
            'Recall@8': test_metrics.get('recall_at_k', 0),
            'NDCG@8': test_metrics.get('ndcg_at_k', 0),
            'AUC': test_metrics.get('auc', 0),
        }
        
        print(f"\n  Final Test Results:")
        print(f"    Precision@8: {results[variant_name]['Precision@8']:.4f}")
        print(f"    Recall@8: {results[variant_name]['Recall@8']:.4f}")
        print(f"    NDCG@8: {results[variant_name]['NDCG@8']:.4f}")
        print(f"    AUC: {results[variant_name]['AUC']:.4f}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print()
    print(format_metrics_table(results))
    
    return results
