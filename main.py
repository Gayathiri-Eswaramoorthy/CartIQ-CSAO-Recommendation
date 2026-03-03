"""
Main orchestration script.
Runs complete CSAO Recommender System pipeline.
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from data import (
    preprocess_full_pipeline,
    CartRecommendationDataset
)
from model import TransformerRecommender, TransformerRecommenderAblation
from train import (
    train_model,
    TrainingConfig,
    get_device,
    set_seed
)
from eval import evaluate_model, format_metrics_table
from baselines import get_baseline_results
from ablation import run_ablation_study
from business_impact import run_business_impact_analysis


def main():
    """Run complete pipeline."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # =====================
    # PHASE 1-2: DATA PREPROCESSING
    # =====================
    
    print("\n" + "="*70)
    print("PHASE 1-2: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*70)
    
    train_df, val_df, test_df, config = preprocess_full_pipeline(
        data_dir=".",
        max_cart_length=6
    )
    
    # =====================
    # CREATE DATALOADERS
    # =====================
    
    print("\n" + "="*70)
    print("Creating PyTorch Dataloaders")
    print("="*70)
    
    batch_size = 1024
    print(f"Batch Size: {batch_size}")
    
    # Create datasets
    train_dataset = CartRecommendationDataset(
        train_df,
        user_feature_cols=config['user_feature_cols'],
        rest_feature_cols=config['rest_feature_cols'],
        context_feature_cols=config['context_feature_cols'],
        max_cart_length=config['max_cart_length']
    )
    
    val_dataset = CartRecommendationDataset(
        val_df,
        user_feature_cols=config['user_feature_cols'],
        rest_feature_cols=config['rest_feature_cols'],
        context_feature_cols=config['context_feature_cols'],
        max_cart_length=config['max_cart_length']
    )
    
    test_dataset = CartRecommendationDataset(
        test_df,
        user_feature_cols=config['user_feature_cols'],
        rest_feature_cols=config['rest_feature_cols'],
        context_feature_cols=config['context_feature_cols'],
        max_cart_length=config['max_cart_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # =====================
    # PHASE 3: BASELINE MODELS
    # =====================
    
    print("\n" + "="*70)
    print("PHASE 3: BASELINE MODELS")
    print("="*70)
    
    baseline_val_results, baseline_test_results = get_baseline_results(
        train_df, val_df, test_df, k=8
    )
    
    # =====================
    # PHASE 4: TRANSFORMER TRAINING
    # =====================
    
    print("\n" + "="*70)
    print("PHASE 4: TRANSFORMER MODEL TRAINING")
    print("="*70)
    
    # Create model
    transformer_model = TransformerRecommender(config, dropout=0.1).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    # Training config
    train_config = TrainingConfig(
        batch_size=batch_size,
        num_epochs=15,
        learning_rate=1e-3,
        weight_decay=1e-5,
        dropout=0.1,
        early_stopping_patience=3,
        early_stopping_metric='ndcg_at_k'
    )
    
    # Train
    best_metrics, history = train_model(
        transformer_model,
        train_loader,
        val_loader,
        train_config,
        device
    )
    
    # =====================
    # PHASE 5: EVALUATION
    # =====================
    
    print("\n" + "="*70)
    print("PHASE 5: MODEL EVALUATION")
    print("="*70)
    
    transformer_test_results = evaluate_model(
        transformer_model, test_loader, device, k=8, per_cart=True
    )
    
    # Rename key for consistency
    transformer_test_results = {
        f"{k}": v for k, v in transformer_test_results.items()
    }
    
    # Create comparison table
    all_results = {
        'Popularity': baseline_test_results['Popularity'],
        'LightGBM': baseline_test_results['LightGBM'],
        'Transformer': {
            'precision_at_k': transformer_test_results.get('precision_at_k', 0),
            'recall_at_k': transformer_test_results.get('recall_at_k', 0),
            'ndcg_at_k': transformer_test_results.get('ndcg_at_k', 0),
            'auc': transformer_test_results.get('auc', 0),
        }
    }
    
    # Format for display
    display_results = {
        name: {
            'Precision@8': metrics.get('precision_at_k', 0),
            'Recall@8': metrics.get('recall_at_k', 0),
            'NDCG@8': metrics.get('ndcg_at_k', 0),
            'AUC': metrics.get('auc', 0),
        }
        for name, metrics in all_results.items()
    }
    
    print("\n" + "="*70)
    print("TEST SET RESULTS COMPARISON")
    print("="*70)
    print()
    print(format_metrics_table(display_results))
    
    # =====================
    # PHASE 6: ABLATION STUDY
    # =====================
    
    print("\n" + "="*70)
    print("PHASE 6: ABLATION STUDY")
    print("="*70)
    
    # Re-train for ablation (smaller model for speed)
    # Optional: skip if you want to save time
    try:
        ablation_results = run_ablation_study(
            train_loader,
            val_loader,
            test_loader,
            config,
            train_config,
            device
        )
    except Exception as e:
        print(f"Ablation study skipped: {e}")
        ablation_results = None
    
    # =====================
    # PHASE 7: BUSINESS IMPACT
    # =====================
    
    print("\n" + "="*70)
    print("PHASE 7: BUSINESS IMPACT SIMULATION")
    print("="*70)
    
    # Get baseline and transformer metrics
    baseline_prec = baseline_test_results['LightGBM'].get('precision_at_k', 0.05)
    transformer_prec = all_results['Transformer'].get('precision_at_k', 0.08)
    
    business_metrics = run_business_impact_analysis(
        baseline_precision_at_8=baseline_prec,
        transformer_precision_at_8=transformer_prec,
        current_attach_rate=0.15,
        avg_addon_value=60.0,
        monthly_orders=10_000_000
    )
    
    # =====================
    # SUMMARY
    # =====================
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"✓ Data preprocessing: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")
    print(f"✓ Baseline models trained (Popularity, LightGBM)")
    print(f"✓ Transformer model: Precision@8={all_results['Transformer']['precision_at_k']:.4f}, NDCG@8={all_results['Transformer']['ndcg_at_k']:.4f}")
    print(f"✓ Projected annual revenue uplift: ₹{business_metrics['incremental_annual_revenue']:,.0f}")
    
    if ablation_results:
        print(f"✓ Ablation study completed ({len(ablation_results)} variants)")
    
    print("\nAll components ready for production deployment.\n")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'model': transformer_model,
        'config': config,
        'results': display_results,
        'business_metrics': business_metrics,
        'ablation_results': ablation_results
    }


if __name__ == "__main__":
    artifacts = main()
