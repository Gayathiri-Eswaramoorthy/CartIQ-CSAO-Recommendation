"""
Quick validation script to test all pipeline components.
Runs on a small subset of data for fast iteration.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

print("Testing imports...")
from data import preprocess_full_pipeline, CartRecommendationDataset
from model import TransformerRecommender
from train import get_device, set_seed
from baselines import PopularityBaseline
from eval import evaluate_per_cart

print("✓ All imports successful")

# Set seed
set_seed(42)
device = get_device()

print(f"Device: {device}")

# Load and preprocess data
print("\nLoading data...")
train_df, val_df, test_df, config = preprocess_full_pipeline()

# Create small sample for testing
train_df = train_df.sample(min(1000, len(train_df)), random_state=42)
val_df = val_df.sample(min(500, len(val_df)), random_state=42)
test_df = test_df.sample(min(500, len(test_df)), random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Test DataLoader
print("\nTesting DataLoader...")
dataset = CartRecommendationDataset(
    train_df,
    user_feature_cols=config['user_feature_cols'],
    rest_feature_cols=config['rest_feature_cols'],
    context_feature_cols=config['context_feature_cols'],
    max_cart_length=config['max_cart_length'],
    num_items=config['num_items']
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(loader))
print(f"✓ Batch keys: {list(batch.keys())}")
print(f"✓ Cart shape: {batch['cart'].shape}")
print(f"✓ Label shape: {batch['label'].shape}")

# Test model forward pass
print("\nTesting Transformer model...")
model = TransformerRecommender(config, dropout=0.1).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

cart = batch['cart'].to(device)
candidate = batch['candidate'].to(device)
user_features = batch['user_features'].to(device)
rest_features = batch['rest_features'].to(device)
context_features = batch['context_features'].to(device)

output = model(cart, candidate, user_features, rest_features, context_features)
print(f"✓ Output shape: {output.shape}")
print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")

# Test baseline
print("\nTesting Popularity baseline...")
pop_baseline = PopularityBaseline()
pop_baseline.fit(train_df)
test_df_with_pred = test_df.copy()
test_df_with_pred['pred_score'] = pop_baseline.predict(test_df)
print(f"✓ Predictions: min={test_df_with_pred['pred_score'].min():.4f}, max={test_df_with_pred['pred_score'].max():.4f}")

# Test evaluation
print("\nTesting evaluation metrics...")
metrics = evaluate_per_cart(test_df_with_pred, k=8)
print(f"✓ Precision@8: {metrics.get('precision_at_k', 0):.4f}")
print(f"✓ NDCG@8: {metrics.get('ndcg_at_k', 0):.4f}")
print(f"✓ AUC: {metrics.get('auc', 0):.4f}")

print("\n" + "="*60)
print("✓ VALIDATION COMPLETE - All components working!")
print("="*60)
print("\nTo run full training, execute: python main.py")
print("Full training will take 10-30 minutes depending on device")
