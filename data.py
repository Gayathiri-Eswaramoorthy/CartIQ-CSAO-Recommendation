"""
Data loading, preprocessing, and utilities.
Phase 1 & 2: Data validation, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset


# =====================
# DATA LOADING
# =====================

def load_raw_datasets(data_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all raw datasets."""
    users = pd.read_csv(f"{data_dir}/users.csv")
    items = pd.read_csv(f"{data_dir}/items.csv")
    restaurants = pd.read_csv(f"{data_dir}/restaurants.csv")
    training = pd.read_csv(f"{data_dir}/training_data.csv")
    
    return users, items, restaurants, training


# =====================
# TEMPORAL PREPROCESSING
# =====================

def preprocess_training_data(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps and extract temporal features.
    """
    df = training_df.copy()
    
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Extract temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    
    return df


# =====================
# FEATURE MERGING
# =====================

def merge_features(
    training_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    restaurants_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge user, restaurant, and item features into training data.
    """
    df = training_df.copy()
    
    # Merge user features
    user_cols = [c for c in users_df.columns if c != "user_id"]
    df = df.merge(
        users_df.rename(columns={c: f"user_{c}" for c in user_cols}),
        left_on="user_id",
        right_on="user_id",
        how="left"
    )
    
    # Merge restaurant features
    rest_cols = [c for c in restaurants_df.columns if c != "restaurant_id"]
    df = df.merge(
        restaurants_df.rename(columns={c: f"rest_{c}" for c in rest_cols}),
        left_on="restaurant_id",
        right_on="restaurant_id",
        how="left"
    )
    
    # Merge item (candidate) features
    item_cols = [c for c in items_df.columns if c != "item_id"]
    df = df.merge(
        items_df.rename(columns={c: f"item_{c}" for c in item_cols}),
        left_on="candidate_item",
        right_on="item_id",
        how="left"
    )
    
    return df


# =====================
# TEMPORAL SPLIT
# =====================

def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into Train (months 1-8), Val (month 9), Test (month 10).
    No data leakage.
    """
    train = df[df["month"].isin([1, 2, 3, 4, 5, 6, 7, 8])].reset_index(drop=True)
    val = df[df["month"] == 9].reset_index(drop=True)
    test = df[df["month"] == 10].reset_index(drop=True)
    
    return train, val, test


# =====================
# DATA VALIDATION
# =====================

def validate_and_report(df: pd.DataFrame, split_name: str = "Full Dataset"):
    """Print sanity checks."""
    print(f"\n{'='*60}")
    print(f"Data Validation: {split_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(df):,}")
    print(f"Positive rate: {df['label'].mean():.4f}")
    
    cart_sizes = df["cart_state"].apply(lambda x: len(eval(str(x))) if isinstance(x, str) else len(x))
    print(f"Average cart length: {cart_sizes.mean():.2f}")
    print(f"Cart size distribution:")
    print(cart_sizes.value_counts().sort_index().to_string())
    print()


# =====================
# FEATURE NORMALIZATION
# =====================

def get_feature_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Calculate mean and std for numerical features."""
    numerical_cols = [
        "user_budget_sensitivity", "user_veg_preference", 
        "user_dessert_affinity", "user_beverage_affinity", "user_order_frequency",
        "item_price",
        "hour"
    ]
    
    stats = {}
    for col in numerical_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std() + 1e-8  # Avoid division by zero
            stats[col] = (mean, std)
    
    return stats


def normalize_features(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Normalize numerical features using provided stats."""
    df = df.copy()
    
    for col, (mean, std) in stats.items():
        if col in df.columns:
            df[col] = (df[col] - mean) / std
    
    return df


# =====================
# CART PARSING & PADDING
# =====================

def parse_cart_state(cart_str, max_length: int = 6) -> List[int]:
    """Parse cart state string to list of item IDs and pad."""
    try:
        if isinstance(cart_str, str):
            cart = eval(cart_str)
        else:
            cart = cart_str
    except:
        cart = []
    
    # Pad with -1 (special padding token)
    if len(cart) < max_length:
        cart = cart + [-1] * (max_length - len(cart))
    else:
        cart = cart[:max_length]
    
    return cart


def prepare_cart_sequences(df: pd.DataFrame, max_cart_length: int = 6) -> pd.DataFrame:
    """Convert cart_state strings to padded sequences."""
    df = df.copy()
    df["cart_sequence"] = df["cart_state"].apply(
        lambda x: parse_cart_state(x, max_cart_length)
    )
    return df


# =====================
# PYTORCH DATASET
# =====================

class CartRecommendationDataset(Dataset):
    def __init__(
        self,
        df,
        user_feature_cols,
        rest_feature_cols,
        context_feature_cols,
        max_cart_length=6
    ):
        # Convert everything ONCE here
        self.cart = torch.LongTensor(df["cart_sequence"].tolist())
        self.candidate = torch.LongTensor(df["candidate_item"].values)

        self.user_features = torch.FloatTensor(
            df[user_feature_cols].values.astype(np.float32)
        )

        if len(rest_feature_cols) > 0:
            self.rest_features = torch.FloatTensor(
                df[rest_feature_cols].values.astype(np.float32)
            )
        else:
            self.rest_features = None

        self.context_features = torch.FloatTensor(
            df[context_feature_cols].values.astype(np.float32)
        )

        self.labels = torch.FloatTensor(df["label"].values.astype(np.float32))

        # Store cart_state_id as a list of strings (not tensorizable)
        if "cart_state_id" in df.columns:
            self.cart_state_ids = df["cart_state_id"].values.tolist()
        else:
            # Fallback: use order_id if cart_state_id not available
            self.cart_state_ids = df["order_id"].astype(str).values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "cart": self.cart[idx],
            "candidate": self.candidate[idx],
            "user_features": self.user_features[idx],
            "rest_features": self.rest_features[idx] if self.rest_features is not None else torch.FloatTensor([]),
            "context_features": self.context_features[idx],
            "label": self.labels[idx],
            "cart_state_id": self.cart_state_ids[idx]
        }


# =====================
# END-TO-END PREPROCESSING
# =====================

def preprocess_full_pipeline(
    data_dir: str = ".",
    max_cart_length: int = 6,
    test_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline.
    Returns: (train_df, val_df, test_df, config_dict)
    """
    
    print("\n" + "="*60)
    print("Loading raw datasets...")
    print("="*60)
    
    users, items, restaurants, training = load_raw_datasets(data_dir)
    
    print("\n" + "="*60)
    print("Preprocessing training data...")
    print("="*60)
    
    training = preprocess_training_data(training)
    training = merge_features(training, users, items, restaurants)
    training = prepare_cart_sequences(training, max_cart_length)
    
    # Temporal split
    train, val, test = temporal_split(training)
    
    # Validate
    validate_and_report(train, "Training Set")
    validate_and_report(val, "Validation Set")
    validate_and_report(test, "Test Set")
    
    # Compute stats on TRAIN ONLY to avoid leakage
    stats = get_feature_stats(train)
    
    # Normalize all splits using train stats
    train = normalize_features(train, stats)
    val = normalize_features(val, stats)
    test = normalize_features(test, stats)
    
    # Define feature columns for dataset (numeric only)
    user_feature_cols = [c for c in train.columns 
                         if c.startswith("user_") 
                         and c != "user_id" 
                         and train[c].dtype in ['float64', 'int64']]
    
    rest_feature_cols = [c for c in train.columns 
                         if c.startswith("rest_") 
                         and train[c].dtype in ['float64', 'int64']]
    
    context_feature_cols = ["hour", "weekend_flag"]
    
    config = {
        "max_cart_length": max_cart_length,
        "num_items": items["item_id"].max() + 1,
        "num_users": users["user_id"].max() + 1,
        "user_feature_cols": user_feature_cols,
        "rest_feature_cols": rest_feature_cols,
        "context_feature_cols": context_feature_cols,
        "user_feature_dim": len(user_feature_cols),
        "rest_feature_dim": len(rest_feature_cols),
        "context_feature_dim": len(context_feature_cols),
        "feature_stats": stats
    }
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(f"\nConfig Summary:")
    print(f"  Max cart length: {config['max_cart_length']}")
    print(f"  Num items: {config['num_items']}")
    print(f"  User feature dim: {config['user_feature_dim']}")
    print(f"  Restaurant feature dim: {config['rest_feature_dim']}")
    print(f"  Context feature dim: {config['context_feature_dim']}")
    print()
    
    return train, val, test, config
