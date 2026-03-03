"""
Baseline models: Popularity and LightGBM.
Phase 3: Simple and non-sequential baselines.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import lightgbm as lgb
from eval import evaluate_per_cart


# =====================
# POPULARITY BASELINE
# =====================

class PopularityBaseline:
    """
    Rank items by historical acceptance frequency per restaurant.
    """
    
    def __init__(self):
        self.item_acceptance_freq = {}  # {restaurant_id: {item_id: count}}
    
    def fit(self, df_train: pd.DataFrame):
        """
        Learn acceptance frequency per restaurant.
        """
        for restaurant_id in df_train['restaurant_id'].unique():
            rest_data = df_train[df_train['restaurant_id'] == restaurant_id]
            
            # Count acceptances per item
            item_accepts = rest_data[rest_data['label'] == 1]['candidate_item'].value_counts()
            
            self.item_acceptance_freq[restaurant_id] = item_accepts.to_dict()
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Score candidates by popularity in their restaurant.
        
        Returns:
            Array of shape (len(df_test),) with scores
        """
        scores = []
        
        for _, row in df_test.iterrows():
            rest_id = row['restaurant_id']
            item_id = row['candidate_item']
            
            if rest_id in self.item_acceptance_freq:
                count = self.item_acceptance_freq[rest_id].get(item_id, 0)
                # Normalize by max in this restaurant
                max_count = max(self.item_acceptance_freq[rest_id].values())
                score = count / max(max_count, 1)
            else:
                score = 0.0
            
            scores.append(score)
        
        return np.array(scores)
    
    def evaluate(self, df_test: pd.DataFrame, k: int = 8) -> Dict[str, float]:
        """Evaluate on test set."""
        df_test = df_test.copy()
        df_test['pred_score'] = self.predict(df_test)
        
        metrics = evaluate_per_cart(df_test, order_id_col="cart_state_id", k=k)
        return metrics


# =====================
# LIGHTGBM BASELINE
# =====================

class LightGBMBaseline:
    """
    Non-sequential baseline: Treat cart as unordered set.
    
    Features:
    - Category counts in cart
    - Cart length
    - User features
    - Restaurant features
    - Context features
    """
    
    def __init__(self, num_workers: int = -1):
        self.model = None
        self.feature_cols = []
        self.num_workers = num_workers
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data."""
        df = df.copy()
        
        # Cart-based features
        # Use pre-computed cart_sequence if available, otherwise parse cart_state
        if 'cart_sequence' in df.columns:
            # Already parsed - just count non-padding items
            df['cart_length'] = df['cart_sequence'].apply(lambda x: len([i for i in x if i >= 0]))
        elif 'cart_state' in df.columns:
            # Need to parse - use vectorized approach
            def get_category_counts(cart_state):
                try:
                    if isinstance(cart_state, str):
                        cart = eval(cart_state)
                    else:
                        cart = cart_state if isinstance(cart_state, list) else []
                except:
                    cart = []
                return len([c for c in cart if c >= 0])
            df['cart_length'] = df['cart_state'].apply(get_category_counts)
        else:
            # Fallback
            df['cart_length'] = 0
        
        # Basic features that should exist
        feature_base = [
            'user_budget_sensitivity',
            'user_veg_preference',
            'user_dessert_affinity',
            'user_beverage_affinity',
            'user_order_frequency',
            'hour',
            'weekend_flag',
            'item_price',
            'item_veg_flag',
            'cart_length'
        ]
        
        # Check which exist
        existing_features = [f for f in feature_base if f in df.columns]
        
        return df, existing_features
    
    def fit(self, df_train: pd.DataFrame):
        """Train baseline model."""
        df_train, self.feature_cols = self._engineer_features(df_train)
        
        X = df_train[self.feature_cols].copy()
        y = df_train['label'].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Train LightGBM
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=self.num_workers,
            verbose=-1
        )
        
        self.model.fit(X, y)
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Generate scores for test set."""
        df_test, _ = self._engineer_features(df_test)
        
        X = df_test[self.feature_cols].copy()
        X = X.fillna(0)
        
        # Get probability of positive class
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, df_test: pd.DataFrame, k: int = 8) -> Dict[str, float]:
        """Evaluate on test set."""
        df_test = df_test.copy()
        df_test['pred_score'] = self.predict(df_test)
        
        metrics = evaluate_per_cart(df_test, order_id_col="cart_state_id", k=k)
        return metrics


# =====================
# UTILITY FUNCTIONS
# =====================

def get_baseline_results(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    k: int = 8
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit and evaluate both baseline models.
    
    Returns:
        (val_results, test_results)
    """
    
    results_val = {}
    results_test = {}
    
    print("\n" + "="*60)
    print("BASELINE 1: Popularity")
    print("="*60)
    
    pop_baseline = PopularityBaseline()
    pop_baseline.fit(df_train)
    
    val_metrics = pop_baseline.evaluate(df_val, k=k)
    test_metrics = pop_baseline.evaluate(df_test, k=k)
    
    results_val['Popularity'] = val_metrics
    results_test['Popularity'] = test_metrics
    
    print(f"Val Precision@{k}: {val_metrics.get('precision_at_k', 0):.4f}")
    print(f"Val NDCG@{k}: {val_metrics.get('ndcg_at_k', 0):.4f}")
    print(f"Test Precision@{k}: {test_metrics.get('precision_at_k', 0):.4f}")
    print(f"Test NDCG@{k}: {test_metrics.get('ndcg_at_k', 0):.4f}")
    
    print("\n" + "="*60)
    print("BASELINE 2: LightGBM")
    print("="*60)
    
    lgbm_baseline = LightGBMBaseline()
    lgbm_baseline.fit(df_train)
    
    val_metrics = lgbm_baseline.evaluate(df_val, k=k)
    test_metrics = lgbm_baseline.evaluate(df_test, k=k)
    
    results_val['LightGBM'] = val_metrics
    results_test['LightGBM'] = test_metrics
    
    print(f"Val Precision@{k}: {val_metrics.get('precision_at_k', 0):.4f}")
    print(f"Val NDCG@{k}: {val_metrics.get('ndcg_at_k', 0):.4f}")
    print(f"Test Precision@{k}: {test_metrics.get('precision_at_k', 0):.4f}")
    print(f"Test NDCG@{k}: {test_metrics.get('ndcg_at_k', 0):.4f}")
    
    return results_val, results_test
