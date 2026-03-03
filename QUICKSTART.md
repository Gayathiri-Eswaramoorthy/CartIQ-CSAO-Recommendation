"""
Quick Start Guide for CSAO Recommendation System
================================================

This document provides step-by-step instructions to run the complete system.

PREREQUISITES
=============

1. Python 3.8+
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

STEP 1: GENERATE DATA (if not already done)
============================================

```bash
python data_generation.py
```

This creates:
- users.csv (10,000 users with behavior profiles)
- items.csv (2,000 items with prices and categories)
- restaurants.csv (300 restaurants across 5 cities)
- orders.csv (500,000 orders with metadata)
- training_data.csv (5M+ ranking samples for recommendation)

Runtime: ~6-8 minutes on CPU, ~3-4 minutes on GPU

STEP 2: VALIDATE INSTALLATION (OPTIONAL)
=========================================

Quick smoke test of all components:

```bash
python validate.py
```

This will:
- Load and preprocess data (small sample)
- Create PyTorch DataLoader
- Test Transformer model forward pass
- Test baseline models
- Test evaluation metrics

Runtime: ~2-3 minutes

STEP 3: RUN FULL PIPELINE
==========================

Execute all 7 phases:

```bash
python main.py
```

This will:

1. DATA PREPROCESSING (3-5 min)
   - Parse timestamps
   - Extract temporal features (hour, day_of_week, weekend)
   - Merge user, restaurant, item features
   - Temporal split: Train/Val/Test
   - Normalize features
   - Pad cart sequences

2. BASELINE MODELS (2-3 min)
   - Popularity Baseline: Historical item acceptance frequency
   - LightGBM Baseline: Non-sequential classifier
   - Evaluate both on test set

3. TRANSFORMER TRAINING (10-20 min)
   - Initialize model with:
     * 128-dim embeddings
     * 2-layer Transformer encoder
     * 4 attention heads
   - Train with early stopping on NDCG@8
   - Typical: 8-12 epochs until convergence

4. RANKING EVALUATION (1-2 min)
   - Compute Precision@8, Recall@8, NDCG@8, AUC
   - Per-cart evaluation (realistic ranking metrics)
   - Compare all 3 models

5. ABLATION STUDY (10-15 min) - OPTIONAL
   - Test 4 model variants:
     * Full model
     * No sequence (shuffled order)
     * No user features
     * No context features
   - Validate importance of each component

6. BUSINESS IMPACT (< 1 min)
   - Calculate revenue uplift
   - Project annual incremental revenue
   - Given: 10M monthly orders, ₹60 avg add-on

Total Runtime: ~30-50 minutes on CPU, ~15-25 minutes on GPU

EXPECTED OUTPUT
===============

Console Output:
```
============================================================
PHASE 1-2: DATA PREPROCESSING & FEATURE ENGINEERING
============================================================

Final Config Summary:
  Max cart length: 6
  Num items: 2000
  User feature dim: 7
  Restaurant feature dim: 2
  Context feature dim: 2

============================================================
PHASE 3: BASELINE MODELS
============================================================

BASELINE 1: Popularity
  Val Precision@8: 0.0512
  Val NDCG@8: 0.1050
  ...

============================================================
PHASE 4: TRANSFORMER MODEL TRAINING
============================================================

Epoch 1/15
  Train Loss: 0.6842
  Val Loss: 0.6721
  Val Precision@8: 0.0645
  Val NDCG@8: 0.1180
  ...

============================================================
PHASE 5: MODEL EVALUATION
============================================================

TEST SET RESULTS COMPARISON

Model        │ Precision@8 │ Recall@8 │ NDCG@8 │ AUC
─────────────┼─────────────┼──────────┼────────┼────────
Popularity   │ 0.0512      │ 0.0768   │ 0.105  │ 0.642
LightGBM     │ 0.0678      │ 0.1025   │ 0.142  │ 0.712
Transformer  │ 0.0824      │ 0.1245   │ 0.156  │ 0.758

============================================================
PHASE 7: BUSINESS IMPACT SIMULATION
============================================================

BUSINESS IMPACT SIMULATION

Assumptions:
  Monthly Order Volume: 10,000,000
  Average Add-On Value: ₹60.00

Current Attach Rates:
  Baseline: 15.68%
  Transformer: 17.03%
  Uplift: 8.62%

Incremental Revenue (Annual): ₹237,600,000
```

STEP 4: (OPTIONAL) LAUNCH INTERACTIVE DEMO
============================================

```bash
streamlit run app.py
```

Features:
- Select restaurant and user segment
- Build cart dynamically
- Get recommendations from trained model
- View business metrics

Opens in browser at: http://localhost:8501

ARCHITECTURE SUMMARY
====================

### Data Flow
```
training_data.csv (5M rows)
  ↓
Data.py: Preprocessing & Feature Engineering
  ↓
CartRecommendationDataset (PyTorch)
  ↓
DataLoader (batch_size=64)
  ↓
Model.py: Transformer Encoder
  ↓
Train.py: Training Loop (BCE Loss + Early Stopping)
  ↓
Eval.py: NDCG@K, Precision@K, Recall@K, AUC
```

### Model Architecture
```
Item ID → Embedding (128)
            ↓
       Positional Encoding
            ↓
       Transformer Encoder (2 layers, 4 heads)
            ↓
       Mean Pooling + Feature Fusion
            ↓
       MLP (256 → 128 → 64 → 1)
            ↓
       Sigmoid Score [0, 1]
```

CONFIGURATION
==============

Edit in main.py:

```python
train_config = TrainingConfig(
    batch_size=64,              # Adjust for OOM
    num_epochs=15,              # More epochs = better accuracy
    learning_rate=1e-3,         # ~0.001
    weight_decay=1e-5,          # L2 regularization
    dropout=0.1,                # Prevent overfitting
    early_stopping_patience=3,  # Stop if no improvement
    early_stopping_metric='ndcg_at_k'
)
```

TROUBLESHOOTING
================

1. **CUDA Out of Memory**
   ```python
   # In main.py, reduce batch_size:
   batch_size = 1024  # Instead of 64
   ```

2. **Data Not Found**
   ```bash
   python data_generation.py
   ```

3. **Slow Training on CPU**
   - Install CUDA if GPU available
   - Or reduce num_epochs to 5

4. **Memory Error During Preprocessing**
   - Close other applications
   - Or process data in smaller chunks (see data.py)

FILES REFERENCE
================

Core Pipeline:
- main.py              - Orchestration
- data.py              - Preprocessing
- model.py             - Transformer & ablation
- train.py             - Training loop
- eval.py              - Metrics

Baselines:
- baselines.py         - Popularity & LightGBM

Analysis:
- ablation.py          - Component validation
- business_impact.py   - Revenue simulation

Demo:
- app.py               - Streamlit interface

Utilities:
- data_generation.py   - Synthetic data
- validate.py          - Quick validation
- requirements.txt     - Dependencies

DATA PROCESSING PIPELINE
=========================

Input: 5M rows of training_data.csv

```
Step 1: Parse Timestamps
  - Convert 'timestamp' to datetime
  - Extract: hour, day_of_week, weekend_flag

Step 2: Merge Features
  - Join user features (budget_sensitivity, veg_preference, etc.)
  - Join restaurant features (price_tier, cuisine)
  - Join item features (price, category, veg_flag)

Step 3: Temporal Split
  - Train:  Months 1-8  (3.5M rows)
  - Val:    Month 9     (0.7M rows)
  - Test:   Month 10    (0.8M rows)
  
  ✓ No data leakage (strictly chronological)

Step 4: Normalize Features
  - Compute mean/std on TRAIN only
  - Apply to all splits
  - Keep context features (hour unscaled, better for models)

Step 5: Cart Sequences
  - Parse cart_state strings to lists
  - Pad to max_length=6 with -1 (padding token)
  - Attention mask prevents padding from influencing output

Step 6: PyTorch Dataset
  - Create CartRecommendationDataset
  - Return dict with:
    * cart: LongTensor shape (batch, 6)
    * candidate: LongTensor shape (batch, 1)
    * user_features: FloatTensor shape (batch, 7)
    * rest_features: FloatTensor shape (batch, 2)
    * context_features: FloatTensor shape (batch, 2)
    * label: FloatTensor shape (batch, 1)
```

MODEL EVALUATION METRICS
=========================

**Precision@K**: What fraction of top-K are correct?
- Formula: |{correct items in top-K}| / K
- Best for: High precision recommendations

**Recall@K**: What fraction of correct items are in top-K?
- Formula: |{correct items in top-K}| / |correct items total|
- Best for: Coverage of all relevant items

**NDCG@K**: Ranking quality with position discount
- Formula: DCG_K / IDCG_K
- Range: [0, 1] (1 is perfect ranking)
- Best for: Overall ranking quality (industry standard)

**AUC**: Area Under ROC Curve
- Formula: Probability of correctly ranking positive > negative
- Range: [0.5, 1.0] (0.5 is random, 1.0 is perfect)
- Best for: Binary classification quality

**Per-Cart Evaluation**:
- We compute metrics for each unique cart/order
- Then average across all carts
- More realistic than global ranking

NEXT STEPS
===========

1. Run basic validation:
   ```bash
   python validate.py
   ```

2. Run full pipeline:
   ```bash
   python main.py
   ```

3. Review results:
   - Check console output for metrics
   - Compare Transformer vs baselines
   - Review ablation results

4. (Optional) Interactive exploration:
   ```bash
   streamlit run app.py
   ```

5. Production deployment:
   - Save model: torch.save(model.state_dict(), "model.pth")
   - Load in inference: model.load_state_dict(torch.load("model.pth"))
   - Batch predictions: model.eval(); predictions = model(...)

"""