#CartIQ: Context-Aware Sequential Add-On Recommendation

This repository contains the full implementation, synthetic dataset generation pipeline, training framework, and evaluation results for the CSAO Rail recommendation problem.

**Project Overview**

CartIQ models the cart as a sequential decision process rather than a static item set.

Instead of relying on popularity-based heuristics, we design a two-stage system with a lightweight Transformer-based ranking model to generate real-time contextual add-on recommendations.

The system is implemented and evaluated on a large-scale synthetic dataset using a temporal hold-out split to simulate real deployment conditions.

**Dataset Overview**

→ 10,000 users  
→ 300 restaurants  
→ 2,000 unique menu items  
→ 500,000 simulated orders  
→ 10-month temporal simulation  
→ 5 metropolitan cities  

The dataset is generated using probabilistic behavioral modeling to simulate realistic meal progression, user preferences, temporal ordering patterns, and incomplete cart scenarios.

**Model Design**

Candidate Generation  
→ Historical co-order relationships  
→ Category complement rules  
→ Restaurant-level popularity  
→ Item similarity signals  

Transformer-Based Ranking  
→ Cart sequence encoding  
→ Context fusion (user, restaurant, time)  
→ Learning-to-rank objective  
→ Binary cross-entropy with negative sampling  

**Evaluation Results**

Compared against a popularity-based baseline:

| Metric        | Popularity | Transformer | % Lift |
|---------------|------------|------------|---------|
| AUC           | 0.7376     | 0.7849     | +6.41%  |
| NDCG@8        | 0.4146     | 0.4575     | +10.35% |
| Precision@8   | 0.0832     | 0.0911     | +9.49%  |
| Recall@8      | 0.7057     | 0.7287     | +3.26%  |

All results are computed on a held-out temporal test set (Month 10).

**Repository Structure**

data_generation.py  → Synthetic dataset generator  
model.py            → Transformer-based ranking model  
train.py            → Model training pipeline  
eval.py             → Evaluation and ranking metrics  
baselines.py        → Popularity baseline implementation  
business_impact.py  → Revenue lift estimation  
users.csv           → User-level features  
items.csv           → Menu item metadata  
restaurants.csv     → Restaurant metadata  
training_sample.csv → Sample ranking dataset  

**Full Dataset Access**

The complete dataset (including full training_data.csv) is available here:

https://drive.google.com/drive/folders/1WNyOLggPxHV9rodOakTl5g1_rOkwL5aA?usp=sharing

**Key Features**

→ Behavior-driven cart simulation  
→ Temporal train/validation/test split  
→ Learning-to-rank formulation  
→ Cold-start handling (user, item, restaurant)  
→ LLM-based semantic enrichment (offline embedding generation)  
→ Business impact estimation module  
