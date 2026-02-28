# CartIQ: Context-Aware Sequential Add-On Recommendation

This repository contains the synthetic dataset generation pipeline and architecture design for the CSAO Rail problem.

**Project Overview**

CartIQ models the cart as a sequential decision process rather than a static item set.
A lightweight Transformer-based ranking model is proposed to generate real-time contextual add-on recommendations.

**Repository Structure**

data_generation.py → Synthetic dataset generator
training_sample.csv → Sample ranking dataset
users.csv → User-level features
items.csv → Menu item metadata
restaurants.csv → Restaurant metadata

**Full Dataset Access**

The complete dataset (including full training_data.csv) is available here:

https://drive.google.com/drive/folders/1WNyOLggPxHV9rodOakTl5g1_rOkwL5aA?usp=sharing

**Key Features**

→ Behavior-driven cart simulation
→ Temporal split (10-month simulation)
→ Learning-to-rank formulation
→ Cold-start handling
→ LLM-based semantic enrichment (offline)
