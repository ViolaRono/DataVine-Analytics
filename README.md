# DataVine Analytics — Summative Lab

## Overview
This notebook implements three prototype machine learning solutions for DataVine 
Analytics' client projects, covering classification, recommendation systems, and 
clustering. Each section follows the standard data science workflow: data preparation, 
dimensionality reduction (PCA), model implementation with hyperparameter tuning, and 
evaluation/interpretation.

## Tools & Libraries
- Python (Jupyter Notebook)
- pandas, numpy
- scikit-learn (preprocessing, decomposition, neighbors, cluster, mixture, metrics, 
  feature_selection)
- matplotlib, seaborn
- rdatasets (for Chickwts and USArrests datasets)


## Notebook Structure

### Step 1: Load & Prepare Datasets
- All three datasets loaded and inspected for missing values (none found in any dataset).
- Dataset shapes and summaries printed to confirm structure before proceeding.

### Step 2: Wine Data Set k-NN Classification
- Data split into train/test (80/20) *before* scaling and PCA, to avoid data leakage.
- Features standardized; PCA retained 95% variance using *10 principal components*.
- GridSearchCV tuned k (1–20) and distance metric across euclidean, manhattan, minkowski.
- *Best parameters:* k=18, euclidean distance, 97.9% CV accuracy.
- *Test set accuracy: 100%*, with perfect precision/recall/f1 across all 3 wine classes.

### Step 3: Chickwts Recommendation System
- Chicken weights standardized; PCA reduced to 1 principal component.
- Cosine similarity computed across all 71 individual chicken records.
- recommend_feeds() function returns the feed types of the most similar chickens by 
  weight for a given feed.
- *Limitation noted:* comparisons are done at the individual-chicken level (not 
  averaged per feed type), so results can vary depending on which specific chicken 
  represents a feed type.

### Step 4: USArrests Dataset — Clustering (K-Means and GMM)
- Standardized Murder, Assault, UrbanPop, Rape; selected Murder, Assault, and Rape as 
  the 3 most relevant features (violent crime rates), dropping UrbanPop (a demographic, 
  not crime, feature).
- PCA reduced to 2 components, explaining *93.9% of variance*.
- *K-Means:* optimal k=2, determined via visual inspection of the elbow curve (not 
  just minimum inertia, which trends toward the highest k tested).
- *GMM:* optimal k=2, determined via lowest BIC score.
- Both methods agree on cluster assignments for most states; PC1 (driven by violent 
  crime rates) appears to separate lower-crime from higher-crime states.

## Notes on Methodology
- Train/test splits are always performed before scaling/PCA to prevent data leakage.
- Feature selection and cluster-count decisions are explained with reasoning in-notebook, 
  rather than applied as unexplained defaults.
- Limitations (e.g., 1D PCA compressing magnitude information in the recommendation 
  system) are called out explicitly rather than glossed over.

## How to Run
1. Ensure rdatasets is installed: run !pip install rdatasets in a notebook cell 
   (note: pip install rdatasets without the ! will raise a SyntaxError in Jupyter).
2. Run cells sequentially from top to bottom.
3. No external files are required — Wine loads from scikit-learn, Chickwts and 
   USArrests load from rdatasets.