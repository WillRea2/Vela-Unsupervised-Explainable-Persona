# Vela-Unsupervised-Explainable-Person
LLM-Explainers: Explaining Black Box Unsupervised Clustering Models via LLMs

## Overview
This repository provides a framework for unsupervised clustering and explainable analysis of founder personas using machine learning techniques. The goal is to create a system that clusters data using black box models and then provides interpretable insights using 'LLM-Explainers'.

### Key Features
- **Data Cleaning and Preprocessing**:
  Handles missing values, categorical variables, and scaling.
- **Latent Encoding**:
  Uses VAE to create compact representations of high-dimensional data.
- **Clustering Tests**:
  Implements tests for clustering methods like K-Means, Spectral Clustering, Hierarchical Clustering, GMM, K-Medoids, and Fuzzy C-Means.
- **LLM-Driven Explanation**:
  Generates concise and detailed cluster-level and subcluster-level explanations using GPT-based models.
- **Prediction**:
  Classifies new data points into clusters and subclusters with probabilistic explanations.
---

## Dependencies

Ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `torch` (PyTorch)
- `scikit-learn`
- `skfuzzy`
- `sklearn-extra`
- `openai`
- `pydantic`
---

## Usage

### 1. Data Preparation
Prepare your dataset in CSV format. Use the `clean()` function to preprocess the data.

```python
from preprocessing import clean
data_unclean = pd.read_csv('path_to_your_file.csv')
data = clean(data_unclean)
