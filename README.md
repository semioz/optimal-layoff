# optimal-layoff

### HR Employee Retention Prediction

This project predicts employee retention using a real-world HR dataset. We explore the dataset through detailed EDA, apply preprocessing and feature engineering, and evaluate multiple machine learning models, both with and without dimensionality reduction.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Workflow](#workflow)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Dimensionality Reduction (PCA)](#4-dimensionality-reduction-pca)
  - [5. Model Training](#5-model-training)
  - [6. Hyperparameter Tuning](#6-hyperparameter-tuning)
  - [7. Evaluation and Comparison](#7-evaluation-and-comparison)
  - [8. Feature Importance](#8-feature-importance)
- [Insights](#insights)

## Project Overview

The dataset includes information about employees' demographics, departments, performance, compensation, and employment history. The goal is to predict employee termination using classification models trained on engineered features.

## Technologies Used

- **Python** — core language
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — data visualization
- **scikit-learn** — preprocessing, modeling, PCA, evaluation
- **TensorFlow / Keras** — neural networks
- **XGBoost** — gradient boosting classifier
- **Ray Tune** — scalable hyperparameter tuning
- **Jupyter Notebook** — for interactive development

## Workflow

### 1. Exploratory Data Analysis (EDA)

- Reviewed column types, dataset shape, and missing values
- Visualized distributions of:
  - Termination reasons
  - Department performance
  - Gender-based terminations
  - Salary by department
  - Manager termination rates

### 2. Data Preprocessing

- Filled missing values and dropped irrelevant ID fields
- Engineered `TenureInYears` and handled categorical columns
- Used LabelEncoder for ordinal and OneHotEncoder for nominal features

### 3. Feature Engineering

- Added features like `Age`
- Defined final feature set, excluding outcome-related columns
- Split dataset into train, validation, and test sets

### 4. Dimensionality Reduction (PCA)

- Applied PCA to scaled features to retain 90% variance
- Transformed datasets accordingly
- Analyzed explained variance and selected principal components

### 5. Model Training

Trained the following models with and without PCA:

- Neural Network (Keras)
- XGBoost
- Decision Tree

### 6. Hyperparameter Tuning

Used Ray Tune to search optimal parameters for each model:

- Neural Network: layers, learning rate, batch size
- Decision Tree: max depth, min samples split
- XGBoost: depth, estimators, learning rate

### 7. Evaluation and Comparison

Evaluated each model using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC

Displayed model comparisons and confusion matrices.

### 8. Feature Importance

- Neural Network: permutation importance
- Decision Tree: `.feature_importances_`
- XGBoost: built-in importance plots
- PCA: analyzed component loadings, F-values, Chi-squared, and Mutual Information

## Insights

- Termination rates differed by manager and department
- Salary, satisfaction, and tenure strongly influenced retention
- PCA helped reduce dimensionality while retaining predictive power
- XGBoost yielded the best performance overall
- Visualizations enhanced understanding of patterns in terminations
