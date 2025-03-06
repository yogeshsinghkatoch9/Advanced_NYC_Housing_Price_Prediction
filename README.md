# Skyline Ensemble: Advanced NYC Housing Price Prediction

Skyline Ensemble is a robust, state-of-the-art ensemble learning framework designed to predict housing prices in New York City. By integrating global stacking, clustering, and local modeling strategies, this project leverages advanced hyperparameter tuning, dimensionality reduction, and stacking techniques to achieve superior prediction performance on the NYC Housing dataset.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Data Preparation](#data-preparation)
- [Ensemble Modeling Framework](#ensemble-modeling-framework)
  - [Global Ensemble](#1-global-ensemble-with-bayesian-hyperparameter-tuning)
  - [Clustered Ensemble](#2-clustered-ensemble-with-pca-and-hyperparameter-tuning)
  - [Local Ensemble](#3-local-ensemble-with-hyperparameter-tuning)
  - [External Ensemble](#4-external-ensemble-with-linear-regression-meta-learner-and-stacking)
- [Model Evaluation and Visualization](#model-evaluation-and-visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

The Skyline Ensemble project addresses the challenges of NYC housing price prediction through an innovative blend of ensemble techniques. The framework is designed to capture both global trends and local nuances within the data by employing multiple modeling strategies. Each ensemble method is rigorously optimized, and their predictions are combined to produce a final, highly accurate forecast.

---

## Features

- **Advanced Ensemble Methods:** Combines global, clustered, and local ensemble strategies.
- **Bayesian Hyperparameter Tuning:** Utilizes Optuna for efficient hyperparameter optimization.
- **Dimensionality Reduction and Clustering:** Applies PCA and KMeans clustering to partition data and capture heterogeneity.
- **Stacked Model Architecture:** Leverages stacking with a Linear Regression meta-learner to integrate predictions.
- **Comprehensive Evaluation:** Provides detailed model evaluation metrics and visualization tools.

---

## Repository Structure

```plaintext
skyline-ensemble/
├── data/
│   └── NYCHousing.csv              # NYC Housing dataset
├── notebooks/
│   └── ensemble_modeling.ipynb     # Jupyter Notebook for interactive exploration
├── results/
│   └── Ensemble_Predictions_NYCHousing.xlsx  # Output file containing ensemble predictions and metrics
├── README.md                       # Project documentation
└── requirements.txt                # Python package dependencies

Installation and Setup

Prerequisites
	•	Python 3.7 or higher
	•	Google Colab or a local Jupyter Notebook environment

Installation Steps
	1.	Clone the Repository:

git clone https://github.com/yogeshsinghkatoch9/skyline-ensemble.git
cd skyline-ensemble


	2.	Install Required Packages:
Install the necessary dependencies via pip:

pip install -r requirements.txt

The key packages include:
	•	numpy
	•	pandas
	•	seaborn
	•	matplotlib
	•	scikit-learn
	•	xgboost
	•	lightgbm
	•	optuna
	•	joblib

Google Drive Integration (For Colab Users)

If you are using Google Colab, mount your Google Drive to save output files by executing:

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

Ensure you update the output directory path in the code accordingly.

Data Preparation

The project utilizes the NYC Housing dataset. Ensure that the NYCHousing.csv file is placed in the data/ directory, or adjust the file path in the source code as needed.

Preprocessing Steps:
	•	Feature Engineering:
Polynomial feature expansion (degree = 2) is applied to capture interaction effects among features.
	•	Scaling:
Standard scaling is performed to normalize the expanded feature set.
	•	Cross-Validation:
The dataset is split into features (X) and target (y) and evaluated using 5-fold cross-validation to ensure robust performance estimates.

Ensemble Modeling Framework

1. Global Ensemble with Bayesian Hyperparameter Tuning
	•	Base Models:
	•	RandomForestRegressor
	•	GradientBoostingRegressor
	•	KNeighborsRegressor
	•	Hyperparameter Tuning:
Employs Optuna with TPESampler to optimize model parameters on a per-fold basis.
	•	Meta-Learner:
A Linear Regression model is used to stack the out-of-fold predictions from the base models.
	•	Workflow:
Each base model is optimized and trained using 5-fold cross-validation. Their predictions are then aggregated and used to train a meta-learner that produces the final global ensemble output.

2. Clustered Ensemble with PCA and Hyperparameter Tuning
	•	Dimensionality Reduction:
PCA reduces the feature space while retaining 95% of the variance.
	•	Clustering:
KMeans clustering partitions the PCA-transformed data. The optimal number of clusters is determined using Silhouette and Davies-Bouldin indices.
	•	Cluster-Specific Modeling:
Within each cluster, base models (Random Forest, Gradient Boosting, AdaBoost) are trained using GridSearchCV and integrated via a Linear Regression meta-learner.
	•	Workflow:
Data is clustered per fold, models are trained within each cluster, and their predictions are combined to assess overall performance.

3. Local Ensemble with Hyperparameter Tuning
	•	Base Models:
	•	Adaptive KNN Regressor (customized for dynamic neighbor selection)
	•	Support Vector Regressor (SVR)
	•	Hyperparameter Tuning:
Utilizes Optuna to fine-tune model parameters.
	•	Meta-Learner:
A Linear Regression model stacks predictions derived from both training and validation sets to improve accuracy.
	•	Workflow:
The local ensemble captures finer-grained patterns in the data, complementing the global modeling approach.

4. External Ensemble with Linear Regression Meta-Learner and Stacking
	•	Stacking Approach:
Aggregates predictions from the Global, Clustered, and Local ensembles.
	•	Meta-Learner:
A Linear Regression model is used to combine the stacked predictions into a final, refined output.
	•	Workflow:
For each fold, predictions from all ensemble strategies are consolidated and fed into the meta-learner, enhancing the overall predictive performance.

Model Evaluation and Visualization

Evaluation Metrics:
	•	R² (Coefficient of Determination):
Measures the proportion of variance in the target variable explained by the model.
	•	MSE (Mean Squared Error):
Quantifies the average squared difference between actual and predicted values.

Visualizations:
	•	Scatter Plots:
Compare actual vs. predicted housing prices.
	•	Residual Analysis:
Examine the distribution of residuals using histograms and KDE plots.
	•	Correlation Heatmap:
Illustrates the inter-relationships among predictions from different ensemble methods.

Usage
	1.	Data Setup:
Place NYCHousing.csv in the data/ directory or update the file path in the code.
	2.	Execute the Pipeline:
Run the provided Jupyter Notebook (ensemble_modeling.ipynb) or Python script. The pipeline will:
	•	Mount Google Drive (if applicable)
	•	Preprocess the data and generate polynomial features
	•	Train the Global, Clustered, and Local ensembles using cross-validation
	•	Stack ensemble predictions to generate the final output
	•	Save predictions and evaluation metrics to an Excel file in the results/ directory
	3.	Analyze the Results:
Review the console output for detailed performance metrics and use the generated visualizations to further analyze model behavior.

Contributing

Contributions are highly appreciated! Please fork this repository and submit your pull requests. For any issues or feature requests, kindly open an issue in the repository.

License

This project is licensed under the MIT License. See the LICENSE file for further details.

Acknowledgements
	•	Optuna: For providing a powerful framework for hyperparameter optimization.
	•	Scikit-Learn: For an extensive suite of machine learning tools.
	•	NYC Open Data: For offering the NYC Housing dataset.
	•	Google Colab: For enabling a seamless and accessible experimental environment.

Skyline Ensemble represents an advanced approach to NYC housing price prediction, combining diverse ensemble techniques to achieve exceptional performance. Enjoy exploring and enhancing this project!

