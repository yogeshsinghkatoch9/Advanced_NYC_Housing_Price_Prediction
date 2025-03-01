# -------------------------------------------
# 1. Mount Google Drive
# -------------------------------------------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# -------------------------------------------
# 2. Install Required Packages
# -------------------------------------------
# Install packages if they are not already installed
!pip install -q xgboost lightgbm optuna

# -------------------------------------------
# 3. Import Libraries
# -------------------------------------------
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVR

from joblib import Parallel, delayed

import optuna
from optuna.samplers import TPESampler

# -------------------------------------------
# 4. Set Up Directory Paths
# -------------------------------------------
# Define the output directory in Google Drive
output_dir = '/content/drive/MyDrive/Research with Meysam/Data'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Define the path to the dataset
dataset_path = '/content/BostonHousing.csv'

# Check if the dataset exists
if not os.path.isfile(dataset_path):
    print(f"Dataset not found at {dataset_path}. Please upload 'BostonHousing.csv' to '/content/' directory.")
else:
    print(f"Dataset found at {dataset_path}.")

# -------------------------------------------
# 5. Load and Preprocess Data
# -------------------------------------------
# Load the dataset
data = pd.read_csv(dataset_path)  # Boston Housing data
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Feature Engineering: Adding Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly_feature_names = poly.get_feature_names_out(X.columns)
X_poly = pd.DataFrame(X_poly, columns=poly_feature_names)

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
print("***********************************************************************")
print("***********************************************************************")
print("Internal Global Ensemble:")

# Ensure X_scaled and y are numpy arrays
X_scaled = np.array(X_scaled)
y = np.array(y)

# Create an instance of 5 folds for cross-validation splitting
cv = KFold(n_splits=5, shuffle=True, random_state=1)
cv_splits = list(cv.split(X_scaled, y))  # Generate the indices for train/test splits

# -------------------------------------------
# 6. Define Ensemble Model Classes
# -------------------------------------------

# -------------------------------------------
# a. Global Ensemble with Bayesian Hyperparameter Tuning
# -------------------------------------------
class GlobalEnsembleModel_ASE:
    def __init__(self):
        # Define base models
        self.base_models = {
            'rf': RandomForestRegressor(random_state=42),
            'gb': GradientBoostingRegressor(random_state=42),
            'knn': KNeighborsRegressor()
        }
        self.meta_model = LinearRegression()

    def objective(self, trial, model_name, X_train, y_train):
        """Objective function for Optuna to optimize hyperparameters."""
        if model_name == 'rf':
            param = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200]),
                'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20]),
                'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5])
            }
            model = RandomForestRegressor(**param, random_state=42)
        elif model_name == 'gb':
            param = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 5)
            }
            model = GradientBoostingRegressor(**param, random_state=42)
        elif model_name == 'knn':
            param = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)  # 1 for Manhattan, 2 for Euclidean
            }
            model = KNeighborsRegressor(**param)
        else:
            raise ValueError("Unsupported model for global ensemble.")

        model.fit(X_train, y_train)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
        return score

    def cross_validate_model(self, model_name, model, X_scaled, y, cv_splits):
        fold_r2_scores = []  # List to store R2 scores for each fold
        fold_mse_scores = []  # List to store MSE scores for each fold

        # Store out-of-fold predictions for training the meta-learner
        gl_fold_predictions = np.zeros_like(y, dtype=float)  # Initialize variable for storing predictions for all data

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            # Split data
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Optimize hyperparameters using Optuna
            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(lambda trial: self.objective(trial, model_name, X_train, y_train), n_trials=20, show_progress_bar=False)

            best_params = study.best_params
            print(f"Model: {model_name} - Fold {fold_idx + 1} Best Params: {best_params}")

            # Set the best hyperparameters
            if model_name == 'rf':
                best_model = RandomForestRegressor(**best_params, random_state=42)
            elif model_name == 'gb':
                best_model = GradientBoostingRegressor(**best_params, random_state=42)
            elif model_name == 'knn':
                best_model = KNeighborsRegressor(**best_params)
            else:
                raise ValueError("Unsupported model for global ensemble.")

            # Fit the model on the training data
            best_model.fit(X_train, y_train)

            # Predict on validation set
            y_pred = best_model.predict(X_test)

            # Store predictions
            gl_fold_predictions[test_idx] = y_pred

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            fold_r2_scores.append(r2)
            fold_mse_scores.append(mse)

            print(f"Model: {model_name} - Fold {fold_idx + 1} R² score: {r2:.4f}, MSE: {mse:.4f}")

        return fold_r2_scores, gl_fold_predictions

    def train_meta_learner(self, gl_best_predictions, y, cv_splits):
        ensemble_predictions = np.zeros_like(y, dtype=float)  # Array to store ensemble predictions
        ensemble_r2_scores = []  # List to store ensemble R² scores for each fold
        ensemble_mse_scores = []  # List to store ensemble MSE scores for each fold

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            # Train meta-learner on the out-of-fold predictions from the base models
            meta_model = LinearRegression()
            meta_model.fit(gl_best_predictions[train_idx], y[train_idx])

            # Make predictions on the test set (out-of-fold)
            ensemble_pred = meta_model.predict(gl_best_predictions[test_idx])
            ensemble_predictions[test_idx] = ensemble_pred

            # Calculate metrics
            r2_ensemble_fold = r2_score(y[test_idx], ensemble_pred)
            mse_ensemble_fold = mean_squared_error(y[test_idx], ensemble_pred)

            ensemble_r2_scores.append(r2_ensemble_fold)
            ensemble_mse_scores.append(mse_ensemble_fold)

            print(f"Ensemble - Fold {fold_idx + 1} R² score: {r2_ensemble_fold:.4f}, MSE score: {mse_ensemble_fold:.4f}")

        return ensemble_predictions, ensemble_r2_scores, ensemble_mse_scores  # Return predictions and scores for all folds

    # Main method to execute the ensemble model training and evaluation
    def main(self, X_scaled, y):
        gl_best_predictions = np.zeros((X_scaled.shape[0], len(self.base_models)))
        gl_model_r2_scores = {}
        gl_model_mse_scores = {}

        # Get the predictions for each model in the ensemble
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"\nTraining base model: {name}")
            fold_r2_scores, gl_fold_predictions = self.cross_validate_model(name, model, X_scaled, y, cv_splits)
            gl_model_r2_scores[name] = fold_r2_scores  # Store R² scores
            gl_model_mse_scores[name] = [mean_squared_error(y[test_idx], gl_fold_predictions[test_idx]) for (_, test_idx) in cv_splits]
            gl_best_predictions[:, i] = gl_fold_predictions  # Store predictions for the meta-learner

        # Train meta-learner and calculate R² scores for the ensemble
        gl_predictions_ASE, ensemble_r2_scores, ensemble_mse_scores = self.train_meta_learner(gl_best_predictions, y, cv_splits)

        return ensemble_r2_scores, ensemble_mse_scores, gl_predictions_ASE  # Return MSE scores, R² scores, and predictions

# -------------------------------------------
# b. Clustered Ensemble with PCA and Hyperparameter Tuning
# -------------------------------------------
class ClusteredEnsemble:
    def __init__(self):
        self.cluster_indices = None  # To store cluster indices for all data points

    def find_optimal_clusters(self, X_train_pca, k_range=range(2, 6)):
        """Find optimal number of clusters using Silhouette and Davies-Bouldin indices."""
        silhouette_scores = []
        db_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_train_pca)
            sil_score = silhouette_score(X_train_pca, clusters)
            db_score = davies_bouldin_score(X_train_pca, clusters)
            silhouette_scores.append(sil_score)
            db_scores.append(db_score)
            print(f"k={k}: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")

        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        optimal_k_db = k_range[np.argmin(db_scores)]
        return optimal_k_silhouette, optimal_k_db

    def train_base_models(self, X_train, y_train):
        """Train the base models with hyperparameter tuning and return the best models and their predictions."""
        models = {
            'rf': RandomForestRegressor(random_state=42),
            'gb': GradientBoostingRegressor(random_state=42),
            'ab': AdaBoostRegressor(random_state=42)
        }

        param_grids = {
            'rf': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'gb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5]
            },
            'ab': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        }

        best_models = {}
        base_predictions = {}

        for name, model in models.items():
            print(f"Training base model: {name} within cluster")
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=param_grids[name],
                                       cv=3,
                                       scoring='r2',
                                       n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_models[name] = best_model
            base_predictions[name] = best_model.predict(X_train)
            print(f"Best params for {name}: {grid_search.best_params_}")

        return best_models, base_predictions

    def train_ensemble(self, base_predictions, y_train):
        """Train the linear regression ensemble using base model predictions."""
        X_ensemble = np.column_stack(list(base_predictions.values()))
        ensemble = LinearRegression()
        ensemble.fit(X_ensemble, y_train)
        return ensemble

    def predict_ensemble(self, base_models, ensemble_model, X):
        """Generate predictions using the ensemble model."""
        base_predictions = np.column_stack([
            model.predict(X) for model in base_models.values()
        ])
        return ensemble_model.predict(base_predictions)

    def clustered_ensemble_cv(self, X, y, metric='silhouette'):
        """Main function implementing the clustered ensemble with cross-validation."""
        # Initialize containers for results
        fold_results = []
        all_predictions = []
        all_true_values = []
        all_cluster_indices = np.zeros(X.shape[0], dtype=int)  # Initialize cluster indices array

        # Apply PCA to reduce dimensionality before clustering
        pca = PCA(n_components=0.95, random_state=42)  # Retain 95% variance
        X_pca = pca.fit_transform(X)

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\nProcessing Fold {fold_idx + 1} for Clustered Ensemble...")
            X_train_pca, X_val_pca = X_pca[train_idx], X_pca[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Find optimal number of clusters
            optimal_k_silhouette, optimal_k_db = self.find_optimal_clusters(X_train_pca)
            k = optimal_k_silhouette if metric == 'silhouette' else optimal_k_db
            print(f"Selected number of clusters (based on {metric}): {k}")

            # Fit KMeans on training data
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            train_clusters = kmeans.fit_predict(X_train_pca)
            val_clusters = kmeans.predict(X_val_pca)

            all_cluster_indices[train_idx] = train_clusters  # Store train cluster indices
            all_cluster_indices[val_idx] = val_clusters      # Store validation cluster indices

            # Train models for each cluster
            cluster_models = {}
            cluster_predictions = defaultdict(list)

            for cluster in range(k):
                print(f"Training models for Cluster {cluster+1}/{k}")
                # Get cluster-specific data
                cluster_mask = train_clusters == cluster
                X_train_cluster_pca = X_train_pca[cluster_mask]
                y_train_cluster = y_train[cluster_mask]

                if len(X_train_cluster_pca) > 0:
                    # Train base models and ensemble
                    base_models, base_preds = self.train_base_models(X_train_cluster_pca, y_train_cluster)
                    ensemble = self.train_ensemble(base_preds, y_train_cluster)
                    cluster_models[cluster] = (base_models, ensemble)

                    # Generate predictions for validation points in this cluster
                    val_mask = val_clusters == cluster
                    if np.any(val_mask):
                        X_val_cluster_pca = X_val_pca[val_mask]
                        y_pred_cluster = self.predict_ensemble(base_models, ensemble, X_val_cluster_pca)
                        cluster_predictions['predictions'].extend(y_pred_cluster)
                        cluster_predictions['true_values'].extend(y_val[val_mask])

            # Calculate metrics for this fold
            fold_predictions = np.array(cluster_predictions['predictions'])
            fold_true_values = np.array(cluster_predictions['true_values'])

            r2 = r2_score(fold_true_values, fold_predictions)
            mse = mean_squared_error(fold_true_values, fold_predictions)

            fold_results.append({
                'fold': fold_idx + 1,
                'r2': r2,
                'mse': mse,
                'n_clusters': k
            })

            all_predictions.extend(fold_predictions)
            all_true_values.extend(fold_true_values)

            print(f"Fold {fold_idx + 1} - R²: {r2:.4f}, MSE: {mse:.4f}, Clusters: {k}")

        # Calculate overall metrics
        overall_r2 = r2_score(all_true_values, all_predictions)
        overall_mse = mean_squared_error(all_true_values, all_predictions)
        self.cluster_indices = all_cluster_indices  # Save cluster indices

        return {
            'fold_results': fold_results,
            'overall_r2': overall_r2,
            'overall_mse': overall_mse,
            'predictions': np.array(all_predictions),
            'true_values': np.array(all_true_values),
            'cluster_indices': all_cluster_indices  # Return cluster indices
        }

# -------------------------------------------
# c. Adaptive KNN Regressor with Dynamic Neighbors
# -------------------------------------------
class AdaptiveKNNRegressor(KNeighborsRegressor):
    def __init__(self, n_neighbors=10):
        """
        Initialize the AdaptiveKNNRegressor with a single neighbor count.
        Inherits directly from KNeighborsRegressor to ensure compatibility with scikit-learn.
        """
        super().__init__(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)

# -------------------------------------------
# d. Local Ensemble with Hyperparameter Tuning
# -------------------------------------------
class LocalEnsembleModel:
    def __init__(self, total_data_points, optimal_clusters):
        self.total_data_points = total_data_points
        self.optimal_clusters = optimal_clusters
        self.models = {
            'adapt': AdaptiveKNNRegressor(),
            'svr': SVR()
        }
        self.meta_model = LinearRegression()

    def objective(self, trial, model_name, X_train, y_train):
        """Objective function for Optuna to optimize hyperparameters."""
        if model_name == 'adapt':
            # AdaptiveKNNRegressor uses a single n_neighbors parameter
            n_neighbors = trial.suggest_categorical('n_neighbors', [5, 10, 15])
            model = AdaptiveKNNRegressor(n_neighbors=n_neighbors)
        elif model_name == 'svr':
            param = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
            }
            model = SVR(**param)
        else:
            raise ValueError("Unsupported model for local ensemble.")

        model.fit(X_train, y_train)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
        return score

    def cross_validate_model(self, model_name, model, X_scaled, y, cv_splits):
        fold_r2_scores = []
        fold_mse_scores = []
        loc_fold_predictions = np.zeros(len(y), dtype=float)
        loc_fold_train_predictions = np.zeros_like(y, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            # Split data
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Optimize hyperparameters using Optuna
            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(lambda trial: self.objective(trial, model_name, X_train, y_train), n_trials=20, show_progress_bar=False)

            best_params = study.best_params
            print(f"Model: {model_name} - Fold {fold_idx + 1} Best Params: {best_params}")

            # Set the best hyperparameters
            if model_name == 'adapt':
                best_model = AdaptiveKNNRegressor(n_neighbors=best_params['n_neighbors'])
            elif model_name == 'svr':
                best_model = SVR(**best_params)
            else:
                raise ValueError("Unsupported model for local ensemble.")

            # Fit the model on training data
            best_model.fit(X_train, y_train)

            # Predict on validation set
            y_pred = best_model.predict(X_test)

            # Predict on training set (for evaluation)
            y_pred_train = best_model.predict(X_train)

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            fold_r2_scores.append(r2)
            fold_mse_scores.append(mse)

            # Store predictions
            loc_fold_predictions[test_idx] = y_pred
            loc_fold_train_predictions[train_idx] = y_pred_train

            print(f"Model: {model_name} - Fold {fold_idx + 1} R² score: {r2:.4f}, MSE: {mse:.4f}")

        return fold_r2_scores, loc_fold_predictions, loc_fold_train_predictions, fold_mse_scores

    def train_meta_learner(self, local_best_predictions, local_best_train_predictions, y, cv_splits):
        ensemble_predictions = np.zeros_like(y, dtype=float)
        ensemble_r2_scores = []
        ensemble_mse_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            # Extract the predictions from all base models for the current fold
            PR_train = local_best_train_predictions[train_idx]  # Training predictions
            PR_test = local_best_predictions[test_idx]  # Test predictions
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Train the meta-learner using only the training data (training predictions)
            meta_learner = LinearRegression()
            meta_learner.fit(PR_train, y_train)

            # Use the meta-learner to predict on the test set for the current fold
            y_pred_fold = meta_learner.predict(PR_test)
            ensemble_predictions[test_idx] = y_pred_fold  # Store test predictions

            # Compute R² and MSE for the meta-learner on this fold
            r2_ensemble_fold = r2_score(y_test, y_pred_fold)
            mse_ensemble_fold = mean_squared_error(y_test, y_pred_fold)

            # Append ensemble R² and MSE for each fold
            ensemble_r2_scores.append(r2_ensemble_fold)
            ensemble_mse_scores.append(mse_ensemble_fold)

            print(f"Ensemble Fold {fold_idx + 1} R² score: {r2_ensemble_fold:.4f}, MSE: {mse_ensemble_fold:.4f}")

        # Calculate overall performance of the ensemble
        overall_ensemble_r2 = np.mean(ensemble_r2_scores)

        return ensemble_predictions, ensemble_r2_scores, ensemble_mse_scores

    # Main method to execute the ensemble model training and evaluation
    def main(self, X_scaled, y):
        loc_best_train_predictions = np.zeros((X_scaled.shape[0], len(self.models)))
        loc_best_predictions = np.zeros((X_scaled.shape[0], len(self.models)))
        loc_model_r2_scores = {}
        loc_model_mse_scores = {}

        for i, (name, model) in enumerate(self.models.items()):
            print(f"\nCross-validating model: {name}")
            fold_r2_scores, loc_fold_predictions, loc_fold_train_predictions, fold_mse_scores = self.cross_validate_model(name, model, X_scaled, y, cv_splits)
            loc_model_r2_scores[name] = fold_r2_scores
            loc_model_mse_scores[name] = fold_mse_scores
            loc_best_train_predictions[:, i] = loc_fold_train_predictions
            loc_best_predictions[:, i] = loc_fold_predictions

        loc_predictions_ASE, ensemble_r2_scores, ensemble_mse_scores = self.train_meta_learner(
            loc_best_predictions, loc_best_train_predictions, y, cv_splits
        )

        return ensemble_mse_scores, ensemble_r2_scores, loc_predictions_ASE  # Return MSE scores, R² scores, and predictions

# -------------------------------------------
# f. External Ensemble with Linear Regression Meta-Learner and Stacking
# -------------------------------------------
class ExternalEnsemble:
    def __init__(self, X, y, cv_splits, gl_predictions_ASE, clustered_fold_predictions, loc_predictions_ASE):
        self.X = X
        self.y = y
        self.cv_splits = cv_splits
        self.gl_best_predictions = gl_predictions_ASE
        self.clustered_fold_predictions = clustered_fold_predictions
        self.local_best_predictions = loc_predictions_ASE

    def train_meta_learner(self, X_train, y_train):
        """Train the meta-learner with linear regression."""
        meta_learner = LinearRegression()
        meta_learner.fit(X_train, y_train)
        return meta_learner

    def main(self):
        r2_scores = []
        mse_scores = []
        all_predictions = np.zeros_like(self.y, dtype=float)  # Store predictions for all folds

        for fold_idx, (train_index, test_index) in enumerate(self.cv_splits):
            print(f"\nProcessing Fold {fold_idx + 1} for External Ensemble...")
            # Stack predictions from Global, Clustered, and Local Ensembles
            X_train_fold = np.column_stack((
                self.gl_best_predictions[train_index],
                self.clustered_fold_predictions[train_index],
                self.local_best_predictions[train_index]
            ))
            y_train_fold = self.y[train_index]

            X_test_fold = np.column_stack((
                self.gl_best_predictions[test_index],
                self.clustered_fold_predictions[test_index],
                self.local_best_predictions[test_index]
            ))
            y_test_fold = self.y[test_index]

            # Train the meta-learner
            meta_learner = self.train_meta_learner(X_train_fold, y_train_fold)

            # Predict on test fold
            y_pred_fold = meta_learner.predict(X_test_fold)

            # Calculate metrics
            r2 = r2_score(y_test_fold, y_pred_fold)
            mse = mean_squared_error(y_test_fold, y_pred_fold)

            r2_scores.append(r2)
            mse_scores.append(mse)

            # Store predictions
            all_predictions[test_index] = y_pred_fold

            print(f"Fold {fold_idx + 1} - External Ensemble R²: {r2:.4f}, MSE: {mse:.4f}")

        # Calculate overall metrics
        overall_r2 = r2_score(self.y, all_predictions)
        overall_mse = mean_squared_error(self.y, all_predictions)

        return r2_scores, mse_scores, all_predictions, overall_r2, overall_mse

# -------------------------------------------
# 7. Train Ensemble Models
# -------------------------------------------

# -------------------------------------------
# a. Train Global Ensemble
# -------------------------------------------
# Instantiate and train the Global Ensemble Model
global_ensemble = GlobalEnsembleModel_ASE()
ensemble_r2_scores, ensemble_mse_scores, gl_predictions_ASE = global_ensemble.main(X_scaled, y)
gl_r2 = np.mean(ensemble_r2_scores)  # Global ensemble R² score
gl_mse = np.mean(ensemble_mse_scores)  # Global ensemble MSE score

print("***********************************************************************")
print("***********************************************************************")
print("Internal Global Ensemble Completed.")

# -------------------------------------------
# b. Train Clustered Ensemble
# -------------------------------------------
# Instantiate the ClusteredEnsemble class
clustered_ensemble = ClusteredEnsemble()

# Run the clustered ensemble cross-validation with both metrics
print("\nRunning Clustered Ensemble with Silhouette Metric...")
silhouette_results = clustered_ensemble.clustered_ensemble_cv(X_scaled, y, metric='silhouette')

print("\nRunning Clustered Ensemble with Davies-Bouldin Metric...")
db_results = clustered_ensemble.clustered_ensemble_cv(X_scaled, y, metric='db')

# Compare and select the best metric
silhouette_r2 = silhouette_results['overall_r2']
db_r2 = db_results['overall_r2']
best_metric = 'silhouette' if silhouette_r2 > db_r2 else 'db'
best_results = silhouette_results if silhouette_r2 > db_r2 else db_results

# Store the best predictions globally
clustered_fold_predictions = best_results['predictions']

# Print results
print(f"\nBest clustering metric: {best_metric}")
print(f"Overall R² score: {best_results['overall_r2']:.4f}")
print(f"Overall MSE: {best_results['overall_mse']:.4f}")
print("\nFold-wise results:")
for fold_result in best_results['fold_results']:
    print(f"Fold {fold_result['fold']}: R² = {fold_result['r2']:.4f}, "
          f"MSE = {fold_result['mse']:.4f}, "
          f"Number of clusters = {fold_result['n_clusters']}")

print("***********************************************************************")
print("***********************************************************************")
print("Internal Clustered Ensemble Completed.")

# -------------------------------------------
# c. Train Local Ensemble
# -------------------------------------------
# Usage
total_data_points = X_scaled.shape[0]

# Since ClusteredEnsemble selected k=2 in all folds, set optimal_clusters=2
optimal_clusters = 2

# Instantiate and train the Local Ensemble Model
local_ensemble_model = LocalEnsembleModel(total_data_points, optimal_clusters)
ensemble_mse_scores_local, ensemble_r2_scores_local, loc_predictions_ASE = local_ensemble_model.main(X_scaled, y)
loc_r2 = np.mean(ensemble_r2_scores_local)  # Local ensemble R² score
loc_mse = np.mean(ensemble_mse_scores_local)  # Local ensemble MSE score

print("***********************************************************************")
print("***********************************************************************")
print("Internal Local Ensemble Completed.")

# -------------------------------------------
# d. Train External Ensemble
# -------------------------------------------
# External Ensemble Predictions and R², MSE Calculation
external_ensemble = ExternalEnsemble(
    X_scaled, y, cv_splits, gl_predictions_ASE, clustered_fold_predictions, loc_predictions_ASE
)

external_r2_scores, external_mse_scores, external_all_predictions, overall_external_r2, overall_external_mse = external_ensemble.main()

# -------------------------------------------
# 8. Save Results to Google Drive
# -------------------------------------------
# Assign fold numbers based on cv_splits
fold_numbers = np.zeros(len(data), dtype=int)  # Initialize array for fold numbers

for fold_idx, (train_index, test_index) in enumerate(cv_splits):
    fold_numbers[test_index] = fold_idx + 1  # Set fold number for each test set

# Add the fold numbers to the dataset
data['Fold_Number'] = fold_numbers
data['GlobalEnsemble_Predictions'] = gl_predictions_ASE
data['ClusteredEnsemble_Predictions'] = clustered_fold_predictions
data['LocalEnsemble_Predictions'] = loc_predictions_ASE
data['ExternalEnsemble_Predictions'] = external_all_predictions
# Store cluster number for each data point
data['Cluster_Indices'] = clustered_ensemble.cluster_indices

# Save the result to an Excel file
output_file_path = os.path.join(output_dir, 'Ensemble_Predictions_BostonHousing.xlsx')
data.to_excel(output_file_path, index=False)
print(f"\nEnsemble predictions saved to {output_file_path}")

# -------------------------------------------
# 9. Display External Ensemble Results
# -------------------------------------------
# Print External R² and MSE
print("\nExternal Ensemble Performance:")
for i in range(5):
    print(f"Fold {i+1}: R² = {external_r2_scores[i]:.4f}, MSE = {external_mse_scores[i]:.4f}")

print(f"\nOverall External Ensemble R²: {overall_external_r2:.4f}")
print(f"Overall External Ensemble MSE: {overall_external_mse:.4f}")
print("\nEnsemble modeling completed successfully!")

# -------------------------------------------
# 10. Additional Performance Monitoring (Optional)
# -------------------------------------------
# Plotting Predicted vs Actual for External Ensemble
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=external_all_predictions, hue=fold_numbers, palette='viridis', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('External Ensemble: Actual vs Predicted MEDV')
plt.legend(title='Fold')
plt.show()

# Plot Residuals for External Ensemble
residuals = y - external_all_predictions
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color='blue')
plt.xlabel('Residuals')
plt.title('External Ensemble Residuals Distribution')
plt.show()

# Display Correlation Heatmap of Predictions
predictions_df = pd.DataFrame({
    'Actual': y,
    'GlobalEnsemble': gl_predictions_ASE,
    'ClusteredEnsemble': clustered_fold_predictions,
    'LocalEnsemble': loc_predictions_ASE,
    'ExternalEnsemble': external_all_predictions
})

plt.figure(figsize=(10, 8))
sns.heatmap(predictions_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Predictions')
plt.show()

# -------------------------------------------
# 11. Summary of Results
# -------------------------------------------
print("\nSummary of Ensemble Performances:")
print(f"Global Ensemble - Average R²: {gl_r2:.4f}, Average MSE: {gl_mse:.4f}")
print(f"Clustered Ensemble - Overall R²: {best_results['overall_r2']:.4f}, Overall MSE: {best_results['overall_mse']:.4f}")
print(f"Local Ensemble - Average R²: {loc_r2:.4f}, Average MSE: {loc_mse:.4f}")
print(f"External Ensemble - Overall R²: {overall_external_r2:.4f}, Overall MSE: {overall_external_mse:.4f}")