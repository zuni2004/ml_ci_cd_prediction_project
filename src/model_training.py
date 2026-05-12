"""
CI/CD BUILD FAILURE PREDICTOR - MODEL TRAINING PIPELINE

This module trains and evaluates 3 classification models:
    1. Logistic Regression (baseline, very fast)
    2. Decision Trees (interpretable, fast)
    3. Random Forest (ensemble, good balance of speed & performance)

Pipeline Steps:
    1. Load preprocessed train/test data
    2. Train 3 models with default parameters using k-fold cross-validation
    3. Evaluate using multiple metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
    4. Select the best performing model
    5. Hyperparameter tuning using Optuna (Bayesian optimization)
    6. Final evaluation and logging
    7. Save trained model + logs as pickle files

Why These 3 Models:
    ✓ Logistic Regression
        - Very fast (O(n*m) complexity)
        - Probabilistic baseline suitable for imbalanced data
        - Interpretable coefficients
        - Best for quick initial assessment
        
    ✓ Decision Trees
        - Fast training and inference (tree depth limited)
        - Handles non-linear relationships
        - No feature scaling required
        - Interpretable decision rules
        - Can capture interactions naturally
        
    ✓ Random Forest
        - Ensemble method: reduces overfitting vs single tree
        - Fast on modern hardware (parallelizable)
        - Good out-of-box performance
        - Feature importance analysis possible
        - Handles mixed feature types well
        
    ✗ Rejected options (slow on 1.5 GB data):
        - SVM: O(n²) or O(n³) complexity, infeasible for 3.7M rows
        - KNN: Must compute distances to all training samples at inference
        - Neural Networks (CNN/RNN): Overkill for tabular data, long training
        - Clustering/PCA: Unsupervised/dimensionality reduction, not classifiers

Data Handling:
    - Load: data/preprocessed/train_processed.csv (2.96M rows × 45 cols)
            data/preprocessed/test_processed.csv (740K rows × 45 cols)
    - Target: 'build_successful' (binary: 0=fail, 1=pass)
    - No additional preprocessing needed (already normalized)

Output:
    - src/saved_models/best_model.pkl : Trained model
    - src/saved_models/training_logs.pkl : Metrics, cross-val scores, history
    - src/saved_models/hyperparameters.pkl : Best hyperparameters found
    - Console: Detailed progress and metrics

Author: ML Pipeline
Date: 2024
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

# ════════════════════════════════════════════════════════════════════════════
# SKLEARN MODELS & UTILITIES
# ════════════════════════════════════════════════════════════════════════════
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from joblib import parallel_backend

# ════════════════════════════════════════════════════════════════════════════
# OPTUNA - BAYESIAN HYPERPARAMETER OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "train_file": "src/train_processed.csv",
    "test_file": "src/test_processed.csv",
    "target_col": "build_successful",
    "test_size": 0.20,
    "random_seed": 42,
    "k_folds": 5,  # 5-fold cross-validation
    "n_jobs": -1,  # Use all CPU cores
    "verbose": 1,
}

OUTPUT_DIR = "src/saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scoring metrics for cross-validation
SCORING_METRICS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════


def print_header(title: str, char: str = "=") -> None:
    """
    Print a formatted header for console output.
    
    Args:
        title (str): Header text
        char (str): Character to use for border (default: '=')
    """
    width = 80
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def print_subheader(title: str) -> None:
    """Print a subheader with dashes."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def log_message(message: str, level: str = "INFO") -> None:
    """
    Log a timestamped message.
    
    Args:
        message (str): Message to log
        level (str): Log level (INFO, SUCCESS, WARNING, ERROR)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix_map = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✓ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌ ",
    }
    prefix = prefix_map.get(level, "ℹ️ ")
    print(f"{prefix} [{timestamp}] {message}")


def load_data(
    train_path: str, test_path: str, target_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed train and test data.
    
    Args:
        train_path (str): Path to training CSV
        test_path (str): Path to test CSV
        target_col (str): Name of target column
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
        
    Raises:
        FileNotFoundError: If CSV files don't exist
        KeyError: If target column not in data
    """
    print_subheader("LOADING PREPROCESSED DATA")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    log_message(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    
    log_message(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Separate features and target
    if target_col not in train_df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in training data. "
            f"Available columns: {list(train_df.columns)}"
        )
    
    y_train = train_df[target_col].values
    X_train = train_df.drop(columns=[target_col]).values
    
    y_test = test_df[target_col].values
    X_test = test_df.drop(columns=[target_col]).values
    
    log_message(f"✓ Loaded training data: {X_train.shape[0]:,} rows × {X_train.shape[1]} features", "SUCCESS")
    log_message(f"✓ Loaded test data: {X_test.shape[0]:,} rows × {X_test.shape[1]} features", "SUCCESS")
    log_message(f"✓ Target distribution (train): {np.sum(y_train==1):,} passed, {np.sum(y_train==0):,} failed", "SUCCESS")
    log_message(f"✓ Target distribution (test): {np.sum(y_test==1):,} passed, {np.sum(y_test==0):,} failed", "SUCCESS")
    
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS WITH DEFAULT PARAMETERS
# ════════════════════════════════════════════════════════════════════════════


def get_baseline_models() -> Dict[str, Any]:
    """
    Create baseline models with default parameters.
    
    Returns:
        dict: Model name -> sklearn model object
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            random_state=CONFIG["random_seed"],
            verbose=0,
            # Default: C=1.0 (inverse regularization strength)
            # Default: solver='lbfgs' (but 'saga' is better for large data)
            solver="saga",  # SAG solver handles large datasets efficiently
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=CONFIG["random_seed"],
            # Default max_depth=None (unlimited)
            # Default min_samples_split=2
            # Default min_samples_leaf=1
            # n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,  # Default number of trees
            random_state=CONFIG["random_seed"],
            n_jobs=-1,
            verbose=0,
            # Default max_depth=None
            # Default min_samples_split=2
            # Default min_samples_leaf=1
        ),
    }
    return models


# ════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION & EVALUATION
# ════════════════════════════════════════════════════════════════════════════


def evaluate_model_with_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    k_folds: int = 3, #pehly 5 thy ab 3 kiay
) -> Dict[str, Any]:
    """
    Evaluate model using k-fold stratified cross-validation.
    
    Args:
        model: Sklearn model object
        X (np.ndarray): Features
        y (np.ndarray): Target
        model_name (str): Model name for logging
        k_folds (int): Number of cross-validation folds
        
    Returns:
        dict: Cross-validation metrics and trained model
    """
    log_message(f"Starting {k_folds}-fold cross-validation for {model_name}...", "INFO")
    
    # Stratified k-fold to maintain class distribution
    cv = StratifiedKFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=CONFIG["random_seed"],
    )
    
    # Cross-validate with multiple metrics
    try:
        with parallel_backend("threading", n_jobs=CONFIG["n_jobs"]):
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=SCORING_METRICS,
                return_train_score=True,
                n_jobs=CONFIG["n_jobs"],
                verbose=0,
            )
    except Exception as e:
        log_message(f"Error during cross-validation: {str(e)}", "ERROR")
        raise
    
    # Aggregate results
    results = {
        "model_name": model_name,
        "cv_folds": k_folds,
        "cv_results_raw": cv_results,
    }
    
    for metric in SCORING_METRICS.keys():
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"
        
        if test_key in cv_results:
            results[f"test_{metric}_mean"] = cv_results[test_key].mean()
            results[f"test_{metric}_std"] = cv_results[test_key].std()
            results[f"train_{metric}_mean"] = cv_results[train_key].mean()
            results[f"train_{metric}_std"] = cv_results[train_key].std()
    
    log_message(
        f"✓ {model_name} CV Complete | "
        f"Mean F1: {results['test_f1_mean']:.4f} ± {results['test_f1_std']:.4f}",
        "SUCCESS",
    )
    
    return results


def print_cv_results(results_list: List[Dict[str, Any]]) -> None:
    """
    Print cross-validation results for all models.
    
    Args:
        results_list (list): List of results dicts from evaluate_model_with_cv()
    """
    print_subheader("BASELINE MODELS - CROSS-VALIDATION RESULTS")
    
    print("\n{:<20} {:<12} {:<12} {:<12} {:<12}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1-Score"
    ))
    print("-" * 80)
    
    for results in results_list:
        model_name = results["model_name"]
        accuracy = results["test_accuracy_mean"]
        precision = results["test_precision_mean"]
        recall = results["test_recall_mean"]
        f1 = results["test_f1_mean"]
        
        print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            model_name, accuracy, precision, recall, f1
        ))
    
    print("\nDetailed CV Scores (mean ± std):")
    print("-" * 80)
    
    for results in results_list:
        model_name = results["model_name"]
        print(f"\n{model_name}:")
        
        for metric in SCORING_METRICS.keys():
            mean_val = results.get(f"test_{metric}_mean", 0)
            std_val = results.get(f"test_{metric}_std", 0)
            print(f"  • {metric.capitalize():<12}: {mean_val:.4f} ± {std_val:.4f}")


def select_best_model(
    results_list: List[Dict[str, Any]],
    metric: str = "f1",
) -> Tuple[str, Dict[str, Any]]:
    """
    Select best model based on a metric.
    
    Args:
        results_list (list): List of results dicts
        metric (str): Metric to use for selection (default: 'f1')
        
    Returns:
        tuple: (best_model_name, best_results_dict)
    """
    best_idx = 0
    best_score = -1
    
    for idx, results in enumerate(results_list):
        score = results.get(f"test_{metric}_mean", -1)
        if score > best_score:
            best_score = score
            best_idx = idx
    
    best_model_name = results_list[best_idx]["model_name"]
    
    log_message(
        f"Best model: {best_model_name} (F1: {best_score:.4f})",
        "SUCCESS",
    )
    
    return best_model_name, results_list[best_idx]


# ════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING WITH OPTUNA
# ════════════════════════════════════════════════════════════════════════════


def create_optuna_study(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 30, #pehly 50 thy ab 30 kiay
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform Bayesian hyperparameter optimization using Optuna.
    
    Args:
        model_name (str): Name of model ('Logistic Regression', 'Decision Tree', 'Random Forest')
        X_train, y_train, X_test, y_test: Training and test data
        n_trials (int): Number of Optuna trials (default: 50)
        
    Returns:
        tuple: (best_params, study_history)
    """
    print_subheader(f"HYPERPARAMETER TUNING: {model_name} (Optuna - {n_trials} trials)")
    
    # Define objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function to minimize (1 - F1 score).
        Optuna minimizes, so we return (1 - score).
        
        Args:
            trial: Optuna trial object for suggesting hyperparameters
            
        Returns:
            float: Validation F1 score (higher is better)
        """
        
        if model_name == "Logistic Regression":
            # Hyperparameters to tune
            C = trial.suggest_float("C", 0.001, 10.0, log=True)
            max_iter = trial.suggest_int("max_iter", 500, 2000, step=100)
            
            model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=CONFIG["random_seed"],
                n_jobs=-1,
                solver="saga",
            )
        
        elif model_name == "Decision Tree":
            # Hyperparameters to tune
            max_depth = trial.suggest_int("max_depth", 3, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=CONFIG["random_seed"],
                # n_jobs=-1,
            )
        
        elif model_name == "Random Forest":
            # Hyperparameters to tune
            n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
            max_depth = trial.suggest_int("max_depth", 5, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=CONFIG["random_seed"],
                n_jobs=-1,
            )
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Return F1 score (Optuna will maximize this via the direction)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        return f1
    
    # Create Optuna study with TPE sampler (Tree-structured Parzen Estimator)
    sampler = TPESampler(seed=CONFIG["random_seed"])
    pruner = MedianPruner()
    
    study = optuna.create_study(
        direction="maximize",  # We want to maximize F1
        sampler=sampler,
        pruner=pruner,
    )
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Optimize
    log_message(f"Running {n_trials} Optuna trials for {model_name}...")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_value = study.best_value
    
    log_message(
        f"✓ Optuna Complete | Best F1: {best_value:.4f}",
        "SUCCESS",
    )
    
    # Log best hyperparameters
    print("\nBest Hyperparameters:")
    print("-" * 80)
    for param_name, param_value in best_params.items():
        print(f"  • {param_name:<25}: {param_value}")
    
    # Store study history
    study_history = {
        "model_name": model_name,
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": len(study.trials),
        "trials_history": [
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
            }
            for trial in study.trials
        ],
    }
    
    return best_params, study_history


def train_final_model(
    model_name: str,
    best_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """
    Train final model with best hyperparameters.
    
    Args:
        model_name (str): Model name
        best_params (dict): Best hyperparameters from Optuna
        X_train, y_train: Training data
        
    Returns:
        Trained model object
    """
    log_message(f"Training final {model_name} with best hyperparameters...")
    
    if model_name == "Logistic Regression":
        model = LogisticRegression(
            **best_params,
            random_state=CONFIG["random_seed"],
            n_jobs=-1,
            solver="saga",
        )
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            **best_params,
            random_state=CONFIG["random_seed"],
            # n_jobs=-1,
        )
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            **best_params,
            random_state=CONFIG["random_seed"],
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.fit(X_train, y_train)
    log_message(f"✓ Final model trained successfully", "SUCCESS")
    
    return model


# ════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION
# ════════════════════════════════════════════════════════════════════════════


def evaluate_final_model(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate final model on test set.
    
    Args:
        model: Trained model
        model_name (str): Model name
        X_test, y_test: Test data
        
    Returns:
        dict: Test metrics
    """
    print_subheader(f"FINAL EVALUATION: {model_name}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["confusion_matrix"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    
    # Detailed classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["classification_report"] = report
    
    # Print results
    print("\nTest Set Metrics:")
    print("-" * 80)
    for metric_name, metric_value in metrics.items():
        if metric_name not in ["confusion_matrix", "classification_report"]:
            print(f"  • {metric_name.capitalize():<15}: {metric_value:.4f}")
    
    print("\nConfusion Matrix:")
    print("-" * 80)
    print(f"  • True Negatives (Correctly Predicted Failures)  : {tn:,}")
    print(f"  • False Positives (Incorrectly Predicted Passes) : {fp:,}")
    print(f"  • False Negatives (Missed Failures)               : {fn:,}")
    print(f"  • True Positives (Correctly Predicted Passes)    : {tp:,}")
    
    print("\nDetailed Classification Report:")
    print("-" * 80)
    print(classification_report(y_test, y_pred, target_names=["Failed", "Passed"]))
    
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ════════════════════════════════════════════════════════════════════════════


def save_results(
    model: Any,
    model_name: str,
    best_params: Dict[str, Any],
    cv_results: Dict[str, Any],
    final_metrics: Dict[str, Any],
    optuna_history: Dict[str, Any],
) -> None:
    """
    Save trained model, hyperparameters, and logs to pickle files.
    
    Args:
        model: Trained model object
        model_name (str): Model name
        best_params (dict): Best hyperparameters
        cv_results (dict): Cross-validation results
        final_metrics (dict): Test set metrics
        optuna_history (dict): Optuna optimization history
    """
    print_subheader("SAVING RESULTS")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save trained model
    model_path = os.path.join(OUTPUT_DIR, f"best_model_{model_name.replace(' ', '_')}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log_message(f"✓ Model saved: {model_path}", "SUCCESS")
    
    # Save hyperparameters
    params_path = os.path.join(OUTPUT_DIR, f"best_hyperparameters_{model_name.replace(' ', '_')}.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(best_params, f)
    log_message(f"✓ Hyperparameters saved: {params_path}", "SUCCESS")
    
    # Compile comprehensive logs
    logs = {
        "timestamp": timestamp,
        "model_name": model_name,
        "configuration": CONFIG,
        "cross_validation_results": cv_results,
        "final_test_metrics": final_metrics,
        "best_hyperparameters": best_params,
        "optuna_history": optuna_history,
        "training_summary": {
            "total_cv_folds": cv_results["cv_folds"],
            "cv_best_f1": cv_results["test_f1_mean"],
            "final_test_f1": final_metrics["f1"],
            "final_test_accuracy": final_metrics["accuracy"],
            "final_test_auc_roc": final_metrics["roc_auc"],
        },
    }
    
    # Save logs
    logs_path = os.path.join(OUTPUT_DIR, f"training_logs_{model_name.replace(' ', '_')}_{timestamp}.pkl")
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    log_message(f"✓ Training logs saved: {logs_path}", "SUCCESS")
    
    # Also save as a human-readable text summary
    summary_path = os.path.join(OUTPUT_DIR, f"training_summary_{model_name.replace(' ', '_')}_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"CI/CD BUILD FAILURE PREDICTOR - TRAINING SUMMARY\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n\n")
        
        f.write(f"CROSS-VALIDATION RESULTS (5-fold)\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"Accuracy  : {cv_results['test_accuracy_mean']:.4f} ± {cv_results['test_accuracy_std']:.4f}\n")
        f.write(f"Precision : {cv_results['test_precision_mean']:.4f} ± {cv_results['test_precision_std']:.4f}\n")
        f.write(f"Recall    : {cv_results['test_recall_mean']:.4f} ± {cv_results['test_recall_std']:.4f}\n")
        f.write(f"F1-Score  : {cv_results['test_f1_mean']:.4f} ± {cv_results['test_f1_std']:.4f}\n")
        f.write(f"ROC-AUC   : {cv_results['test_roc_auc_mean']:.4f} ± {cv_results['test_roc_auc_std']:.4f}\n\n")
        
        f.write(f"FINAL TEST SET METRICS\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"Accuracy  : {final_metrics['accuracy']:.4f}\n")
        f.write(f"Precision : {final_metrics['precision']:.4f}\n")
        f.write(f"Recall    : {final_metrics['recall']:.4f}\n")
        f.write(f"F1-Score  : {final_metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC   : {final_metrics['roc_auc']:.4f}\n\n")
        
        f.write(f"BEST HYPERPARAMETERS (Optuna)\n")
        f.write(f"{'-' * 80}\n")
        for param_name, param_value in best_params.items():
            f.write(f"{param_name}: {param_value}\n")
        f.write(f"\n")
        
        f.write(f"CONFUSION MATRIX\n")
        f.write(f"{'-' * 80}\n")
        cm = final_metrics["confusion_matrix"]
        f.write(f"True Negatives  : {cm['true_negatives']:,}\n")
        f.write(f"False Positives : {cm['false_positives']:,}\n")
        f.write(f"False Negatives : {cm['false_negatives']:,}\n")
        f.write(f"True Positives  : {cm['true_positives']:,}\n")
    
    log_message(f"✓ Summary saved: {summary_path}", "SUCCESS")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════


def main():
    """Main execution pipeline."""
    
    print_header("CI/CD BUILD FAILURE PREDICTOR - MODEL TRAINING PIPELINE")
    
    try:
        # ── STEP 1: Load Data ─────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = load_data(
            CONFIG["train_file"],
            CONFIG["test_file"],
            CONFIG["target_col"],
        )
        
        # ── STEP 2: Train Baseline Models ─────────────────────────────────────
        print_subheader("TRAINING BASELINE MODELS WITH DEFAULT PARAMETERS")
        
        baseline_models = get_baseline_models()
        cv_results_list = []
        
        for model_name, model in baseline_models.items():
            log_message(f"Training {model_name}...", "INFO")
            cv_results = evaluate_model_with_cv(
                model,
                X_train,
                y_train,
                model_name,
                k_folds=CONFIG["k_folds"],
            )
            cv_results_list.append(cv_results)
        
        # ── STEP 3: Print CV Results ──────────────────────────────────────────
        print_cv_results(cv_results_list)
        
        # ── STEP 4: Select Best Model ─────────────────────────────────────────
        best_model_name, best_cv_results = select_best_model(
            cv_results_list,
            metric="f1",
        )
        
        # ── STEP 5: Hyperparameter Tuning ─────────────────────────────────────
        best_params, optuna_history = create_optuna_study(
            best_model_name,
            X_train,
            y_train,
            X_test,
            y_test,
            n_trials=50,  # Tune this based on available time
        )
        
        # ── STEP 6: Train Final Model ─────────────────────────────────────────
        final_model = train_final_model(
            best_model_name,
            best_params,
            X_train,
            y_train,
        )
        
        # ── STEP 7: Evaluate Final Model ──────────────────────────────────────
        final_metrics = evaluate_final_model(
            final_model,
            best_model_name,
            X_test,
            y_test,
        )
        
        # ── STEP 8: Save Results ──────────────────────────────────────────────
        save_results(
            final_model,
            best_model_name,
            best_params,
            best_cv_results,
            final_metrics,
            optuna_history,
        )
        
        # ── Final Summary ─────────────────────────────────────────────────────
        print_header("✅  TRAINING PIPELINE COMPLETE", "=")
        
        print("\n📊  FINAL SUMMARY")
        print("-" * 80)
        print(f"  Best Model           : {best_model_name}")
        print(f"  CV F1-Score          : {best_cv_results['test_f1_mean']:.4f}")
        print(f"  Final Test F1-Score  : {final_metrics['f1']:.4f}")
        print(f"  Final Test Accuracy  : {final_metrics['accuracy']:.4f}")
        print(f"  Final Test ROC-AUC   : {final_metrics['roc_auc']:.4f}")
        print(f"  Optuna Trials        : {optuna_history['n_trials']}")
        
        print("\n📁  OUTPUT FILES")
        print("-" * 80)
        print(f"  ✓ Model: {OUTPUT_DIR}/best_model_{best_model_name.replace(' ', '_')}.pkl")
        print(f"  ✓ Hyperparameters: {OUTPUT_DIR}/best_hyperparameters_{best_model_name.replace(' ', '_')}.pkl")
        print(f"  ✓ Training Logs: {OUTPUT_DIR}/training_logs_{best_model_name.replace(' ', '_')}_*.pkl")
        print(f"  ✓ Summary: {OUTPUT_DIR}/training_summary_{best_model_name.replace(' ', '_')}_*.txt")
        
        print("\n🚀  NEXT STEPS")
        print("-" * 80)
        print("  1. Load the trained model for inference:")
        print(f"     import pickle")
        print(f"     model = pickle.load(open('{OUTPUT_DIR}/best_model_*.pkl', 'rb'))")
        print("  2. Load encoders/scalers (saved during preprocessing)")
        print("  3. Deploy via REST API (FastAPI/Flask)")
        print("  4. Monitor model performance in production")
        
        print("\n" + "=" * 80 + "\n")
        
    except FileNotFoundError as e:
        log_message(f"File not found: {e}", "ERROR")
        sys.exit(1)
    except KeyError as e:
        log_message(f"Column not found: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        log_message(f"Unexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
