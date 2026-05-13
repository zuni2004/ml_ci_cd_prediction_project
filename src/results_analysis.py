import pickle
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ════════════════════════════════════════════════════════════════════════════

MODELS_DIR = "src/saved_models"


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════


def list_saved_models() -> List[str]:
    """
    List all saved model files in src/saved_models.

    Returns:
        list: Filenames of saved models
    """
    if not os.path.exists(MODELS_DIR):
        print(f"Directory not found: {MODELS_DIR}")
        return []

    model_files = glob.glob(os.path.join(MODELS_DIR, "best_model_*.pkl"))
    return [os.path.basename(f) for f in model_files]


def load_model(filename: str):
    """
    Load a trained model from pickle file.

    Args:
        filename (str): Model filename (e.g., "best_model_Random_Forest.pkl")

    Returns:
        Trained model object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")

    print(f"Loading model: {filepath}")

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"✓ Model loaded successfully")
    return model


def load_hyperparameters(filename: str) -> Dict[str, Any]:
    """
    Load best hyperparameters for a model.

    Args:
        filename (str): Hyperparameters filename

    Returns:
        dict: Hyperparameters dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Hyperparameters file not found: {filepath}")

    print(f"Loading hyperparameters: {filepath}")

    with open(filepath, "rb") as f:
        hyperparams = pickle.load(f)

    print(f"✓ Hyperparameters loaded: {list(hyperparams.keys())}")
    return hyperparams


def load_logs(filename_pattern: str = None) -> Dict[str, Any]:
    """
    Load training logs from pickle file.

    Args:
        filename_pattern (str): Log filename or pattern (e.g., "training_logs_Random_Forest_*.pkl")

    Returns:
        dict: Training logs dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if filename_pattern is None:
        # Find the most recent logs file
        log_files = glob.glob(os.path.join(MODELS_DIR, "training_logs_*.pkl"))
        if not log_files:
            raise FileNotFoundError(f"No training logs found in {MODELS_DIR}")
        filepath = max(log_files)  # Get most recent
    else:
        # Check if pattern has wildcard
        if "*" in filename_pattern:
            matches = glob.glob(os.path.join(MODELS_DIR, filename_pattern))
            if not matches:
                raise FileNotFoundError(f"No files match pattern: {filename_pattern}")
            filepath = max(matches)  # Get most recent if multiple matches
        else:
            filepath = os.path.join(MODELS_DIR, filename_pattern)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Logs file not found: {filepath}")

    print(f"Loading logs: {filepath}")

    with open(filepath, "rb") as f:
        logs = pickle.load(f)

    print(f"✓ Logs loaded successfully")
    return logs


# ════════════════════════════════════════════════════════════════════════════
# RESULTS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════


def print_training_summary(logs: Dict[str, Any]) -> None:
    """
    Print a formatted training summary from logs.

    Args:
        logs (dict): Training logs dictionary
    """
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY".center(80))
    print("=" * 80)

    # Header info
    print(f"\nModel         : {logs['model_name']}")
    print(f"Timestamp     : {logs['timestamp']}")
    print(f"Random Seed   : {logs['configuration']['random_seed']}")
    print(f"K-Folds       : {logs['configuration']['k_folds']}")

    # Cross-validation results
    print("\n" + "-" * 80)
    print("CROSS-VALIDATION RESULTS (Training Set)")
    print("-" * 80)

    cv_results = logs["cross_validation_results"]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print(f"\n{'Metric':<15} {'Mean':<12} {'Std Dev':<12} {'Train Mean':<12}")
    print("-" * 80)

    for metric in metrics:
        test_mean = cv_results.get(f"test_{metric}_mean", 0)
        test_std = cv_results.get(f"test_{metric}_std", 0)
        train_mean = cv_results.get(f"train_{metric}_mean", 0)

        print(
            f"{metric.capitalize():<15} {test_mean:<12.4f} {test_std:<12.4f} {train_mean:<12.4f}"
        )

    # Final test results
    print("\n" + "-" * 80)
    print("FINAL TEST SET PERFORMANCE")
    print("-" * 80)

    final_metrics = logs["final_test_metrics"]

    print(f"\n{'Metric':<15} {'Value':<12}")
    print("-" * 80)

    for metric in metrics:
        value = final_metrics.get(metric, 0)
        print(f"{metric.capitalize():<15} {value:<12.4f}")

    # Confusion matrix
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX (Test Set)")
    print("-" * 80)

    cm = final_metrics["confusion_matrix"]
    print(f"\nTrue Negatives (Correct Failures)  : {cm['true_negatives']:>10,}")
    print(f"False Positives (Incorrect Passes) : {cm['false_positives']:>10,}")
    print(f"False Negatives (Missed Failures)  : {cm['false_negatives']:>10,}")
    print(f"True Positives (Correct Passes)    : {cm['true_positives']:>10,}")

    total = sum(cm.values())
    print(f"\nTotal Predictions: {total:,}")

    # Best hyperparameters
    print("\n" + "-" * 80)
    print("BEST HYPERPARAMETERS (Optuna)")
    print("-" * 80)

    best_params = logs["best_hyperparameters"]
    for param_name, param_value in best_params.items():
        print(f"  {param_name:<25}: {param_value}")

    # Optuna history
    optuna_hist = logs["optuna_history"]
    print("\n" + "-" * 80)
    print("OPTUNA OPTIMIZATION HISTORY")
    print("-" * 80)
    print(f"\nTotal Trials      : {optuna_hist['n_trials']}")
    print(f"Best Trial Value  : {optuna_hist['best_value']:.4f}")

    # Print top 5 trials
    trials = sorted(
        optuna_hist["trials_history"],
        key=lambda x: x["value"] if x["value"] is not None else -1,
        reverse=True,
    )

    print("\nTop 5 Trials:")
    print("-" * 80)
    print(f"{'Trial':<8} {'F1-Score':<12} {'Key Hyperparameters'}")
    print("-" * 80)

    for idx, trial in enumerate(trials[:5], 1):
        f1_score = trial["value"] if trial["value"] is not None else 0
        params_str = ", ".join(
            [f"{k}={v}" for k, v in list(trial["params"].items())[:2]]
        )
        print(f"{trial['trial_number']:<8} {f1_score:<12.4f} {params_str}")

    print("\n" + "=" * 80 + "\n")


def get_classification_report(logs: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract classification report as a pandas DataFrame.

    Args:
        logs (dict): Training logs

    Returns:
        pd.DataFrame: Classification report
    """
    report_dict = logs["final_test_metrics"]["classification_report"]

    # Convert to DataFrame
    df = pd.DataFrame(report_dict).transpose()
    df = df[["precision", "recall", "f1-score", "support"]]

    return df


def get_optuna_trials_dataframe(logs: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert Optuna trials to a pandas DataFrame for analysis.

    Args:
        logs (dict): Training logs

    Returns:
        pd.DataFrame: Optuna trials with hyperparameters and scores
    """
    trials_data = []

    for trial in logs["optuna_history"]["trials_history"]:
        row = {
            "trial_number": trial["trial_number"],
            "f1_score": trial["value"],
            **trial["params"],
        }
        trials_data.append(row)

    df = pd.DataFrame(trials_data)
    return df.sort_values("f1_score", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# MODEL INFERENCE
# ════════════════════════════════════════════════════════════════════════════


def predict_single(
    model,
    features: np.ndarray,
    return_proba: bool = False,
) -> Tuple[int, float]:
    """
    Make a prediction for a single sample.

    Args:
        model: Trained model
        features (np.ndarray): Feature vector (1D array)
        return_proba (bool): Whether to return probabilities

    Returns:
        tuple: (prediction, probability) or just prediction
    """
    # Ensure correct shape
    if features.ndim == 1:
        features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    if return_proba:
        proba = model.predict_proba(features)[0]
        return prediction, proba

    return prediction


def predict_batch(
    model,
    features: np.ndarray,
    return_proba: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions for multiple samples.

    Args:
        model: Trained model
        features (np.ndarray): Feature matrix
        return_proba (bool): Whether to return probabilities

    Returns:
        tuple: (predictions, probabilities) or just predictions
    """
    predictions = model.predict(features)

    if return_proba:
        probabilities = model.predict_proba(features)
        return predictions, probabilities

    return predictions


def get_feature_importance(model) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.

    Args:
        model: Trained model (must have feature_importances_ attribute)

    Returns:
        pd.DataFrame: Features with importance scores (sorted)

    Raises:
        AttributeError: If model doesn't support feature importance
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model {type(model).__name__} does not have feature_importances_ attribute"
        )

    importances = model.feature_importances_
    n_features = len(importances)

    df = pd.DataFrame(
        {
            "feature_index": np.arange(n_features),
            "importance": importances,
            "importance_percent": 100.0 * importances / np.sum(importances),
        }
    )

    df = df.sort_values("importance", ascending=False)
    return df


# ════════════════════════════════════════════════════════════════════════════
# QUICK START FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════


def quick_load_best_model() -> Tuple[Any, Dict[str, Any]]:
    """
    Quickly load the best available model and its logs.

    Returns:
        tuple: (model, logs)
    """
    # Find most recent model
    model_files = glob.glob(os.path.join(MODELS_DIR, "best_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}")

    latest_model_file = max(model_files)
    model = load_model(os.path.basename(latest_model_file))

    # Find corresponding logs
    model_name = (
        os.path.basename(latest_model_file)
        .replace("best_model_", "")
        .replace(".pkl", "")
    )
    logs = load_logs(f"training_logs_{model_name}_*.pkl")

    return model, logs


def print_available_models() -> None:
    """Print all available saved models."""
    models = list_saved_models()

    if not models:
        print("No saved models found.")
        return

    print("\n" + "=" * 80)
    print("AVAILABLE SAVED MODELS".center(80))
    print("=" * 80 + "\n")

    for idx, model_file in enumerate(models, 1):
        print(f"  {idx}. {model_file}")

    print("\n" + "=" * 80 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE (when run as script)
# ════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    """
    Example usage of the results analysis module.

    Uncomment and modify as needed for your use case.
    """

    print("\n" + "=" * 80)
    print("CI/CD BUILD FAILURE PREDICTOR - RESULTS ANALYSIS".center(80))
    print("=" * 80 + "\n")

    try:
        # List available models
        print_available_models()

        # Load best model and logs
        print("Loading best model and logs...")
        model, logs = quick_load_best_model()

        # Print training summary
        print_training_summary(logs)

        # Print classification report
        print("\nCLASSIFICATION REPORT")
        print("-" * 80)
        class_report = get_classification_report(logs)
        print(class_report)

        # Print Optuna trials
        print("\n\nOPTUNA TRIALS (Top 10)")
        print("-" * 80)
        trials_df = get_optuna_trials_dataframe(logs)
        print(trials_df.head(10).to_string())

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            print("\n\nTOP 10 MOST IMPORTANT FEATURES")
            print("-" * 80)
            feature_imp = get_feature_importance(model)
            print(feature_imp.head(10).to_string())

        print("\n" + "=" * 80 + "\n")

        print("✓ Analysis complete!")
        print("\nTo use this module in your code:")
        print("  from results_analysis import load_model, quick_load_best_model")
        print("  model, logs = quick_load_best_model()")
        print("  predictions = model.predict(X_new)")

    except Exception as e:
        print(f"\n Error: {e}")
        print("\nMake sure you have:")
        print("  1. Run model_training.py to train models")
        print("  2. Saved logs are in src/saved_models/")
        print("  3. All dependencies installed")
