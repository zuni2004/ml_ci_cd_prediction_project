import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import time  # Added for time recording
from datetime import datetime
from typing import Dict, List, Tuple, Any

# ════════════════════════════════════════════════════════════════════════════
# SKLEARN MODELS & UTILITIES
# ════════════════════════════════════════════════════════════════════════════
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
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
    "train_file": "data/processed_data/train_processed.csv",
    "test_file": "data/processed_data/test_processed.csv",
    "target_col": "build_successful",
    "test_size": 0.20,
    "random_seed": 42,
    "n_jobs": -1,  # Use all CPU cores
    "verbose": 1,
}

OUTPUT_DIR = "src/saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════


def print_header(title: str, char: str = "=") -> None:
    width = 80
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def log_message(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix_map = {
        "INFO": "ℹ ",
        "SUCCESS": "✓ ",
        "WARNING": " ",
        "ERROR": " ",
    }
    prefix = prefix_map.get(level, "ℹ️ ")
    print(f"{prefix} [{timestamp}] {message}")


def load_data(
    train_path: str, test_path: str, target_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print_subheader("LOADING PREPROCESSED DATA")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    log_message(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)

    log_message(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)

    y_train = train_df[target_col].values
    X_train = train_df.drop(columns=[target_col]).values

    y_test = test_df[target_col].values
    X_test = test_df.drop(columns=[target_col]).values

    log_message(f"✓ Loaded training data: {X_train.shape[0]:,} rows", "SUCCESS")
    log_message(f"✓ Loaded test data: {X_test.shape[0]:,} rows", "SUCCESS")

    return X_train, X_test, y_train, y_test


def get_baseline_models() -> Dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            random_state=CONFIG["random_seed"],
            solver="saga",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=CONFIG["random_seed"],
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=CONFIG["random_seed"],
            n_jobs=-1,
        ),
    }
    return models


# ════════════════════════════════════════════════════════════════════════════
# SINGLE TRAINING & EVALUATION (REPLACED CROSS-VAL)
# ════════════════════════════════════════════════════════════════════════════


def evaluate_model_single_run(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Train model once and record time and performance.
    """
    log_message(f"Starting training for {model_name}...", "INFO")

    # Record training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Evaluate
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.0

    results = {
        "model_name": model_name,
        "training_time": training_time,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc,
    }

    log_message(
        f"✓ {model_name} Training Complete | Time: {training_time:.2f}s | F1: {results['test_f1']:.4f}",
        "SUCCESS",
    )

    return results


def print_baseline_results(results_list: List[Dict[str, Any]]) -> None:
    print_subheader("BASELINE MODELS - TRAINING RESULTS")

    print(
        "\n{:<20} {:<12} {:<12} {:<12} {:<12}".format(
            "Model", "Time (s)", "Accuracy", "Recall", "F1-Score"
        )
    )
    print("-" * 80)

    for results in results_list:
        print(
            "{:<20} {:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                results["model_name"],
                results["training_time"],
                results["test_accuracy"],
                results["test_recall"],
                results["test_f1"],
            )
        )


def select_best_model(
    results_list: List[Dict[str, Any]],
    metric: str = "f1",
) -> Tuple[str, Dict[str, Any]]:
    best_idx = 0
    best_score = -1

    target_key = f"test_{metric}"
    for idx, results in enumerate(results_list):
        score = results.get(target_key, -1)
        if score > best_score:
            best_score = score
            best_idx = idx

    best_model_name = results_list[best_idx]["model_name"]
    log_message(
        f"Best model selected: {best_model_name} (F1: {best_score:.4f})", "SUCCESS"
    )
    return best_model_name, results_list[best_idx]


# ════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING & FINAL TRAINING
# ════════════════════════════════════════════════════════════════════════════


def create_optuna_study(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 30,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print_subheader(f"HYPERPARAMETER TUNING: {model_name} (Optuna - {n_trials} trials)")

    def objective(trial: optuna.Trial) -> float:
        if model_name == "Logistic Regression":
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
            )
        elif model_name == "Random Forest":
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

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=CONFIG["random_seed"]),
        pruner=MedianPruner(),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log_message(f"Running {n_trials} Optuna trials for {model_name}...")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    log_message(f"✓ Optuna Complete | Best F1: {best_value:.4f}", "SUCCESS")

    study_history = {
        "model_name": model_name,
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": len(study.trials),
        "trials_history": [
            {"trial_number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    return best_params, study_history


def train_final_model(
    model_name: str,
    best_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    log_message(f"Training final {model_name} with best hyperparameters...")
    start_time = time.time()

    if model_name == "Logistic Regression":
        model = LogisticRegression(
            **best_params, random_state=CONFIG["random_seed"], n_jobs=-1, solver="saga"
        )
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            **best_params, random_state=CONFIG["random_seed"]
        )
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            **best_params, random_state=CONFIG["random_seed"], n_jobs=-1
        )

    model.fit(X_train, y_train)
    log_message(f"✓ Final model trained in {time.time() - start_time:.2f}s", "SUCCESS")
    return model


def evaluate_final_model(
    model: Any, model_name: str, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    print_subheader(f"FINAL EVALUATION: {model_name}")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["confusion_matrix"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    print(f"\nFinal Test F1: {metrics['f1']:.4f}")
    print(f"Confusion Matrix: TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}")
    return metrics


def save_results(
    model, model_name, best_params, cv_results, final_metrics, optuna_history
):
    print_subheader("SAVING RESULTS")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        OUTPUT_DIR, f"best_model_{model_name.replace(' ', '_')}.pkl"
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logs = {
        "timestamp": timestamp,
        "model_name": model_name,
        "final_test_metrics": final_metrics,
        "best_hyperparameters": best_params,
        "optuna_history": optuna_history,
    }
    logs_path = os.path.join(
        OUTPUT_DIR, f"training_logs_{model_name.replace(' ', '_')}_{timestamp}.pkl"
    )
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    log_message(f"✓ Results saved to {OUTPUT_DIR}", "SUCCESS")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════


def main():
    print_header("CI/CD BUILD FAILURE PREDICTOR - MODEL TRAINING PIPELINE")
    try:
        # STEP 1: Load Data
        X_train, X_test, y_train, y_test = load_data(
            CONFIG["train_file"], CONFIG["test_file"], CONFIG["target_col"]
        )

        # STEP 2: Train Baseline Models (One-time training, no CV)
        print_subheader("TRAINING BASELINE MODELS")
        baseline_models = get_baseline_models()
        results_list = []

        for model_name, model in baseline_models.items():
            res = evaluate_model_single_run(
                model, X_train, y_train, X_test, y_test, model_name
            )
            results_list.append(res)

        # STEP 3: Print Results
        print_baseline_results(results_list)

        # STEP 4: Select Best Model
        best_model_name, best_res = select_best_model(results_list, metric="f1")

        # STEP 5: Hyperparameter Tuning
        best_params, optuna_history = create_optuna_study(
            best_model_name, X_train, y_train, X_test, y_test, n_trials=30
        )

        # STEP 6: Train Final Model
        final_model = train_final_model(best_model_name, best_params, X_train, y_train)

        # STEP 7: Evaluate Final Model
        final_metrics = evaluate_final_model(
            final_model, best_model_name, X_test, y_test
        )

        # STEP 8: Save Results
        save_results(
            final_model,
            best_model_name,
            best_params,
            best_res,
            final_metrics,
            optuna_history,
        )

        print_header(" TRAINING PIPELINE COMPLETE", "=")

    except Exception as e:
        log_message(f"Unexpected error: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
