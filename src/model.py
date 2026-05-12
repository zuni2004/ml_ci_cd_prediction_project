import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

# ML Models & Utilities
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Optimization & Tracking
import optuna
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Using raw string for Windows Path
BASE_DIR = r"D:\Downloads\ml_ci_cd_prediction_project-main (1)\ml_ci_cd_prediction_project-main"
OUTPUT_DIR = os.path.join(BASE_DIR, "src", "saved_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    "train_file": os.path.join(BASE_DIR, "src", "train_processed.csv"),
    "test_file": os.path.join(BASE_DIR, "src", "test_processed.csv"),
    "target_col": "build_successful",
    "random_seed": 42,
    "k_folds": 5,
    "optuna_trials": 15, # As requested
    "experiment_name": "CI_CD_Failure_Prediction_DT"
}

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def load_data():
    train_df = pd.read_csv(CONFIG["train_file"])
    test_df = pd.read_csv(CONFIG["test_file"])
    
    y_train = train_df[CONFIG["target_col"]].values
    X_train = train_df.drop(columns=[CONFIG["target_col"]]).values
    y_test = test_df[CONFIG["target_col"]].values
    X_test = test_df.drop(columns=[CONFIG["target_col"]]).values
    
    return X_train, X_test, y_train, y_test

def objective(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for Decision Tree"""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "random_state": CONFIG["random_seed"]
    }
    
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds, zero_division=0)

# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Setup MLflow
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    print(f"--- Starting Pipeline: Decision Tree Optimization ---")
    
    # 2. Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    with mlflow.start_run(run_name="Decision_Tree_Final_Training"):
        # 3. Hyperparameter Tuning with Optuna
        print(f"Running {CONFIG['optuna_trials']} Optuna trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), 
                       n_trials=CONFIG["optuna_trials"])
        
        best_params = study.best_params
        print(f"Best Hyperparameters: {best_params}")
        
        # Log params to MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", CONFIG["optuna_trials"])

        # 4. Train Final Model
        print("Training final model with best parameters...")
        final_model = DecisionTreeClassifier(**best_params, random_state=CONFIG["random_seed"])
        final_model.fit(X_train, y_train)

        # 5. Evaluate
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_proba)
        }
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print(f"Final Metrics: {metrics}")

        # 6. Save Model Locally (.pkl) for AWS Deployment
        model_filename = "best_model_decision_tree.pkl"
        save_path = os.path.join(OUTPUT_DIR, model_filename)
        
        with open(save_path, "wb") as f:
            pickle.dump(final_model, f)
            
        print(f"Successfully saved model to: {save_path}")

        # 7. Log Model and Artifacts to MLflow
        mlflow.sklearn.log_model(final_model, "decision_tree_model")
        mlflow.log_artifact(save_path)
        
        print("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()