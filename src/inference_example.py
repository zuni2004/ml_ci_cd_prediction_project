"""
CI/CD BUILD FAILURE PREDICTOR - QUICK START & INFERENCE EXAMPLE

This script demonstrates how to:
1. Load a trained model from pickle files
2. Make predictions on new data
3. Interpret results
4. Use in production

Usage:
    python inference_example.py
"""

import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════


def load_trained_model(model_path: str = "src/saved_models/best_model_Random_Forest.pkl"):
    """
    Load the trained model from a pickle file.
    
    Args:
        model_path (str): Path to the saved model pickle file
        
    Returns:
        Trained sklearn model object
        
    Example:
        model = load_trained_model()
        predictions = model.predict(X_new)
    """
    print(f"Loading model from: {model_path}")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Run model_training.py first to train the model")
        raise


def load_encoders_and_scaler(
    target_encoder_path: str = "src/saved_models/target_encoder.pkl",
    minmax_scaler_path: str = "src/saved_models/minmax_scaler.pkl",
) -> Tuple[Any, Any]:
    """
    Load preprocessing artifacts (target encoder and feature scaler).
    
    These were created during data preprocessing and are needed for
    preprocessing new data before inference.
    
    Args:
        target_encoder_path (str): Path to target encoder pickle
        minmax_scaler_path (str): Path to MinMax scaler pickle
        
    Returns:
        tuple: (target_encoder, minmax_scaler)
    """
    print("Loading preprocessing artifacts...")
    
    try:
        with open(target_encoder_path, "rb") as f:
            target_encoder = pickle.load(f)
        print(f"  ✓ Target encoder loaded")
    except FileNotFoundError:
        print(f"  ⚠️  Target encoder not found (may not be needed for inference)")
        target_encoder = None
    
    try:
        with open(minmax_scaler_path, "rb") as f:
            minmax_scaler = pickle.load(f)
        print(f"  ✓ MinMax scaler loaded")
    except FileNotFoundError:
        print(f"  ⚠️  MinMax scaler not found")
        minmax_scaler = None
    
    return target_encoder, minmax_scaler


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════


def predict_single_build(
    model,
    features: np.ndarray,
    return_proba: bool = True,
) -> Dict[str, Any]:
    """
    Make a prediction for a single build.
    
    Args:
        model: Trained sklearn model
        features (np.ndarray): Feature vector (1D array of 44 features)
        return_proba (bool): Whether to return prediction probabilities
        
    Returns:
        dict: {
            'prediction': 0 (fail) or 1 (pass),
            'prediction_label': 'FAIL' or 'PASS',
            'probability_fail': float between 0-1,
            'probability_pass': float between 0-1,
            'confidence': float (max probability)
        }
        
    Example:
        features = np.array([...])  # 44 preprocessed features
        result = predict_single_build(model, features)
        print(f"Prediction: {result['prediction_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
    """
    # Ensure correct shape
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Make prediction
    prediction = int(model.predict(features)[0])
    
    result = {
        'prediction': prediction,
        'prediction_label': 'PASS' if prediction == 1 else 'FAIL',
    }
    
    if return_proba and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        result['probability_fail'] = float(proba[0])
        result['probability_pass'] = float(proba[1])
        result['confidence'] = float(max(proba))
    
    return result


def predict_batch(
    model,
    features: np.ndarray,
    return_proba: bool = True,
) -> Dict[str, Any]:
    """
    Make predictions for multiple builds.
    
    Args:
        model: Trained sklearn model
        features (np.ndarray): Feature matrix (shape: n × 44)
        return_proba (bool): Whether to return probabilities
        
    Returns:
        dict: {
            'predictions': array of 0s and 1s,
            'probabilities': array of [P(fail), P(pass)] for each sample
        }
        
    Example:
        X_new = np.array([...])  # Shape: (100, 44)
        results = predict_batch(model, X_new)
        print(f"Pass rate: {results['predictions'].sum() / len(results['predictions']):.1%}")
    """
    # Make predictions
    predictions = model.predict(features)
    
    result = {
        'predictions': predictions,
        'prediction_counts': {
            'failures': int(np.sum(predictions == 0)),
            'passes': int(np.sum(predictions == 1)),
        }
    }
    
    if return_proba and hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)
        result['probabilities_fail'] = probabilities[:, 0]
        result['probabilities_pass'] = probabilities[:, 1]
        result['mean_confidence'] = float(np.max(probabilities, axis=1).mean())
    
    return result


def interpret_prediction(prediction_dict: Dict[str, Any]) -> str:
    """
    Generate a human-readable interpretation of a prediction.
    
    Args:
        prediction_dict (dict): Result from predict_single_build()
        
    Returns:
        str: Formatted interpretation
    """
    label = prediction_dict['prediction_label']
    confidence = prediction_dict.get('confidence', 0)
    prob_fail = prediction_dict.get('probability_fail', 0)
    prob_pass = prediction_dict.get('probability_pass', 0)
    
    interpretation = f"\n{'='*70}\n"
    interpretation += f"PREDICTION: {label}\n"
    interpretation += f"{'='*70}\n"
    
    interpretation += f"  • Decision        : Build will {'PASS ✓' if label == 'PASS' else 'FAIL ❌'}\n"
    interpretation += f"  • Confidence      : {confidence:.1%}\n"
    interpretation += f"  • P(Fail)         : {prob_fail:.1%}\n"
    interpretation += f"  • P(Pass)         : {prob_pass:.1%}\n"
    
    if confidence > 0.9:
        interpretation += f"  • Assessment      : Very confident prediction\n"
    elif confidence > 0.75:
        interpretation += f"  • Assessment      : Confident prediction\n"
    elif confidence > 0.6:
        interpretation += f"  • Assessment      : Moderate confidence, review recommended\n"
    else:
        interpretation += f"  • Assessment      : Low confidence, manual review needed\n"
    
    interpretation += f"{'='*70}\n"
    
    return interpretation


# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════


def load_test_data_sample(
    test_file: str = "data/preprocessed/test_processed.csv",
    n_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a sample of test data for demonstration.
    
    Args:
        test_file (str): Path to test CSV
        n_samples (int): Number of samples to load
        
    Returns:
        tuple: (X_test, y_test) - first n_samples from test set
    """
    print(f"Loading test data sample from: {test_file}")
    
    try:
        df = pd.read_csv(test_file, nrows=n_samples)
        y_test = df['build_successful'].values
        X_test = df.drop(columns=['build_successful']).values
        
        print(f"✓ Loaded {len(X_test)} samples with {X_test.shape[1]} features")
        return X_test, y_test
    
    except FileNotFoundError:
        print(f"❌ Test file not found: {test_file}")
        raise


def create_dummy_features(n_samples: int = 1) -> np.ndarray:
    """
    Create dummy feature vectors for testing.
    
    Real features would come from your CI/CD system:
    - Code churn (lines changed, files modified)
    - Team metrics (number of collaborators, commit frequency)
    - Project history (previous builds, age)
    - Language/framework (one-hot encoded)
    - Timestamps (cyclic encoded as sin/cos)
    
    Args:
        n_samples (int): Number of dummy samples to create
        
    Returns:
        np.ndarray: Random features (shape: n_samples × 44)
    """
    print(f"Creating {n_samples} dummy feature vector(s) for testing...")
    print("  Note: Real features should come from your CI/CD system")
    
    # Create random features in [0, 1] (already normalized)
    dummy_features = np.random.uniform(0, 1, size=(n_samples, 44))
    
    print(f"✓ Created dummy features with shape {dummy_features.shape}")
    return dummy_features


# ════════════════════════════════════════════════════════════════════════════
# MAIN DEMONSTRATION
# ════════════════════════════════════════════════════════════════════════════


def main():
    """Run inference demonstration."""
    
    print("\n" + "="*70)
    print("CI/CD BUILD FAILURE PREDICTOR - INFERENCE EXAMPLE".center(70))
    print("="*70)
    
    try:
        # ── Step 1: Load model ────────────────────────────────────────────
        print("\n[1] Loading trained model...")
        model = load_trained_model()
        
        # ── Step 2: Load encoders (optional, for preprocessing) ──────────
        print("\n[2] Loading preprocessing artifacts...")
        target_encoder, minmax_scaler = load_encoders_and_scaler()
        
        # ── Step 3: Load sample test data ─────────────────────────────────
        print("\n[3] Loading test data sample...")
        try:
            X_test, y_test = load_test_data_sample(n_samples=5)
        except FileNotFoundError:
            print("     (Test file not found, using dummy data instead)")
            X_test = create_dummy_features(n_samples=5)
            y_test = np.array([0, 1, 1, 0, 1])  # Dummy labels
        
        # ── Step 4: Make single prediction ────────────────────────────────
        print("\n[4] Making single prediction...")
        print("-" * 70)
        
        single_result = predict_single_build(model, X_test[0])
        print(interpret_prediction(single_result))
        
        # ── Step 5: Make batch predictions ───────────────────────────────
        print("\n[5] Making batch predictions...")
        print("-" * 70)
        
        batch_result = predict_batch(model, X_test)
        
        print("\nBatch Results:")
        print(f"  • Total predictions     : {len(X_test)}")
        print(f"  • Predicted failures    : {batch_result['prediction_counts']['failures']}")
        print(f"  • Predicted passes      : {batch_result['prediction_counts']['passes']}")
        print(f"  • Mean confidence       : {batch_result['mean_confidence']:.1%}")
        print(f"  • Pass rate             : {batch_result['prediction_counts']['passes']/len(X_test):.1%}")
        
        # ── Step 6: Compare with actual labels (if available) ───────────
        if len(y_test) == len(batch_result['predictions']):
            print("\n[6] Comparing with actual labels...")
            print("-" * 70)
            
            matches = np.sum(batch_result['predictions'] == y_test)
            accuracy = matches / len(y_test)
            
            print(f"  • Correct predictions   : {matches}/{len(y_test)}")
            print(f"  • Accuracy on sample    : {accuracy:.1%}")
            
            # Confusion matrix
            tp = np.sum((batch_result['predictions'] == 1) & (y_test == 1))
            fp = np.sum((batch_result['predictions'] == 1) & (y_test == 0))
            fn = np.sum((batch_result['predictions'] == 0) & (y_test == 1))
            tn = np.sum((batch_result['predictions'] == 0) & (y_test == 0))
            
            print(f"\n  Confusion Matrix (on sample):")
            print(f"    True Positives  : {tp}")
            print(f"    False Positives : {fp}")
            print(f"    False Negatives : {fn}")
            print(f"    True Negatives  : {tn}")
        
        # ── Step 7: Show prediction probabilities ──────────────────────
        print("\n[7] Prediction probabilities for each sample:")
        print("-" * 70)
        print(f"{'Sample':<8} {'Prediction':<12} {'P(Fail)':<12} {'P(Pass)':<12} {'Confidence'}")
        print("-" * 70)
        
        for idx in range(len(X_test)):
            pred = batch_result['predictions'][idx]
            p_fail = batch_result['probabilities_fail'][idx]
            p_pass = batch_result['probabilities_pass'][idx]
            confidence = max(p_fail, p_pass)
            
            label = "PASS" if pred == 1 else "FAIL"
            print(f"{idx:<8} {label:<12} {p_fail:<12.4f} {p_pass:<12.4f} {confidence:>10.1%}")
        
        # ── Final Summary ────────────────────────────────────────────────
        print("\n" + "="*70)
        print("SUMMARY".center(70))
        print("="*70)
        
        print("\n✓ Model inference working correctly!")
        print("\nNext steps:")
        print("  1. Load your actual build features (from CI/CD system)")
        print("  2. Preprocess using saved encoders/scaler")
        print("  3. Call predict_batch(model, X_new)")
        print("  4. Use probabilities for decision thresholds:")
        print("     - High confidence (>0.9): Auto-block or allow")
        print("     - Medium confidence (0.6-0.9): Require review")
        print("     - Low confidence (<0.6): Manual inspection needed")
        
        print("\nFor production deployment:")
        print("  - Wrap in FastAPI/Flask for REST API")
        print("  - Integrate with your CI/CD pipeline")
        print("  - Monitor predictions vs actual outcomes")
        print("  - Retrain monthly to keep model fresh")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

    # ════════════════════════════════════════════════════════════════════
    # CODE SNIPPETS FOR YOUR INTEGRATION
    # ════════════════════════════════════════════════════════════════════
    
    """
    ── Example 1: Simple Load & Predict ──
    
    import pickle
    import numpy as np
    
    # Load model
    model = pickle.load(open("src/saved_models/best_model_Random_Forest.pkl", "rb"))
    
    # Your features (44 values, already preprocessed)
    X = np.array([0.5, 0.3, 0.8, ...])  # Shape: (44,)
    
    # Predict
    pred = model.predict([X])[0]  # 0 or 1
    proba = model.predict_proba([X])[0]  # [P(fail), P(pass)]
    
    print(f"Prediction: {'PASS' if pred == 1 else 'FAIL'}")
    print(f"Confidence: {max(proba):.1%}")
    
    
    ── Example 2: FastAPI Integration ──
    
    from fastapi import FastAPI
    from pydantic import BaseModel
    import pickle
    
    app = FastAPI()
    model = pickle.load(open("src/saved_models/best_model_Random_Forest.pkl", "rb"))
    
    class BuildRequest(BaseModel):
        features: list[float]
    
    @app.post("/predict")
    def predict(request: BuildRequest):
        pred = model.predict([request.features])[0]
        proba = model.predict_proba([request.features])[0]
        
        return {
            "result": "PASS" if pred == 1 else "FAIL",
            "confidence": float(max(proba))
        }
    
    # Run: uvicorn app:app --host 0.0.0.0 --port 8000
    
    
    ── Example 3: Pre-commit Hook Integration ──
    
    #!/bin/bash
    # hooks/pre-push
    
    # Get features from current commit
    FEATURES=$(python extract_features.py)
    
    # Get prediction from API
    RESPONSE=$(curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d "{\"features\": $FEATURES}")
    
    RESULT=$(echo $RESPONSE | jq -r '.result')
    
    if [ "$RESULT" == "FAIL" ]; then
        echo "⚠️  Prediction: This build is likely to FAIL"
        echo "    Continue anyway? (y/n)"
        read -r response
        [ "$response" != "y" ] && exit 1
    fi
    
    exit 0
    """
