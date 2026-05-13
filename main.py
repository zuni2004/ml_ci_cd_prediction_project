import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import (
    load_and_clean_data,
    handle_null_values,
    identify_categorical_features,
    classify_categorical_features,
    extract_cyclic_features,
    train_test_split_with_target_encoding,
    encode_categorical_features,
    encode_target,
    normalize_features,
    save_processed_datasets,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# Update this path to match your CSV filename.
INPUT_FILE = "data/raw_data/travistorrent_8_2_2017.csv"
TARGET_COL = "build_successful"
TEST_SIZE = 0.20  # 80 / 20 train-test split
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────


def print_header(title):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def main():
    print_header("CI/CD BUILD FAILURE PREDICTOR — PREPROCESSING PIPELINE")
    print(f"Dataset : {INPUT_FILE}")
    print(f"Target  : {TARGET_COL}  (binary)")

    # ── STEP 1: Load & clean ──────────────────────────────────────────────────
    print("\n📂  STEP 1: Loading and cleaning dataset...")
    print("-" * 80)

    if not os.path.exists(INPUT_FILE):
        print(f"\n  File not found: {INPUT_FILE}")
        print(f"\n  Working directory: {os.getcwd()}")
        print("\n  Fix:")
        print(f"    1. Place your CSV at:  {INPUT_FILE}")
        print(f"    2. Or update INPUT_FILE at the top of main.py")
        sys.exit(1)

    df = load_and_clean_data(INPUT_FILE, target_col=TARGET_COL)
    print(f"\n✓  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── STEP 2: Handle nulls ──────────────────────────────────────────────────
    print("\n🧹  STEP 2: Handling null values...")
    print("-" * 80)
    df = handle_null_values(df)

    # ── STEP 3: Identify feature types ────────────────────────────────────────
    print("\n   STEP 3: Identifying categorical vs numerical features...")
    print("-" * 80)
    categorical, numerical = identify_categorical_features(df)

    # ── STEP 4: Classify categorical features ─────────────────────────────────
    print("\n  STEP 4: Classifying categorical features by cardinality...")
    print("-" * 80)
    ordinal, low_card, high_card, ordinal_features = classify_categorical_features(
        df, categorical, target=TARGET_COL
    )

    # ── STEP 5: Cyclic timestamp features ─────────────────────────────────────
    print("\n  STEP 5: Extracting cyclic features from timestamps...")
    print("-" * 80)
    df = extract_cyclic_features(df)

    # ── STEP 6: Train-test split + encode target ───────────────────────────────
    print("\n   STEP 6: Stratified train-test split (LabelEncoder on y_train only)...")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split_with_target_encoding(
        df, target=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # ── STEP 7: Encode categorical features ───────────────────────────────────
    print("\n  STEP 7: Encoding categorical features...")
    print("-" * 80)
    X_train, X_test = encode_categorical_features(
        X_train, X_test, y_train, ordinal, low_card, high_card, ordinal_features
    )

    # ── STEP 8: Confirm target encoding ───────────────────────────────────────
    # Note: encoding happens in Step 6; this step prints a status summary only.
    print("\n  STEP 8: Confirming target encoding...")
    print("-" * 80)
    y_train, y_test = encode_target(y_train, y_test)

    # ── STEP 9: Normalise features ────────────────────────────────────────────
    print("\n  STEP 9: Normalising features (MinMax → [0, 1])...")
    print("-" * 80)
    X_train, X_test = normalize_features(X_train, X_test)

    # ── STEP 10: Save processed data ──────────────────────────────────────────
    print("\n  STEP 10: Saving processed datasets...")
    print("-" * 80)
    save_processed_datasets(X_train, X_test, y_train, y_test)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_header("  PREPROCESSING COMPLETE")

    n_total = X_train.shape[0] + X_test.shape[0]
    train_pct = X_train.shape[0] / n_total * 100
    test_pct = X_test.shape[0] / n_total * 100

    print("\n  FINAL STATISTICS")
    print("-" * 80)
    print(
        f"  Training set  : {X_train.shape[0]:,} rows × {X_train.shape[1]} features  ({train_pct:.1f}%)"
    )
    print(
        f"  Test set      : {X_test.shape[0]:,} rows × {X_test.shape[1]} features  ({test_pct:.1f}%)"
    )
    print(f"  Target        : {TARGET_COL}  (0 = failed, 1 = passed)")

    print("\n  OUTPUT FILES")
    print("-" * 80)
    output_files = [
        ("data/processed_data/train_processed.csv", "Training features + target"),
        ("data/processed_data/test_processed.csv", "Test features + target"),
        ("src/saved_models/target_label_encoder.pkl", "Target label encoder"),
        ("src/saved_models/ordinal_encoder.pkl", "Ordinal feature encoder"),
        ("src/saved_models/target_encoder.pkl", "Target encoder (high-cardinality)"),
        ("src/saved_models/minmax_scaler.pkl", "MinMax feature scaler"),
    ]
    for filepath, description in output_files:
        mark = "✓" if os.path.exists(filepath) else "✗"
        print(f"  {mark}  {filepath}")
        print(f"       {description}")

    print("\n  NEXT STEPS")
    print("-" * 80)
    print("  1. Train a model on data/processed_data/train_processed.csv")
    print("     Recommended: XGBoost, LightGBM, or Random Forest")
    print("  2. Evaluate on data/processed_data/test_processed.csv")
    print("  3. Expose the model via a REST API (FastAPI / Flask)")
    print("  4. Apply saved encoders in the same order at inference time:")
    print(
        "       LabelEncoder (target) → OrdinalEncoder → OneHot → TargetEncoder → MinMaxScaler"
    )

    print("\n   WHAT WAS DROPPED (post-run / leaky columns)")
    print("-" * 80)
    print("  Identifiers  : tr_build_id, tr_job_id, tr_build_number, git hashes")
    print("  Post-run     : tr_duration, tr_log_* test counts, tr_log_bool_tests_ran")
    print("  Status cols  : tr_status, tr_status_* (direct encoding of the outcome)")
    print("  Log analysis : tr_log_status, tr_log_status_*, tr_log_analyzer_*")
    print("  Temporal     : tr_prev_build, git_prev_built_commit, git_prev_commit_*")
    print("\n  All encoders were fit on TRAINING data only — no test leakage.")

    print("\n" + "=" * 80)
    print("CI/CD Preprocessing Pipeline — done.".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"\n  FILE NOT FOUND: {exc}")
        print("    Check that your CSV exists and INPUT_FILE is correct.")
        sys.exit(1)
    except KeyError as exc:
        print(f"\n  COLUMN NOT FOUND: {exc}")
        print(f"    Expected target column: '{TARGET_COL}'")
        print("    Check column names in your CSV.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n  UNEXPECTED ERROR: {exc}")
        print("\nDebugging checklist:")
        print("  • CSV is not corrupted")
        print("  • Required libraries installed: pandas, scikit-learn, joblib, numpy")
        print("  • data_preprocessing.py is in src/")
        print()
        import traceback

        traceback.print_exc()
        sys.exit(1)
