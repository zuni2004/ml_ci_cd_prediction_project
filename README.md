# CI/CD Build Failure Predictor — Data Preprocessing
Train and Test data: https://drive.google.com/drive/folders/1SvKcRz_Q1gExsJO9CEcvu0n3rQCiKrHP?usp=sharing

A production-ready preprocessing pipeline for predicting software build failures before execution using commit metadata and repository features.

## Overview

This project predicts whether a CI/CD build will **fail or pass** based on:
- **Pre-commit features**: code churn, team size, test coverage, project history
- **Temporal patterns**: commit timing (hour, day, month encoded as cyclic features)
- **Language & framework**: via target encoding to handle high cardinality

**Key property**: Uses **only features available before the build runs** — no post-execution metrics, no test results, no logs. Suitable for pre-build gates that block risky commits before pipeline execution.

## Dataset

- **Source**: Travis CI (travistorrent_8_2_2017.csv)
- **Size**: 3.7M+ builds, 62 raw features
- **Target**: `build_successful` (binary: True=passed, False=failed)
- **Class balance**: ~67% passed, ~33% failed (stratified in train/test)

## Pipeline Output

| Artifact | Rows | Columns | Use |
|----------|------|---------|-----|
| `train_processed.csv` | 2,962,076 | 45 | Model training |
| `test_processed.csv` | 740,519 | 45 | Model evaluation |
| `target_label_encoder.pkl` | — | 1 | Encode binary target in inference |
| `target_encoder.pkl` | — | 6 | High-cardinality feature encoding |
| `minmax_scaler.pkl` | — | 44 | Feature normalization in inference |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Minimum (preprocessing only)**:
```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Prepare your dataset

Place your CSV at:
```
project_root/
├── data/
│   └── raw_data/
│       └── travistorrent_8_2_2017.csv
├── main.py
└── src/
    └── data_preprocessing.py
```

Update `INPUT_FILE` in `main.py` if your filename differs.

### 3. Run the pipeline

```bash
python main.py
```

Expected output: ~2 minutes for 3.7M rows on a standard machine.

### 4. Train a model

```python
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

# Load processed data
X_train = pd.read_csv("data/processed_data/train_processed.csv")
y_train = X_train.pop("build_successful")

X_test = pd.read_csv("data/processed_data/test_processed.csv")
y_test = X_test.pop("build_successful")

# Train
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Architecture

### 10-Step Pipeline

```
1. Load & Clean          Drop identifiers, post-run metrics, sparse features (24 cols)
                         ↓ 3.7M × 62 → 3.7M × 38
2. Handle Nulls          Drop high-null cols, fill remaining with median/mode
                         ↓ 3.7M × 38 → 3.7M × 29
3. Identify Types        Separate categorical from numerical
                         ↓ 7 categorical, 19 numerical
4. Classify Cat.         Ordinal (0) / Low-card (1) / High-card (6)
5. Cyclic Features       Convert timestamps → sin/cos pairs (16 new features)
                         ↓ 3.7M × 42
6. Train-Test Split      80/20 stratified split + LabelEncoder on y_train only
                         ↓ Train: 2.96M × 42, Test: 740K × 42
7. Encode Categorical    OrdinalEncoder (0) + OneHot (1) + TargetEncoder (6)
                         ↓ 3.7M × 44 features
8. Confirm Target        Status check (no-op, encoding done in Step 6)
9. Normalize Features    MinMaxScaler fit on training set, applied to test
                         ↓ All features in [0, 1] (mostly; see note below)
10. Save Datasets        Export train/test CSVs + 4 encoder pkl files
```

### Feature Engineering

**Dropped (24 columns)**:
- Build IDs: `tr_build_id`, `tr_job_id`, `tr_build_number`
- Git hashes: `gh_commits_in_push`, `git_all_built_commits`, `git_trigger_commit`, `tr_original_commit`
- Sparse: `tr_jobs`, `gh_build_started_at`
- Temporal leakage: `tr_prev_build`, `git_prev_built_commit`, `git_prev_commit_resolution_status`
- **Post-run (critical)**: `tr_status`, `tr_log_status`, `tr_log_analyzer`, `tr_log_bool_tests_ran`, `tr_log_bool_tests_failed`, `tr_duration`, `tr_log_num_tests_*`, `tr_log_testduration`, `tr_log_setup_time`

**Kept (38 columns)**:
- Numerical (19): code churn, team size, commit history, test metrics
- Categorical (7): project name, branch, language, framework, commit timestamps

**Engineered**:
- **One-Hot**: `gh_lang` (3 unique) → 3 binary features
- **Target Encoding** (fit on train only): 6 high-cardinality features → 6 numeric
  - `gh_project_name` (1283 unique)
  - `git_branch` (40K unique)
  - `tr_log_lan` (14 unique)
  - `tr_log_frameworks` (12 unique)
  - + 2 timestamp-derived features
- **Cyclic**: 2 timestamps → 16 sin/cos features (hour, day-of-week, month, day-of-month)

**Final**: 44 features + 1 target = 45 columns

## Data Quality Assurances

✓ **No leakage**: All encoders fit on training data only
✓ **Stratified split**: Class distribution maintained (67.2% pass, 32.8% fail in both train/test)
✓ **Null handling**: 9 columns dropped (>50% nulls), 4 filled (median/mode)
✓ **No duplicates**: Raw dataset has no duplicate rows
✓ **Features available pre-build**: No post-execution metrics included

## Production Inference

To use trained model + saved encoders in a REST API:

```python
import joblib
import pandas as pd

# Load encoders (in order)
label_encoder = joblib.load("src/saved_models/target_label_encoder.pkl")
target_encoder = joblib.load("src/saved_models/target_encoder.pkl")
scaler = joblib.load("src/saved_models/minmax_scaler.pkl")

# Preprocess new data
def preprocess_build(raw_build_data):
    # 1. One-hot encode gh_lang (if needed)
    # 2. Target encode high-cardinality features using target_encoder
    # 3. Extract cyclic features from timestamps
    # 4. Scale all features using scaler
    # 5. Return [44 features]
    pass

# Predict
X_new = preprocess_build(raw_build_data)
y_pred = model.predict(X_new)  # 0 = will fail, 1 = will pass
y_proba = model.predict_proba(X_new)  # [P(fail), P(pass)]
```

See `INFERENCE_PIPELINE.md` (coming soon) for full example.

## Important Notes

### Test set scaling anomaly

After MinMax normalization, the test set has max value **4.69** (should be ≤1.0). This is expected behavior:

- MinMaxScaler learns [min, max] from training data only
- If test set contains a value never seen in training, it can exceed 1.0
- This is not an error; it's a sign the test set has an outlier
- **For tree models**: Harmless (trees don't care about absolute scale)
- **For neural networks**: Clip to [0, 1] at inference time if needed

```python
X_test_clipped = np.clip(X_test, 0, 1)
```

### Class imbalance

The target is imbalanced (~67% pass, ~33% fail), but this is natural for CI builds.

Mitigation strategies:
```python
# Option 1: Class weights
model = xgb.XGBClassifier(scale_pos_weight=1.0)  # Tune based on your threshold

# Option 2: Threshold tuning
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.4).astype(int)  # Lower threshold to catch more failures

# Option 3: SMOTE (in training pipeline, not preprocessing)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
```

## Troubleshooting

### 1. File not found: `travistorrent_8_2_2017.csv`

```
❌ Error: File not found: data/raw_data/travistorrent_8_2_2017.csv
```

**Fix**:
1. Download the CSV and place in `data/raw_data/`
2. Or update `INPUT_FILE` in `main.py` to match your filename

### 2. Column not found: `build_successful`

```
❌ KeyError: 'build_successful'
```

**Cause**: Target column name differs in your dataset.

**Fix**: Update `TARGET_COL` in `main.py`:
```python
TARGET_COL = "your_column_name"  # e.g., "build_passed", "success"
```

### 3. Out of memory on large datasets

For datasets > 10M rows, use **chunking**:

```python
chunk_size = 100_000
chunks = []
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    # preprocess chunk
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
```

### 4. Missing required library

```
ModuleNotFoundError: No module named 'sklearn'
```

**Fix**:
```bash
pip install -r requirements.txt
# or
pip install scikit-learn pandas numpy joblib
```

### 5. Encoders not loaded in inference

```
KeyError: 'gh_project_name'  # when applying target_encoder to test data
```

**Cause**: Encoder was fit on different features than test data has.

**Fix**: Ensure the same preprocessing order is applied to new data as was applied to training data.

## Files

```
project_root/
├── main.py                              # Pipeline orchestrator
├── requirements.txt                     # Dependencies
├── README.md                            # This file
│
├── src/
│   └── data_preprocessing.py            # All preprocessing functions
│
├── data/
│   ├── raw_data/
│   │   └── travistorrent_8_2_2017.csv   # Input (download separately)
│   └── processed_data/
│       ├── train_processed.csv          # Output: 2.96M rows × 45 cols
│       └── test_processed.csv           # Output: 740K rows × 45 cols
│
└── src/saved_models/
    ├── target_label_encoder.pkl        # Binary target encoder
    ├── target_encoder.pkl              # High-cardinality encoder (1.1 MB)
    ├── minmax_scaler.pkl               # Feature scaler
    └── ordinal_encoder.pkl             # Ordinal encoder (unused)
```

## Changes from original (fixes applied)

| Fix | Issue | Solution |
|-----|-------|----------|
| 1 | `tr_log_bool_tests_ran` was not dropped | Added to cols_to_drop dict |
| 2 | `tr_status*` OHE columns were leaking | Drop parent `tr_status` before processing |
| 3 | `tr_log_status*` & `tr_log_analyzer*` were leaking | Drop parent columns in cols_to_drop |
| 4 | LabelEncoder fit on full dataset before split | Now fits on y_train only post-split |

All fixes prevent **label leakage** — the model now only sees pre-build information.

## Performance Notes

- **Pipeline runtime**: ~2 minutes for 3.7M rows (on CPU)
- **Memory**: ~4-6 GB RAM during execution
- **Disk**: 
  - CSV output: ~600 MB (train) + ~150 MB (test)
  - Encoder pickles: ~1.2 MB total

## License

This preprocessing pipeline is part of the CI/CD Build Failure Predictor project.

## Support

For issues, questions, or contributions:
1. Check the Troubleshooting section above
2. Review the function docstrings in `data_preprocessing.py`
3. See output summary at the end of `main.py` execution
