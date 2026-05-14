import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# CUSTOM TARGET ENCODER (prevents leakage with smoothing)
# ════════════════════════════════════════════════════════════════════════════


class SimpleTargetEncoder:
    """
    Target Encoder that computes mean target value per category.

    Key Features:
    - Fits ONLY on training data to prevent leakage
    - Smoothing to handle rare categories
    - Replaces unseen categories with global mean

    Formula:
        Smoothed Mean = (Category Mean × Count + Global Mean × Smoothing) / (Count + Smoothing)
    """

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.encodings = {}
        self.global_mean = None

    def fit(self, X, y):
        """
        Fit encoder on training data.

        Args:
            X (pd.DataFrame): Features to fit (only categorical columns)
            y (pd.Series): Target variable
        """
        self.global_mean = y.mean()
        for col in X.columns:
            col_means = y.groupby(X[col]).mean()
            counts = X[col].value_counts()
            # Apply smoothing to handle rare categories
            smoothed = (col_means * counts + self.global_mean * self.smoothing) / (
                counts + self.smoothing
            )
            self.encodings[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        """
        Apply encoder to data.
        Unseen categories are replaced with global mean.

        Args:
            X (pd.DataFrame): Features to transform

        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()
        for col in X.columns:
            if col in self.encodings:
                X_encoded[col] = (
                    X[col].map(self.encodings[col]).fillna(self.global_mean)
                )
        return X_encoded

    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Load and Clean Data
# ════════════════════════════════════════════════════════════════════════════


def load_and_clean_data(file_path, target_col="build_successful"):
    """
    Load raw dataset and drop unnecessary features.

    Drops:
    - Identifiers: Unique IDs and git hashes
    - Post-Run Metrics: Metrics only available AFTER execution
    - Sparse Features: Columns with too many unique values

     IMPORTANT: Only use features available BEFORE the pipeline executes
    to prevent data leakage in production.

    Args:
        file_path (str): Path to CSV file
        target_col (str): Name of target column (build_successful)

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"  Initial shape: {df.shape}")

    # Features to drop (identifiers, post-run metrics, unstructured text, sparse)
    cols_to_drop = {
        # === BUILD IDENTIFIERS (UNIQUE IDs) ===
        "tr_build_id": "Unique build ID",
        "tr_job_id": "Unique job ID",
        "tr_build_number": "Build number (sequential)",
        # === GIT/COMMIT IDENTIFIERS (too sparse to be useful) ===
        "gh_commits_in_push": "Git commit hashes (too sparse: 412K unique)",
        "git_all_built_commits": "All commit hashes (too sparse: 613K unique)",
        "git_trigger_commit": "Trigger commit hash (too sparse: 613K unique)",
        "tr_original_commit": "Original commit hash (too sparse: 644K unique)",
        # === EXTREMELY SPARSE FEATURES ===
        "tr_jobs": "Job IDs (too sparse: 680K unique)",
        "gh_build_started_at": "Build start timestamp (too sparse: 674K unique)",
        # === PREVIOUS BUILD INFO (temporal leakage) ===
        "tr_prev_build": "Previous build ID (temporal leakage)",
        "git_prev_commit_resolution_status": "Previous commit status (temporal leakage)",
        "git_prev_built_commit": "Previous built commit (temporal leakage)",
        # ═══════════════════════════════════════════════════════════════════
        # POST-RUN METRICS — only known AFTER the build executes.
        # Including ANY of these would cause label leakage: the model learns
        # from information that does not exist at prediction time.
        # ═══════════════════════════════════════════════════════════════════
        # --- Raw test counts ---
        "tr_duration": "Build duration (post-run)",
        "tr_log_testduration": "Test duration (post-run)",
        "tr_log_setup_time": "Setup time (post-run)",
        "tr_log_num_tests_ok": "Passed test count (post-run)",
        "tr_log_num_tests_failed": "Failed test count (post-run)",
        "tr_log_num_tests_run": "Total test count (post-run)",
        "tr_log_num_tests_skipped": "Skipped test count (post-run)",
        # --- Test result flags ---
        "tr_log_bool_tests_failed": "Whether tests failed (post-run)",
        # [FIX 1] was missing — bool flag derived from post-run log
        "tr_log_bool_tests_ran": "Whether tests ran at all (post-run)",
        # --- Build/log status (direct encoding of the outcome) ---
        # [FIX 2] tr_status is the raw status field that becomes the OHE
        #         columns tr_status_passed / tr_status_failed / etc.
        #         These are post-run by definition and extremely close to
        #         the target variable itself.
        "tr_status": "Build status field (post-run — parent of tr_status_* OHE cols)",
        # [FIX 3] tr_log_status and tr_log_analyzer are derived from build
        #         logs that only exist after execution finishes.
        "tr_log_status": "Log-analysis status (post-run — parent of tr_log_status_* OHE cols)",
        "tr_log_analyzer": "Log analyzer result (post-run — parent of tr_log_analyzer_* OHE cols)",
    }

    # Only drop columns that actually exist in this dataset
    existing_drops = [c for c in cols_to_drop if c in df.columns]

    # Also drop any one-hot-encoded children that may already be in the CSV
    # (e.g. if the dataset was pre-processed and tr_status_passed exists as
    # its own column rather than the raw tr_status string).
    ohe_prefixes = ("tr_status_", "tr_log_status_", "tr_log_analyzer_")
    ohe_children = [
        c for c in df.columns if c.startswith(ohe_prefixes) and c not in existing_drops
    ]

    all_drops = existing_drops + ohe_children

    if all_drops:
        df = df.drop(columns=all_drops)
        print(
            f"\n✓ Dropped {len(existing_drops)} named columns (post-run / identifiers):"
        )
        for col in existing_drops:
            print(f"    - {col}: {cols_to_drop[col]}")
        if ohe_children:
            print(f"\n✓ Dropped {len(ohe_children)} post-run OHE child columns:")
            for col in ohe_children:
                print(f"    - {col}")

    print(f"\n✓ Shape after dropping: {df.shape}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: Handle Null Values
# ════════════════════════════════════════════════════════════════════════════


def handle_null_values(df, null_threshold=0.5):
    """
    Check and handle null values intelligently.

    Strategy:
    - Drop columns with >50% nulls (not enough data)
    - Fill numerical nulls with median (robust to outliers)
    - Fill categorical nulls with mode or "Unknown"

    Args:
        df (pd.DataFrame): Input dataframe
        null_threshold (float): Drop columns with null% > threshold

    Returns:
        pd.DataFrame: Dataframe with nulls handled
    """
    print("\n" + "=" * 60)
    print("HANDLING NULL VALUES")
    print("=" * 60)

    null_before = df.isnull().sum()
    null_cols = null_before[null_before > 0].sort_values(ascending=False)

    if len(null_cols) == 0:
        print("✓ No null values found")
        return df

    print("\nNull values BEFORE handling:")
    for col, count in null_cols.items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count:,} ({pct:.2f}%)")

    # Drop columns with too many nulls
    cols_dropped = []
    for col in null_cols.index:
        pct = (null_cols[col] / len(df)) * 100
        if pct > null_threshold * 100:
            df = df.drop(columns=[col])
            cols_dropped.append(col)

    if cols_dropped:
        print(
            f"\n✓ Dropped {len(cols_dropped)} columns (>{null_threshold*100:.0f}% nulls):"
        )
        for col in cols_dropped:
            print(f"    - {col}")

    # Fill remaining nulls
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns

    filled_num = []
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            filled_num.append(col)

    filled_cat = []
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")
            filled_cat.append(col)

    if filled_num:
        print(f"\n✓ Filled {len(filled_num)} numerical columns with median")
    if filled_cat:
        print(f"✓ Filled {len(filled_cat)} categorical columns with mode/Unknown")

    null_after = df.isnull().sum().sum()
    print(f"\n✓ Final null count: {null_after}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: Identify Categorical Features
# ════════════════════════════════════════════════════════════════════════════


def identify_categorical_features(df):
    """
    Identify which features are categorical vs numerical.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        tuple: (categorical_list, numerical_list)
    """
    categorical = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("\n" + "=" * 60)
    print(f"CATEGORICAL FEATURES ({len(categorical)})")
    print("=" * 60)
    for col in categorical:
        unique = df[col].nunique()
        sample_values = df[col].unique()[:3]
        print(f"  {col}: {unique} unique values")
        print(f"    Sample: {sample_values}")

    print("\n" + "=" * 60)
    print(f"NUMERICAL FEATURES ({len(numerical)})")
    print("=" * 60)
    for col in numerical:
        print(
            f"  {col}: min={df[col].min():.2f}, "
            f"max={df[col].max():.2f}, mean={df[col].mean():.2f}"
        )

    return categorical, numerical


# ════════════════════════════════════════════════════════════════════════════
# STEP 4: Classify Categorical Features
# ════════════════════════════════════════════════════════════════════════════


def classify_categorical_features(df, categorical, target="build_successful"):
    """
    Classify categorical features into encoding strategies:

    1. ORDINAL: Natural ordering (e.g., severity: LOW < MEDIUM < HIGH)
       → Use OrdinalEncoder to preserve order

    2. LOW CARDINALITY (≤10 unique): Few distinct values
       → Use One-Hot Encoding (creates <10 new features)

    3. HIGH CARDINALITY (>10 unique): Many distinct values
       → Use Target Encoding (compresses to 1 feature)

    Args:
        df (pd.DataFrame): Input dataframe
        categorical (list): List of categorical column names
        target (str): Target column name

    Returns:
        tuple: (ordinal, low_card, high_card, ordinal_features)
    """
    # Define ordinal features with their natural order.
    # Add entries here if your dataset has ordered categories.
    ordinal_features = {}

    ordinal = []
    high_card = []
    low_card = []

    print("\n" + "=" * 60)
    print("CLASSIFYING CATEGORICAL FEATURES")
    print("=" * 60)

    for col in categorical:
        if col == target:
            continue

        unique = df[col].nunique()

        if col in ordinal_features:
            ordinal.append(col)
            print(f"\nORDINAL: {col}")
            print(f"   Order: {' < '.join(ordinal_features[col])}")
            print(f"   Unique values: {unique}")
        elif unique > 10:
            high_card.append(col)
            print(f"\nHIGH CARDINALITY: {col}")
            print(f"   Unique values: {unique}")
            print(f"   → Will use Target Encoding")
        else:
            low_card.append(col)
            print(f"\nLOW CARDINALITY: {col}")
            print(f"   Unique values: {unique}")
            print(f"   → Will use One-Hot Encoding")

    print("\n" + "-" * 60)
    print(f"SUMMARY:")
    print(f"  Ordinal features:  {len(ordinal)}")
    print(f"  Low cardinality:   {len(low_card)}")
    print(f"  High cardinality:  {len(high_card)}")

    return ordinal, low_card, high_card, ordinal_features


# ════════════════════════════════════════════════════════════════════════════
# STEP 5: Extract Cyclic Features from Timestamps
# ════════════════════════════════════════════════════════════════════════════


def extract_cyclic_features(df):
    """
    Convert timestamps to cyclic sin/cos features.

    Timestamps have circular properties:
    - Hour: 23:00 is close to 00:00 (not far apart)
    - Day of week: Sunday (6) is close to Monday (0)
    - Month: December (12) is close to January (1)

    Sin/cos encoding preserves this circularity so models can learn it.

    Args:
        df (pd.DataFrame): Input dataframe with timestamp columns

    Returns:
        pd.DataFrame: Dataframe with cyclic features added, originals dropped
    """
    print("\n" + "=" * 60)
    print("EXTRACTING CYCLIC FEATURES FROM TIMESTAMPS")
    print("=" * 60)

    timestamp_candidates = [
        "gh_first_commit_created_at",
        "gh_pushed_at",
        "timestamp",
        "created_at",
        "pushed_at",
    ]

    timestamp_cols_found = [col for col in timestamp_candidates if col in df.columns]

    if not timestamp_cols_found:
        print("✓ No timestamp columns found (skipping cyclic feature extraction)")
        return df

    print(f"Found timestamp columns: {timestamp_cols_found}")

    for timestamp_col in timestamp_cols_found:
        print(f"\n✓ Processing: {timestamp_col}")

        dt = pd.to_datetime(df[timestamp_col])

        hour = dt.dt.hour
        day_of_week = dt.dt.dayofweek
        month = dt.dt.month
        day_of_month = dt.dt.day

        # Sine / cosine pairs for each cyclic component
        df[f"{timestamp_col}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df[f"{timestamp_col}_hour_cos"] = np.cos(2 * np.pi * hour / 24)

        df[f"{timestamp_col}_day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df[f"{timestamp_col}_day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

        df[f"{timestamp_col}_month_sin"] = np.sin(2 * np.pi * month / 12)
        df[f"{timestamp_col}_month_cos"] = np.cos(2 * np.pi * month / 12)

        df[f"{timestamp_col}_day_of_month_sin"] = np.sin(2 * np.pi * day_of_month / 31)
        df[f"{timestamp_col}_day_of_month_cos"] = np.cos(2 * np.pi * day_of_month / 31)

        # Drop the original timestamp — it has no meaning after encoding
        df = df.drop(columns=[timestamp_col])

        print(f"  Created 8 cyclic features from {timestamp_col}")

    print(f"\n✓ Cyclic extraction complete for {len(timestamp_cols_found)} column(s)")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 6: Train-Test Split (BEFORE encoding to prevent leakage)
# ════════════════════════════════════════════════════════════════════════════


def train_test_split_with_target_encoding(
    df, target="build_successful", test_size=0.2, random_state=42
):
    """
    Split data into train and test sets BEFORE any encoding.

    CRITICAL: All encoders must be fit ONLY on training data.
    This function also encodes the target variable (LabelEncoder fit on
    y_train only, then applied to y_test).

    Args:
        df (pd.DataFrame): Input dataframe
        target (str): Target column name
        test_size (float): Test set ratio (default 0.20 → 80/20 split)
        random_state (int): Reproducibility seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
               y values are integer-encoded (0 / 1)
    """
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT (80/20, stratified)")
    print("=" * 60)

    os.makedirs("src/saved_models", exist_ok=True)

    print(f"\nTarget variable : {target}")
    print(f"  Unique classes : {df[target].nunique()}")
    print(f"  Distribution:\n{df[target].value_counts()}\n")

    # Separate features and raw target
    X = df.drop(columns=[target])
    y = df[target]

    # ── Stratified split FIRST ────────────────────────────────────────────
    # We need stratify on the raw (un-encoded) y because sklearn accepts
    # string labels in stratify just fine, and this ensures the split
    # reflects the true class balance before we touch the labels.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # [FIX 4] Fit LabelEncoder on y_train ONLY, then transform both splits.
    # The original code called le.fit_transform(df[target]) on the full
    # dataset before the split, which is technically fitting on test data.
    # For a binary True/False target this makes no practical difference
    # (both classes appear everywhere), but it is inconsistent with how
    # every other encoder is handled and would matter for multi-class targets.
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train), index=y_train.index, name=target)
    y_test = pd.Series(le.transform(y_test), index=y_test.index, name=target)

    joblib.dump(le, "src/saved_models/target_label_encoder.pkl")
    print("✓ LabelEncoder fit on training labels only")
    print("✓ Saved to src/saved_models/target_label_encoder.pkl")
    print(f"  Classes: {list(le.classes_)}  →  {list(range(len(le.classes_)))}")

    print(f"\n✓ Training set : {X_train.shape[0]:,} rows × {X_train.shape[1]} columns")
    print(f"✓ Test set     : {X_test.shape[0]:,} rows × {X_test.shape[1]} columns")
    print(f"✓ Test ratio   : {test_size * 100:.1f}%")

    print(f"\nClass distribution after split:")
    print(f"  Train: {y_train.value_counts().sort_index().to_dict()}")
    print(f"  Test : {y_test.value_counts().sort_index().to_dict()}")

    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════════════════
# STEP 7: Encode Categorical Features
# ════════════════════════════════════════════════════════════════════════════


def encode_categorical_features(
    X_train, X_test, y_train, ordinal, low_card, high_card, ordinal_features
):
    """
    Encode categorical features using three strategies.

    CRITICAL: All encoders fit on training data ONLY.

    Encoding order matters for the production inference pipeline:
        1. Ordinal  → 2. One-Hot  → 3. Target

    Args:
        X_train, X_test : Training and test feature DataFrames
        y_train         : Training target (integer-encoded)
        ordinal         : List of ordinal column names
        low_card        : List of low-cardinality column names
        high_card       : List of high-cardinality column names
        ordinal_features: Dict mapping ordinal col → ordered category list

    Returns:
        tuple: (X_train_encoded, X_test_encoded)
    """
    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 60)

    # ── 1. ORDINAL ENCODING ───────────────────────────────────────────────
    print("\n1️ORDINAL ENCODING (preserves natural order)")
    print("-" * 60)

    if ordinal:
        ordinal_cols = [col for col in ordinal if col in X_train.columns]
        if ordinal_cols:
            print(f"Applying to: {ordinal_cols}\n")

            categories = [ordinal_features[col] for col in ordinal_cols]
            oe = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )

            X_train[ordinal_cols] = oe.fit_transform(X_train[ordinal_cols])
            X_test[ordinal_cols] = oe.transform(X_test[ordinal_cols])

            joblib.dump(oe, "src/saved_models/ordinal_encoder.pkl")

            for col in ordinal_cols:
                order_str = " < ".join(ordinal_features[col])
                print(f"  ✓ {col}: {order_str}")
        else:
            print("  (No ordinal features present in dataset)")
    else:
        print("  (No ordinal features defined)")

    # ── 2. ONE-HOT ENCODING ───────────────────────────────────────────────
    print("\n2️ONE-HOT ENCODING (low cardinality, ≤10 unique values)")
    print("-" * 60)

    if low_card:
        low_card_cols = [col for col in low_card if col in X_train.columns]
        if low_card_cols:
            print(f"Applying to: {low_card_cols}\n")

            X_train = pd.get_dummies(
                X_train, columns=low_card_cols, dtype=int, drop_first=False
            )
            X_test = pd.get_dummies(
                X_test, columns=low_card_cols, dtype=int, drop_first=False
            )

            # Align columns: test may be missing categories seen only in train
            missing_in_test = set(X_train.columns) - set(X_test.columns)
            for col in missing_in_test:
                X_test[col] = 0
            X_test = X_test[X_train.columns]

            for col in low_card_cols:
                n = len([c for c in X_train.columns if c.startswith(col + "_")])
                print(f"  ✓ {col}: created {n} binary features")
        else:
            print("  (No low-cardinality features present in dataset)")
    else:
        print("  (No low-cardinality features)")

    # ── 3. TARGET ENCODING ────────────────────────────────────────────────
    print("\n3️TARGET ENCODING (high cardinality, >10 unique values)")
    print("-" * 60)
    print("  Fitting encoder ONLY on training data (prevents leakage)\n")

    if high_card:
        high_card_cols = [col for col in high_card if col in X_train.columns]
        if high_card_cols:
            print(f"Applying to: {high_card_cols}\n")

            te = SimpleTargetEncoder(smoothing=1.0)

            X_train[high_card_cols] = te.fit_transform(X_train[high_card_cols], y_train)
            X_test[high_card_cols] = te.transform(X_test[high_card_cols])

            joblib.dump(te, "src/saved_models/target_encoder.pkl")

            for col in high_card_cols:
                print(f"  ✓ {col}: compressed to 1 numeric feature")
            print(f"\n  Encoder fit on : {len(X_train):,} training samples")
            print(f"  Encoder applied: {len(X_test):,} test samples")
        else:
            print("  (No high-cardinality features present in dataset)")
    else:
        print("  (No high-cardinality features)")

    return X_train, X_test


# ════════════════════════════════════════════════════════════════════════════
# STEP 8: Encode Target Variable  (no-op — encoding done in Step 6)
# ════════════════════════════════════════════════════════════════════════════


def encode_target(y_train, y_test):
    """
    Confirm target encoding status.

    The LabelEncoder is fit and applied inside
    train_test_split_with_target_encoding() (Step 6) so that it is fit on
    training labels only.  This function exists for pipeline clarity and
    prints a confirmation; it does not re-encode.

    Args:
        y_train, y_test: Integer-encoded target Series

    Returns:
        tuple: (y_train, y_test) unchanged
    """
    print("\n" + "=" * 60)
    print("TARGET ENCODING — STATUS CHECK")
    print("=" * 60)
    print("\n✓ Target already encoded by LabelEncoder in Step 6")
    print(f"  y_train classes: {sorted(y_train.unique().tolist())}")
    print(f"  y_test  classes: {sorted(y_test.unique().tolist())}")
    print(
        f"  Train positives: {y_train.sum():,} / {len(y_train):,} "
        f"({y_train.mean()*100:.1f}%)"
    )
    print(
        f"  Test  positives: {y_test.sum():,} / {len(y_test):,}  "
        f"({y_test.mean()*100:.1f}%)"
    )
    return y_train, y_test


# ════════════════════════════════════════════════════════════════════════════
# STEP 9: Normalize Features
# ════════════════════════════════════════════════════════════════════════════


def normalize_features(X_train, X_test):
    """
    Scale all features to [0, 1] using MinMaxScaler.

    CRITICAL: Scaler is fit ONLY on training data, then applied to test.

    Args:
        X_train, X_test: Feature DataFrames (post-encoding)

    Returns:
        tuple: (X_train_scaled, X_test_scaled) as DataFrames
    """
    print("\n" + "=" * 60)
    print("NORMALIZING FEATURES  (MinMax scaling → [0, 1])")
    print("=" * 60)

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    print(f"\n✓ Scaler fit on training data : {X_train_scaled.shape}")

    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Scaler applied to test data : {X_test_scaled.shape}")

    # Preserve column names
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    joblib.dump(scaler, "src/saved_models/minmax_scaler.pkl")
    print("✓ Saved scaler → src/saved_models/minmax_scaler.pkl")

    print(f"\nPost-scaling statistics (should all be in [0, 1]):")
    print(
        f"  Train — min: {X_train_scaled.values.min():.4f}  "
        f"max: {X_train_scaled.values.max():.4f}"
    )
    print(
        f"  Test  — min: {X_test_scaled.values.min():.4f}  "
        f"max: {X_test_scaled.values.max():.4f}"
    )

    return X_train_scaled, X_test_scaled


# ════════════════════════════════════════════════════════════════════════════
# STEP 10: Save Processed Datasets
# ════════════════════════════════════════════════════════════════════════════


def save_processed_datasets(X_train, X_test, y_train, y_test):
    """
    Save processed train and test sets to CSV.

    Args:
        X_train, X_test: Scaled feature DataFrames
        y_train, y_test: Integer-encoded target Series
    """
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATASETS")
    print("=" * 60)

    os.makedirs("data/processed_data", exist_ok=True)

    train_df = X_train.copy()
    train_df["build_successful"] = y_train.values

    test_df = X_test.copy()
    test_df["build_successful"] = y_test.values

    train_path = "data/processed_data/train_processed.csv"
    test_path = "data/processed_data/test_processed.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n✓ Training set → {train_path}")
    print(f"  Rows: {train_df.shape[0]:,}  |  Columns: {train_df.shape[1]}")

    print(f"\n✓ Test set     → {test_path}")
    print(f"  Rows: {test_df.shape[0]:,}  |  Columns: {test_df.shape[1]}")

    # Report saved encoder sizes
    print("\n" + "-" * 60)
    print("SAVED ENCODERS (needed for production inference):")
    print("-" * 60)
    saved_models = [
        "src/saved_models/target_label_encoder.pkl",
        "src/saved_models/ordinal_encoder.pkl",
        "src/saved_models/target_encoder.pkl",
        "src/saved_models/minmax_scaler.pkl",
    ]
    for path in saved_models:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✓ {path}  ({size_kb:.2f} KB)")
        else:
            print(f"  ✗ {path}  (not found — may not have been needed)")
