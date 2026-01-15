import re
from datetime import datetime
from io import BytesIO
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(uploaded_file, cache_dir=".data_cache"):
    """Load data from uploaded file using DuckDB for memory efficiency"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Create a unique cache key based on file name and last modified time
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    parquet_path = cache_dir / f"{hash(file_id)}.parquet"
    
    # Convert to parquet if not already done
    if not parquet_path.exists():
        if uploaded_file.name.endswith('.xlsx'):
            # Read first 1000 rows to infer schema
            df_sample = pd.read_excel(uploaded_file, nrows=1000)
            schema = pa.Schema.from_pandas(df_sample)
            
            # Read and write in chunks
            chunks = pd.read_excel(uploaded_file, chunksize=10000)
            with pq.ParquetWriter(str(parquet_path), schema) as writer:
                for chunk in chunks:
                    table = pa.Table.from_pandas(chunk, schema=schema)
                    writer.write_table(table)
        else:
            # For CSV/other formats
            chunks = pd.read_csv(uploaded_file, chunksize=10000)
            first_chunk = True
            for chunk in chunks:
                if first_chunk:
                    schema = pa.Schema.from_pandas(chunk)
                    writer = pq.ParquetWriter(str(parquet_path), schema)
                    first_chunk = False
                table = pa.Table.from_pandas(chunk, schema=schema)
                writer.write_table(table)
            writer.close()
    
    # Connect to DuckDB and register the parquet file
    con = duckdb.connect(database=':memory:')
    con.execute(f"""
        CREATE OR REPLACE TABLE loan_data AS 
        SELECT * FROM read_parquet('{str(parquet_path)}')
    """)
    return con


def process_data_chunked(con, process_func, batch_size=10000):
    """Process data in chunks using DuckDB"""
    total_rows = con.execute("SELECT COUNT(*) FROM loan_data").fetchone()[0]
    results = []
    
    for offset in range(0, total_rows, batch_size):
        df_chunk = con.execute(f"""
            SELECT * FROM loan_data 
            ORDER BY rowid 
            LIMIT {batch_size} OFFSET {offset}
        """).fetchdf()
        
        if df_chunk.empty:
            break
            
        # Process the chunk
        result = process_func(df_chunk)
        if result is not None:
            results.append(result)
    
    # Combine results if needed
    if results and isinstance(results[0], pd.DataFrame):
        return pd.concat(results, ignore_index=True)
    return results


def train_model_chunked(con, X_cols, y_col, batch_size=10000, thorough_training=False):
    """Train model using chunked data"""
    # Get total rows for progress tracking
    total_rows = con.execute("SELECT COUNT(*) FROM loan_data").fetchone()[0]
    
    # Get feature types for preprocessing
    sample = con.execute(f"SELECT * FROM loan_data LIMIT 1").fetchdf()
    numeric_features = sample[X_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = list(set(X_cols) - set(numeric_features))
    
    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Initialize model
    if thorough_training:
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Train in chunks
        X_chunks = []
        y_chunks = []
        
        for offset in range(0, total_rows, batch_size):
            # Update progress
            progress = min((offset + batch_size) / total_rows, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {offset//batch_size + 1}...")
            
            # Get chunk
            query = f"""
                SELECT * FROM loan_data 
                ORDER BY rowid 
                LIMIT {batch_size} OFFSET {offset}
            """
            chunk = con.execute(query).fetchdf()
            
            if chunk.empty:
                break
                
            X_chunk = chunk[X_cols]
            y_chunk = chunk[y_col]
            
            if thorough_training:
                # For thorough training, collect all data first
                X_chunks.append(X_chunk)
                y_chunks.append(y_chunk)
            else:
                # For regular training, fit incrementally
                if offset == 0:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                    ])
                    model.fit(X_chunk, y_chunk)
                else:
                    # For models that support partial_fit
                    if hasattr(model, 'partial_fit'):
                        X_processed = preprocessor.transform(X_chunk)
                        model.partial_fit(X_processed, y_chunk, classes=[0, 1])
                    else:
                        # Retrain on accumulated data if partial_fit not available
                        X_all = pd.concat([X_all, X_chunk])
                        y_all = pd.concat([y_all, y_chunk])
                        model.fit(X_all, y_all)
        
        if thorough_training:
            # Combine all data for thorough training
            X_all = pd.concat(X_chunks)
            y_all = pd.concat(y_chunks)
            
            # Update status
            status_text.text("Performing thorough hyperparameter search...")
            
            # Define parameter grid for RandomizedSearchCV
            param_dist = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
                'classifier__min_child_samples': [20, 50, 100],
                'classifier__subsample': [0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Create pipeline for RandomizedSearchCV
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
            ])
            
            # Perform randomized search with cross-validation
            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=10,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            # Fit the model with progress updates
            search.fit(X_all, y_all)
            model = search.best_estimator_
            
            # Update status
            status_text.text(f"Best parameters: {search.best_params_}")
        
        return model
        
    finally:
        # Clean up progress bar
        progress_bar.empty()
        status_text.empty()


def parse_emp_length(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip()
    if val in {"nan", "none", "null", ""}:
        return np.nan
    if "< 1" in val or "less than 1" in val:
        return 0.0
    if "10+" in val or "more than 10" in val:
        return 10.0
    match = re.search(r"(\d+)", val)
    return float(match.group(1)) if match else np.nan


def parse_percentage(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        v = val.replace("%", "").strip()
        if v.lower() in {"nan", "none", "null", ""}:
            return np.nan
        try:
            return float(v)
        except ValueError:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def calculate_months_since(date_val, reference_date="2015-01-01"):
    try:
        dt = pd.to_datetime(date_val, errors="coerce")
        ref = pd.to_datetime(reference_date)
        if pd.isna(dt):
            return np.nan
        return float((ref - dt).days / 30.44)
    except Exception:
        return np.nan


def clean_dataframe(df):
    df = df.copy()

    if "outcome" in df.columns:
        def _map_outcome(x):
            s = str(x).strip().upper()
            if s == "DEFAULTED":
                return 1
            if s == "FULLY PAID":
                return 0
            return np.nan

        df["is_default"] = df["outcome"].apply(_map_outcome)
        df = df.drop("outcome", axis=1)

    if "term" in df.columns:
        term_series = df["term"].astype(str)
        df["term_months"] = term_series.str.extract(r"(\d+)")[0].astype(float)
        df = df.drop("term", axis=1)

    if "emp_length" in df.columns:
        df["emp_length_years"] = df["emp_length"].apply(parse_emp_length)
        df = df.drop("emp_length", axis=1)

    percentage_cols = ["revol_util", "bc_util", "percent_bc_gt_75", "all_util", "il_util"]
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_percentage)

    if "earliest_cr_line" in df.columns:
        df["credit_history_months"] = df["earliest_cr_line"].apply(
            lambda x: calculate_months_since(x, reference_date="2015-01-01")
        )
        df = df.drop("earliest_cr_line", axis=1)

    categorical_cols = ["home_ownership", "purpose", "addr_state", "emp_title"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace(["NAN", "NONE", "NULL", ""], np.nan)

    non_numeric_object_candidates = [
        c
        for c in df.columns
        if c not in set(categorical_cols + ["borrower_id"]) and df[c].dtype == object
    ]
    for col in non_numeric_object_candidates:
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue
        success_ratio = coerced.notna().sum() / non_null
        if success_ratio >= 0.8:
            df[col] = coerced

    if "emp_title" in df.columns:
        df = df.drop("emp_title", axis=1)

    return df


def _missingness_diff_table(X, y, min_missing_diff=0.05):
    if pd.Series(y).nunique() < 2:
        return pd.DataFrame()
    rows = []
    for col in X.columns:
        if col == "borrower_id":
            continue
        miss = X[col].isna()
        m0 = float(miss[y == 0].mean())
        m1 = float(miss[y == 1].mean())
        diff = m1 - m0
        if abs(diff) < min_missing_diff:
            continue
        rows.append(
            {
                "feature": col,
                "missing_paid": m0,
                "missing_default": m1,
                "missing_diff_default_minus_paid": diff,
                "abs_missing_diff": abs(diff),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("abs_missing_diff", ascending=False)


def engineer_fraud_features(df):
    df = df.copy()

    delinq_cols = ["delinq_2yrs", "num_tl_30dpd", "num_tl_90g_dpd_24m"]
    existing_delinq = [c for c in delinq_cols if c in df.columns]
    if existing_delinq:
        delinq_numeric = df[existing_delinq].apply(pd.to_numeric, errors="coerce")
        df["total_recent_delinq"] = delinq_numeric.sum(axis=1)
    else:
        df["total_recent_delinq"] = 0.0
    df["has_recent_delinq"] = (df["total_recent_delinq"].fillna(0) > 0).astype(int)

    if "num_tl_120dpd_2m" in df.columns:
        col = pd.to_numeric(df["num_tl_120dpd_2m"], errors="coerce")
        df["has_severe_delinq"] = (col.fillna(0) > 0).astype(int)

    if "chargeoff_within_12_mths" in df.columns:
        col = pd.to_numeric(df["chargeoff_within_12_mths"], errors="coerce")
        df["has_recent_chargeoff"] = (col.fillna(0) > 0).astype(int)

    if "pub_rec_bankruptcies" in df.columns:
        col = pd.to_numeric(df["pub_rec_bankruptcies"], errors="coerce")
        df["has_bankruptcy"] = (col.fillna(0) > 0).astype(int)

    if "tax_liens" in df.columns:
        col = pd.to_numeric(df["tax_liens"], errors="coerce")
        df["has_tax_lien"] = (col.fillna(0) > 0).astype(int)

    if "revol_util" in df.columns:
        col = pd.to_numeric(df["revol_util"], errors="coerce")
        df["revol_util_very_high"] = (col.fillna(0) > 80).astype(int)

    if "bc_util" in df.columns:
        col = pd.to_numeric(df["bc_util"], errors="coerce")
        df["bc_util_very_high"] = (col.fillna(0) > 80).astype(int)

    if "inq_last_6mths" in df.columns:
        col = pd.to_numeric(df["inq_last_6mths"], errors="coerce")
        df["many_recent_inquiries"] = (col.fillna(0) >= 3).astype(int)

    if "dti" in df.columns:
        col = pd.to_numeric(df["dti"], errors="coerce")
        df["dti_very_high"] = (col.fillna(0) > 30).astype(int)

    if "acc_now_delinq" in df.columns:
        col = pd.to_numeric(df["acc_now_delinq"], errors="coerce")
        df["currently_delinquent"] = (col.fillna(0) > 0).astype(int)

    if "pct_tl_nvr_dlq" in df.columns:
        col = pd.to_numeric(df["pct_tl_nvr_dlq"], errors="coerce")
        df["excellent_payment_history"] = (col.fillna(0) > 95).astype(int)

    if "tot_coll_amt" in df.columns and "annual_inc" in df.columns:
        tot_coll = pd.to_numeric(df["tot_coll_amt"], errors="coerce")
        inc = pd.to_numeric(df["annual_inc"], errors="coerce")
        df["collections_to_income"] = tot_coll / (inc + 1)

    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        loan = pd.to_numeric(df["loan_amnt"], errors="coerce")
        inc = pd.to_numeric(df["annual_inc"], errors="coerce")
        df["loan_to_income"] = loan / (inc + 1)

    return df


def build_pipeline(df_train):
    numeric_features = df_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ["is_default", "borrower_id"]]

    categorical_features = df_train.select_dtypes(include=["object"]).columns.tolist()
    categorical_features = [f for f in categorical_features if f not in ["is_default", "borrower_id"]]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                (
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames
                    else OneHotEncoder(handle_unknown="ignore", sparse=False)
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    return pipeline, numeric_features, categorical_features


def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "ROC_AUC": roc_auc_score(y_true, y_pred_proba),
        "PR_AUC": average_precision_score(y_true, y_pred_proba),
        "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "F1": f1_score(y_true, y_pred),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "False_Positive_Rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "Cost": float(fp * 100 + fn * 1000),
    }
    return metrics, y_pred


def find_optimal_threshold(y_true, y_pred_proba, metric="f1"):
    thresholds = np.arange(0.1, 0.91, 0.05)
    best_score = -np.inf
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "recall":
            score = recall_score(y_true, y_pred)
        elif metric == "precision":
            score = precision_score(y_true, y_pred)
        elif metric == "cost":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            score = -(fp * 100 + fn * 1000)
        else:
            score = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    return best_threshold


def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return output.getvalue()


@st.cache_data(show_spinner=False)
def load_excel(uploaded_file):
    return pd.read_excel(uploaded_file, engine="openpyxl")


def get_pipeline_feature_names(pipeline):
    try:
        pre = pipeline.named_steps["preprocessor"]
        return pre.get_feature_names_out().tolist()
    except Exception:
        return None


def ensure_required_columns(df, required_columns):
    df = df.copy()
    for c in required_columns:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _numeric_diff_table(X, y, min_non_null_per_class=100):
    if pd.Series(y).nunique() < 2:
        return pd.DataFrame()
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["borrower_id"]]
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(X[col], errors="coerce")
        s0 = s[y == 0]
        s1 = s[y == 1]
        if s0.notna().sum() < min_non_null_per_class or s1.notna().sum() < min_non_null_per_class:
            continue
        mean0 = float(s0.mean())
        mean1 = float(s1.mean())
        med0 = float(s0.median())
        med1 = float(s1.median())
        v0 = float(s0.var(ddof=0))
        v1 = float(s1.var(ddof=0))
        pooled = float(np.sqrt((v0 + v1) / 2))
        std_diff = (mean1 - mean0) / pooled if pooled and not np.isnan(pooled) else np.nan

        tmp = s.copy()
        if tmp.nunique(dropna=True) >= 2:
            tmp = tmp.fillna(tmp.median())
            auc = float(roc_auc_score(y, tmp))
            auc_strength = float(max(auc, 1 - auc))
        else:
            auc_strength = np.nan

        rows.append(
            {
                "feature": col,
                "mean_paid": mean0,
                "mean_default": mean1,
                "median_paid": med0,
                "median_default": med1,
                "std_diff_default_minus_paid": std_diff,
                "abs_std_diff": abs(std_diff) if not pd.isna(std_diff) else np.nan,
                "auc_strength": auc_strength,
                "direction": "higher_in_default" if mean1 > mean0 else "lower_in_default",
                "missing_pct": float(s.isna().mean()),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["abs_std_diff", "auc_strength"], ascending=[False, False])
    return df


def _categorical_risk_table(X, y, min_count=200, top_per_feature=5):
    if pd.Series(y).nunique() < 2:
        return pd.DataFrame()
    base_rate = float(y.mean())
    if base_rate <= 0:
        return pd.DataFrame()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["borrower_id"]]
    rows = []
    for col in cat_cols:
        s = X[col].astype(str).str.strip().str.upper()
        s = s.replace(["NAN", "NONE", "NULL", ""], np.nan)
        vc = s.value_counts(dropna=True)
        if vc.empty:
            continue
        candidates = vc[vc >= min_count].head(30)
        for cat_val, cnt in candidates.items():
            mask = s == cat_val
            rate = float(y[mask].mean()) if mask.any() else np.nan
            if pd.isna(rate):
                continue
            lift = (rate / base_rate) if base_rate > 0 else np.nan
            rows.append(
                {
                    "feature": col,
                    "category": cat_val,
                    "count": int(cnt),
                    "default_rate": rate,
                    "lift_vs_base": lift,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["lift_vs_base", "count"], ascending=[False, False])
    df = df.groupby("feature", as_index=False).head(top_per_feature)
    df = df.sort_values(["lift_vs_base", "count"], ascending=[False, False])
    return df


def _fraud_flag_table(X, y, min_count_flag1=50):
    if pd.Series(y).nunique() < 2:
        return pd.DataFrame()
    base_rate = float(y.mean())
    if base_rate <= 0:
        return pd.DataFrame()
    candidate_flags = [
        "has_recent_delinq",
        "has_severe_delinq",
        "has_recent_chargeoff",
        "has_bankruptcy",
        "has_tax_lien",
        "revol_util_very_high",
        "bc_util_very_high",
        "many_recent_inquiries",
        "dti_very_high",
        "currently_delinquent",
        "excellent_payment_history",
    ]
    rows = []
    for col in candidate_flags:
        if col not in X.columns:
            continue
        s = pd.to_numeric(X[col], errors="coerce")
        mask1 = s.fillna(0) == 1
        mask0 = s.fillna(0) == 0
        cnt1 = int(mask1.sum())
        if cnt1 < min_count_flag1:
            continue
        rate1 = float(y[mask1].mean())
        rate0 = float(y[mask0].mean()) if mask0.any() else np.nan
        lift = (rate1 / base_rate) if base_rate > 0 else np.nan
        rows.append(
            {
                "indicator": col,
                "count_flag_1": cnt1,
                "default_rate_flag_0": rate0,
                "default_rate_flag_1": rate1,
                "lift_vs_base": lift,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["lift_vs_base", "count_flag_1"], ascending=[False, False])
    return df


def _loan_amount_bucket_table(X, y_true, y_pred_proba, threshold, buckets=5):
    if "loan_amnt" not in X.columns:
        return pd.DataFrame()
    loan = pd.to_numeric(X["loan_amnt"], errors="coerce")
    mask = loan.notna()
    if mask.sum() < 200:
        return pd.DataFrame()
    try:
        bucket = pd.qcut(loan[mask], q=buckets, duplicates="drop")
    except Exception:
        return pd.DataFrame()

    y_pred = (y_pred_proba >= threshold).astype(int)
    df = pd.DataFrame(
        {
            "bucket": bucket.astype(str),
            "y_true": np.array(y_true)[mask],
            "y_pred": np.array(y_pred)[mask],
        }
    )
    out = (
        df.groupby("bucket")
        .apply(
            lambda g: pd.Series(
                {
                    "count": int(len(g)),
                    "default_rate": float(g["y_true"].mean()),
                    "recall": float(
                        (g["y_pred"].eq(1) & g["y_true"].eq(1)).sum()
                        / max(1, int(g["y_true"].eq(1).sum()))
                    ),
                    "precision": float(
                        (g["y_pred"].eq(1) & g["y_true"].eq(1)).sum()
                        / max(1, int(g["y_pred"].eq(1).sum()))
                    ),
                }
            )
        )
        .reset_index()
    )
    return out


def build_analysis_report(df_feat, y, categorical_min_count=200):
    X = df_feat.drop(columns=["is_default"], errors="ignore")
    analyzed_rows = int(len(y))
    base_rate = float(y.mean()) if analyzed_rows else np.nan
    overview = {
        "rows_in_file": int(len(df_feat)),
        "rows_analyzed": analyzed_rows,
        "num_features_after_cleaning": int(X.shape[1]),
        "num_defaulters": int((y == 1).sum()),
        "num_fully_paid": int((y == 0).sum()),
        "default_rate": base_rate,
    }

    missing = X.isna().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["feature", "missing_pct"]

    min_class = int(min(int((y == 0).sum()), int((y == 1).sum()))) if analyzed_rows else 0
    min_non_null = max(30, min(200, int(min_class * 0.2))) if min_class else 100
    numeric_diff = _numeric_diff_table(X, y, min_non_null_per_class=min_non_null)
    categorical_risk = _categorical_risk_table(X, y, min_count=categorical_min_count)
    fraud_flags = _fraud_flag_table(X, y)
    missing_diff = _missingness_diff_table(X, y)

    conclusions = []
    if analyzed_rows:
        conclusions.append(
            f"Analyzed {analyzed_rows:,} labeled loans. Default rate = {base_rate:.2%} ({int((y==1).sum()):,} defaults)."
        )

    if pd.Series(y).nunique() < 2:
        conclusions.append("Only one class present in labeled rows; defaulter-vs-payer comparisons are not available.")
        return {
            "overview": overview,
            "missing": missing,
            "missing_diff": missing_diff,
            "numeric_diff": numeric_diff,
            "categorical_risk": categorical_risk,
            "fraud_flags": fraud_flags,
            "conclusions": conclusions,
        }

    if not fraud_flags.empty:
        high_risk_flags = fraud_flags.sort_values("lift_vs_base", ascending=False).head(5)
        for _, r in high_risk_flags.iterrows():
            conclusions.append(
                f"Fraud indicator '{r['indicator']}' is high-risk: default rate {r['default_rate_flag_1']:.2%} vs {r['default_rate_flag_0']:.2%} when flag=0 (lift {r['lift_vs_base']:.2f}x, n={int(r['count_flag_1']):,})."
            )

        protective_flags = fraud_flags.sort_values("lift_vs_base", ascending=True).head(3)
        for _, r in protective_flags.iterrows():
            if pd.isna(r["lift_vs_base"]):
                continue
            conclusions.append(
                f"Indicator '{r['indicator']}' looks protective: default rate {r['default_rate_flag_1']:.2%} vs {r['default_rate_flag_0']:.2%} when flag=0 (lift {r['lift_vs_base']:.2f}x, n={int(r['count_flag_1']):,})."
            )

    if not numeric_diff.empty:
        top_num = numeric_diff.head(8)
        for _, r in top_num.iterrows():
            conclusions.append(
                f"Numeric driver '{r['feature']}' differs for defaulters: {r['direction']} (mean paid={_safe_float(r['mean_paid']):.3g}, mean default={_safe_float(r['mean_default']):.3g}, |std diff|={_safe_float(r['abs_std_diff']):.2f})."
            )

    if not categorical_risk.empty:
        top_cat = categorical_risk.head(10)
        for _, r in top_cat.iterrows():
            conclusions.append(
                f"Category risk: {r['feature']}='{r['category']}' default rate {r['default_rate']:.2%} (lift {r['lift_vs_base']:.2f}x, n={int(r['count']):,})."
            )

        protective_cat = categorical_risk.sort_values("lift_vs_base", ascending=True).head(5)
        for _, r in protective_cat.iterrows():
            if pd.isna(r["lift_vs_base"]):
                continue
            conclusions.append(
                f"Category looks protective: {r['feature']}='{r['category']}' default rate {r['default_rate']:.2%} (lift {r['lift_vs_base']:.2f}x, n={int(r['count']):,})."
            )

    return {
        "overview": overview,
        "missing": missing,
        "missing_diff": missing_diff,
        "numeric_diff": numeric_diff,
        "categorical_risk": categorical_risk,
        "fraud_flags": fraud_flags,
        "conclusions": conclusions,
    }


def analysis_report_to_excel_bytes(report):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([report["overview"]]).to_excel(writer, index=False, sheet_name="overview")
        report["missing"].to_excel(writer, index=False, sheet_name="missingness")
        report["missing_diff"].to_excel(writer, index=False, sheet_name="missingness_by_class")
        report["numeric_diff"].to_excel(writer, index=False, sheet_name="numeric_diff")
        report["categorical_risk"].to_excel(writer, index=False, sheet_name="categorical_risk")
        report["fraud_flags"].to_excel(writer, index=False, sheet_name="fraud_indicators")
        pd.DataFrame({"conclusion": report["conclusions"]}).to_excel(
            writer, index=False, sheet_name="conclusions"
        )
    return output.getvalue()


def main():
    st.set_page_config(page_title="Loan Default / Fraud Detection", layout="wide")
    st.title("Loan Default Prediction (First-Party Fraud Detection)")

    if "artifacts" not in st.session_state:
        st.session_state.artifacts = None

    with st.sidebar:
        st.header("Data & Model")

        train_file = st.file_uploader("Upload training data (Excel)", type=["xlsx"], key="train")
        score_file = st.file_uploader("Upload scoring data (Excel)", type=["xlsx"], key="score")

        st.divider()
        st.subheader("Threshold")
        threshold_mode = st.selectbox(
            "Threshold mode",
            options=["Optimize", "Manual"],
            index=0,
        )
        optimize_metric = st.selectbox(
            "Optimize metric",
            options=["recall", "f1", "precision", "cost"],
            index=0,
            disabled=threshold_mode != "Optimize",
        )
        manual_threshold = st.slider(
            "Manual threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            step=0.05,
            disabled=threshold_mode != "Manual",
        )

        st.divider()
        st.subheader("Training")
        training_mode = st.selectbox(
            "Training mode",
            options=["Quick", "Thorough (CV tuning)"],
            index=0,
        )
        tuning_iterations = st.slider(
            "Tuning iterations (Thorough mode)",
            min_value=5,
            max_value=40,
            value=15,
            step=5,
            disabled=training_mode != "Thorough (CV tuning)",
        )

        st.divider()
        st.subheader("Load / Save model")
        uploaded_model = st.file_uploader("Upload a saved model (.joblib)", type=["joblib"], key="model")
        if uploaded_model is not None:
            try:
                artifacts = joblib.load(uploaded_model)
                st.session_state.artifacts = artifacts
                st.success("Model loaded")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

        if st.session_state.artifacts is not None:
            model_bytes = BytesIO()
            joblib.dump(st.session_state.artifacts, model_bytes)
            st.download_button(
                "Download current model",
                data=model_bytes.getvalue(),
                file_name="fraud_detection_model.joblib",
                mime="application/octet-stream",
            )

    tab_train, tab_score, tab_insights = st.tabs(["Train Model", "Score New Data", "Model Insights"])

    with tab_train:
        st.subheader("Training")
        if train_file is None:
            st.info("Upload training data in the sidebar to begin.")
        else:
            df_raw = load_excel(train_file)
            st.write("Preview")
            st.dataframe(df_raw.head(20), use_container_width=True)

            if "outcome" not in df_raw.columns:
                st.error("Training file must contain an 'outcome' column with values 'Fully Paid' or 'Defaulted'.")
            else:
                df_clean = clean_dataframe(df_raw)
                df_feat = engineer_fraud_features(df_clean)

                if "is_default" not in df_feat.columns:
                    st.error("Could not create target column 'is_default'. Check 'outcome' values.")
                    return

                y = df_feat["is_default"].astype(float)
                mask = y.notna()
                df_feat = df_feat.loc[mask].reset_index(drop=True)
                y = y.loc[mask].astype(int).reset_index(drop=True)

                class_counts = y.value_counts().rename({0: "Fully Paid", 1: "Defaulted"}).reset_index()
                class_counts.columns = ["class", "count"]
                fig_class = px.bar(class_counts, x="class", y="count", title="Class distribution")
                st.plotly_chart(fig_class, use_container_width=True)

                st.markdown("### Data analysis & fraud indicators")
                cat_min = st.slider(
                    "Minimum category count (for categorical risk analysis)",
                    min_value=25,
                    max_value=1000,
                    value=200,
                    step=25,
                )
                report = build_analysis_report(df_feat, y, categorical_min_count=int(cat_min))
                ov = report["overview"]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows analyzed", f"{ov['rows_analyzed']:,}")
                c2.metric("Defaulters", f"{ov['num_defaulters']:,}")
                c3.metric("Fully Paid", f"{ov['num_fully_paid']:,}")
                c4.metric("Default rate", f"{ov['default_rate']:.2%}")

                with st.expander("Conclusions (auto-generated)", expanded=True):
                    if report["conclusions"]:
                        for line in report["conclusions"][:80]:
                            st.write(f"- {line}")
                    else:
                        st.info("Not enough labeled rows to generate conclusions.")

                with st.expander("Missingness (top 30 columns)"):
                    st.dataframe(report["missing"].head(30), use_container_width=True)

                with st.expander("Missingness differences: defaulters vs payers"):
                    if report["missing_diff"].empty:
                        st.info("No large missingness differences found.")
                    else:
                        st.dataframe(report["missing_diff"].head(50), use_container_width=True)

                with st.expander("Fraud indicators (engineered flags)"):
                    if report["fraud_flags"].empty:
                        st.info("No engineered fraud indicator columns found in this dataset.")
                    else:
                        st.dataframe(report["fraud_flags"].head(30), use_container_width=True)
                        fig_flags = px.bar(
                            report["fraud_flags"].head(15),
                            x="lift_vs_base",
                            y="indicator",
                            orientation="h",
                            title="Engineered fraud indicators (lift vs base default rate)",
                        )
                        st.plotly_chart(fig_flags, use_container_width=True)

                with st.expander("Numeric differences: defaulters vs payers"):
                    if report["numeric_diff"].empty:
                        st.info("Not enough numeric columns with sufficient non-missing values.")
                    else:
                        st.dataframe(report["numeric_diff"].head(50), use_container_width=True)
                        top_feature = report["numeric_diff"].iloc[0]["feature"]
                        plot_df = pd.DataFrame(
                            {
                                "value": pd.to_numeric(df_feat[top_feature], errors="coerce"),
                                "class": np.where(y.values == 1, "Defaulted", "Fully Paid"),
                            }
                        )
                        plot_df = plot_df.dropna()
                        if len(plot_df) > 0:
                            fig_box = px.box(plot_df, x="class", y="value", title=f"{top_feature} distribution")
                            st.plotly_chart(fig_box, use_container_width=True)

                with st.expander("Categorical differences: high-risk categories"):
                    if report["categorical_risk"].empty:
                        st.info("No categorical columns with categories above the minimum count.")
                    else:
                        st.dataframe(report["categorical_risk"].head(50), use_container_width=True)

                analysis_bytes = analysis_report_to_excel_bytes(report)
                st.download_button(
                    "Download analysis report (Excel)",
                    data=analysis_bytes,
                    file_name="training_data_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                train_btn = st.button("Train model", type="primary")

                if train_btn:
                    if y.nunique() < 2:
                        st.error("Need both classes present in training data.")
                        return

                    X = df_feat.drop(columns=["is_default"], errors="ignore")

                    X_train, X_val, y_train, y_val = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42,
                        stratify=y,
                    )

                    pipeline, numeric_features, categorical_features = build_pipeline(df_feat)

                    progress = st.progress(0)
                    status = st.empty()

                    status.write("Fitting model...")
                    if training_mode == "Thorough (CV tuning)":
                        status.write("Hyperparameter tuning (this can take a few minutes)...")
                        param_distributions = {
                            "classifier__n_estimators": [200, 300, 500, 800],
                            "classifier__learning_rate": [0.03, 0.05, 0.07],
                            "classifier__max_depth": [-1, 4, 6, 8],
                            "classifier__num_leaves": [15, 31, 63, 127],
                            "classifier__min_child_samples": [10, 20, 50],
                            "classifier__subsample": [0.8, 1.0],
                            "classifier__colsample_bytree": [0.8, 1.0],
                            "classifier__reg_alpha": [0.0, 0.1, 0.5],
                            "classifier__reg_lambda": [0.0, 0.1, 0.5],
                        }

                        search = RandomizedSearchCV(
                            estimator=pipeline,
                            param_distributions=param_distributions,
                            n_iter=int(tuning_iterations),
                            scoring="average_precision",
                            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                            n_jobs=-1,
                            verbose=0,
                            random_state=42,
                        )
                        search.fit(X_train, y_train)
                        pipeline = search.best_estimator_
                        progress.progress(60)
                    else:
                        pipeline.fit(X_train, y_train)
                    progress.progress(60)

                    status.write("Evaluating...")
                    y_proba_val = pipeline.predict_proba(X_val)[:, 1]

                    if threshold_mode == "Optimize":
                        threshold = find_optimal_threshold(y_val, y_proba_val, metric=optimize_metric)
                    else:
                        threshold = float(manual_threshold)

                    metrics, y_pred_val = evaluate_model(y_val, y_proba_val, threshold=threshold)
                    progress.progress(85)

                    feature_names = get_pipeline_feature_names(pipeline)

                    cv_auc = None
                    cv_pr_auc = None
                    try:
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        aucs = []
                        prs = []
                        for tr_idx, te_idx in cv.split(X, y):
                            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
                            p, _, _ = build_pipeline(df_feat)
                            p.fit(X_tr, y_tr)
                            proba = p.predict_proba(X_te)[:, 1]
                            aucs.append(roc_auc_score(y_te, proba))
                            prs.append(average_precision_score(y_te, proba))
                        cv_auc = float(np.mean(aucs))
                        cv_pr_auc = float(np.mean(prs))
                    except Exception:
                        cv_auc = None
                        cv_pr_auc = None

                    artifacts = {
                        "pipeline": pipeline,
                        "threshold": threshold,
                        "feature_names": feature_names,
                        "numeric_features": numeric_features,
                        "categorical_features": categorical_features,
                        "training_date": datetime.now(),
                        "performance_metrics": metrics,
                        "cv_roc_auc_mean": cv_auc,
                        "cv_pr_auc_mean": cv_pr_auc,
                        "required_columns": numeric_features + categorical_features,
                    }
                    
                    # Display loading state
                    with st.spinner('Loading and processing data (this may take a while)...'):
                        # Load data using memory-efficient method
                        con = load_data(train_file)
                        
                        # Get column names and sample data for display
                        columns = con.execute("DESCRIBE loan_data").fetchdf()['column_name'].tolist()
                        sample_data = con.execute("SELECT * FROM loan_data LIMIT 5").fetchdf()
                        
                        # Clean and engineer features in chunks
                        def process_chunk(chunk):
                            chunk = clean_dataframe(chunk)
                            return engineer_fraud_features(chunk)
                        
                        # Process data in chunks
                        processed_data = process_data_chunked(con, process_chunk)
                        
                        # Get the updated column names after feature engineering
                        sample_processed = process_chunk(sample_data)
                        
                        # Display data info
                        st.subheader("Data Preview")
                        st.write(f"Shape: {con.execute('SELECT COUNT(*) FROM loan_data').fetchone()[0]} rows, {len(sample_processed.columns)} columns")
                        st.dataframe(sample_processed)
                        
                        # Model training options
                        st.subheader("2. Model Training Options")
                        thorough_training = st.checkbox("Enable thorough training (slower but more accurate)", 
                                                     help="Uses RandomizedSearchCV for hyperparameter tuning")
                        
                        # Train button
                        train_btn = st.button("Train Model")
                        
                        if train_btn:
                            with st.spinner('Training model (this may take a while)...'):
                                try:
                                    # Get column names for features and target
                                    y_col = 'is_default'
                                    X_cols = [c for c in sample_processed.columns if c not in [y_col, 'borrower_id']]
                                    
                                    # Train model using chunked data
                                    model = train_model_chunked(
                                        con, 
                                        X_cols, 
                                        y_col, 
                                        thorough_training=thorough_training
                                    )
                                    
                                    # Save model
                                    joblib.dump(model, 'fraud_detection_model.joblib')
                                    st.success('Model trained and saved successfully!')
                                    
                                    # Display model info
                                    if hasattr(model, 'best_params_') and thorough_training:
                                        st.subheader("Best Parameters from Tuning")
                                        st.json(model.best_params_)
                                    
                                except Exception as e:
                                    st.error(f"Error during training: {str(e)}")
                                    st.exception(e)
                                finally:
                                    # Clean up DuckDB connection
                                    if 'con' in locals():
                                        con.close()
                            
                            # Get predictions for validation set if available
                            if 'X_val' in globals() and 'y_val' in globals() and hasattr(model, 'predict_proba'):
                                try:
                                    y_proba_val = model.predict_proba(X_val)[:, 1]
                                    
                                    # Plot precision-recall curve
                                    try:
                                        prec, rec, _ = precision_recall_curve(y_val, y_proba_val)
                                        fig_pr = go.Figure()
                                        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
                                        fig_pr.update_layout(
                                            title="Precision-Recall Curve (Validation)",
                                            xaxis_title="Recall",
                                            yaxis_title="Precision",
                                            height=380,
                                        )
                                        st.plotly_chart(fig_pr, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not generate precision-recall curve: {str(e)}")
                                    
                                    # Show performance by loan amount buckets if data is available
                                    try:
                                        with st.expander("Performance by loan amount buckets"):
                                            if all(col in X_val.columns for col in ['loan_amnt']):
                                                bucket_df = _loan_amount_bucket_table(X_val, y_val, y_proba_val, threshold=0.5, buckets=5)
                                                if not bucket_df.empty:
                                                    st.dataframe(bucket_df, use_container_width=True)
                                                else:
                                                    st.info("Loan amount buckets not available (missing loan_amnt or not enough data).")
                                            else:
                                                st.info("Loan amount data not available for bucketing.")
                                    except Exception as e:
                                        st.warning(f"Could not generate loan amount buckets: {str(e)}")
                                    
                                    # Show feature importances if available
                                    if hasattr(model, 'feature_importances_') and 'feature_names' in globals() and feature_names is not None:
                                        try:
                                            importances = model.feature_importances_
                                            fi = pd.DataFrame({"feature": feature_names, "importance": importances})
                                            fi = fi.sort_values("importance", ascending=False).head(30)
                                            fig_fi = px.bar(fi, x="importance", y="feature", 
                                                          orientation="h", 
                                                          title="Top feature importances")
                                            st.plotly_chart(fig_fi, use_container_width=True)
                                        except Exception as e:
                                            st.warning(f"Could not generate feature importance plot: {str(e)}")
                                    
                                except Exception as e:
                                    st.warning(f"Error during model evaluation: {str(e)}")
                            else:
                                st.info("Validation data not available for generating evaluation plots.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)

    with tab_score:
        st.subheader("Scoring")
        artifacts = st.session_state.artifacts
        if artifacts is None:
            st.info("Train a model or upload a saved model in the sidebar.")
        elif score_file is None:
            st.info("Upload scoring data in the sidebar.")
        else:
            df_score_raw = load_excel(score_file)
            st.write("Preview")
            st.dataframe(df_score_raw.head(20), use_container_width=True)

            score_btn = st.button("Score data", type="primary")

            if score_btn:
                pipeline = artifacts["pipeline"]
                threshold = float(artifacts.get("threshold", 0.35))
                required_columns = artifacts.get("required_columns")

                df_score_clean = clean_dataframe(df_score_raw)
                df_score_feat = engineer_fraud_features(df_score_clean)

                borrower_id = df_score_feat["borrower_id"] if "borrower_id" in df_score_feat.columns else None

                X_score = df_score_feat.copy()
                if "is_default" in X_score.columns:
                    X_score = X_score.drop(columns=["is_default"], errors="ignore")

                if required_columns is not None:
                    X_score = ensure_required_columns(X_score, required_columns)

                try:
                    proba = pipeline.predict_proba(X_score)[:, 1]
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                    return

                pred = (proba >= threshold).astype(int)
                predicted_outcome = np.where(pred == 1, "Defaulted", "Fully Paid")

                risk_category = pd.cut(
                    proba,
                    bins=[-np.inf, 0.3, 0.6, np.inf],
                    labels=["Low Risk", "Medium Risk", "High Risk"],
                ).astype(str)

                output = df_score_raw.copy()
                if "borrower_id" not in output.columns:
                    output["borrower_id"] = borrower_id if borrower_id is not None else np.arange(len(proba))
                output["predicted_outcome"] = predicted_outcome
                output["predicted_probability_default"] = proba
                output["risk_category"] = risk_category

                st.markdown("### Prediction distribution")
                fig_hist = px.histogram(
                    output,
                    x="predicted_probability_default",
                    nbins=30,
                    color="risk_category",
                    title="Risk score distribution",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("### Results")
                st.dataframe(output.head(50), use_container_width=True)

                excel_bytes = to_excel_bytes(output)
                st.download_button(
                    "Download predictions (Excel)",
                    data=excel_bytes,
                    file_name="loan_default_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with tab_insights:
        st.subheader("Insights")
        artifacts = st.session_state.artifacts
        if artifacts is None:
            st.info("Train a model or upload a saved model to view insights.")
        else:
            st.write("Model summary")
            summary = {
                "Training date": artifacts.get("training_date"),
                "Threshold": artifacts.get("threshold"),
                "ROC_AUC": (artifacts.get("performance_metrics") or {}).get("ROC_AUC"),
                "PR_AUC": (artifacts.get("performance_metrics") or {}).get("PR_AUC"),
                "Recall": (artifacts.get("performance_metrics") or {}).get("Recall"),
                "Precision": (artifacts.get("performance_metrics") or {}).get("Precision"),
                "F1": (artifacts.get("performance_metrics") or {}).get("F1"),
                "CV_ROC_AUC_Mean": artifacts.get("cv_roc_auc_mean"),
            }
            st.dataframe(pd.DataFrame([summary]), use_container_width=True)

            pipeline = artifacts["pipeline"]
            feature_names = artifacts.get("feature_names")

            if feature_names is not None:
                try:
                    importances = pipeline.named_steps["classifier"].feature_importances_
                    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
                    fi = fi.sort_values("importance", ascending=False)
                    top = fi.head(20)
                    fig = px.bar(top, x="importance", y="feature", orientation="h", title="Top fraud indicators")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Feature importance not available.")


if __name__ == "__main__":
    main()
