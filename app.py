import re
import warnings
from datetime import datetime
from io import BytesIO

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


warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


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


def clean_dataframe(df, copy=True):
    df = df.copy() if copy else df

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

    for col in categorical_cols:
        if col in df.columns:
            try:
                df[col] = df[col].astype("category")
            except Exception:
                pass

    for col in df.select_dtypes(include=["float64"]).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
        except Exception:
            pass

    for col in df.select_dtypes(include=["int64"]).columns:
        if col == "borrower_id":
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore", downcast="integer")
        except Exception:
            pass

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


def engineer_fraud_features(df, copy=True):
    df = df.copy() if copy else df

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

    if "term_months" in df.columns:
        term = pd.to_numeric(df["term_months"], errors="coerce")
        df["term_is_60"] = (term.fillna(0) >= 60).astype(int)

    if "home_ownership" in df.columns:
        ho = df["home_ownership"].astype(str).str.strip().str.upper()
        df["home_ownership_is_rent"] = (ho == "RENT").astype(int)

    if "purpose" in df.columns:
        purpose = df["purpose"].astype(str).str.strip().str.upper()
        df["purpose_is_house"] = (purpose == "HOUSE").astype(int)
        df["purpose_is_small_business"] = (purpose == "SMALL_BUSINESS").astype(int)
        df["purpose_is_debt_consolidation"] = (purpose == "DEBT_CONSOLIDATION").astype(int)

    if "addr_state" in df.columns:
        st_code = df["addr_state"].astype(str).str.strip().str.upper()
        high_risk_states = {"AZ", "PA", "NM", "NV", "UT"}
        df["high_risk_state"] = st_code.isin(high_risk_states).astype(int)

    velocity_cols = [
        "acc_open_past_24mths",
        "num_tl_op_past_12m",
        "open_rv_24m",
        "inq_last_12m",
        "inq_last_6mths",
    ]
    present_velocity = [c for c in velocity_cols if c in df.columns]
    if present_velocity:
        parts = []
        for c in present_velocity:
            s = pd.to_numeric(df[c], errors="coerce")
            if c == "acc_open_past_24mths":
                parts.append(s / 24.0)
            elif c == "num_tl_op_past_12m":
                parts.append(s / 12.0)
            elif c == "open_rv_24m":
                parts.append(s / 24.0)
            elif c == "inq_last_12m":
                parts.append(s / 12.0)
            elif c == "inq_last_6mths":
                parts.append(s / 6.0)
            else:
                parts.append(s)
        df["credit_velocity_score"] = pd.concat(parts, axis=1).sum(axis=1).astype(np.float32)

    if "mo_sin_rcnt_tl" in df.columns:
        mo_recent = pd.to_numeric(df["mo_sin_rcnt_tl"], errors="coerce")
        df["recent_trade_within_6m"] = (mo_recent.fillna(9999) <= 6).astype(int)

    util_cols = ["revol_util", "bc_util", "percent_bc_gt_75", "all_util", "il_util"]
    present_util = [c for c in util_cols if c in df.columns]
    if present_util:
        util_parts = [pd.to_numeric(df[c], errors="coerce") for c in present_util]
        util_mean = pd.concat(util_parts, axis=1).mean(axis=1)
        df["utilization_stress"] = (util_mean / 100.0).astype(np.float32)

    if "emp_length_years" in df.columns:
        emp = pd.to_numeric(df["emp_length_years"], errors="coerce")
        df["employment_lt_1yr"] = (emp.fillna(999) < 1).astype(int)
        df["employment_ge_10yr"] = (emp.fillna(0) >= 10).astype(int)

    if "credit_history_months" in df.columns:
        ch = pd.to_numeric(df["credit_history_months"], errors="coerce")
        df["credit_history_short"] = (ch.fillna(999999) < 60).astype(int)
        df["credit_history_long"] = (ch.fillna(0) >= 120).astype(int)

    if "avg_cur_bal" in df.columns:
        bal = pd.to_numeric(df["avg_cur_bal"], errors="coerce")
        df["avg_cur_bal_low"] = (bal.fillna(np.inf) < 12000).astype(int)

    if "tot_hi_cred_lim" in df.columns:
        lim = pd.to_numeric(df["tot_hi_cred_lim"], errors="coerce")
        df["tot_hi_cred_lim_low"] = (lim.fillna(np.inf) < 170000).astype(int)

    stability_parts = []
    if "employment_ge_10yr" in df.columns:
        stability_parts.append(pd.to_numeric(df["employment_ge_10yr"], errors="coerce"))
    if "credit_history_long" in df.columns:
        stability_parts.append(pd.to_numeric(df["credit_history_long"], errors="coerce"))
    if "home_ownership_is_rent" in df.columns:
        stability_parts.append(1 - pd.to_numeric(df["home_ownership_is_rent"], errors="coerce"))
    if stability_parts:
        df["stability_index"] = pd.concat(stability_parts, axis=1).sum(axis=1).astype(np.float32)

    if "mo_sin_rcnt_tl" in df.columns:
        mo_recent = pd.to_numeric(df["mo_sin_rcnt_tl"], errors="coerce")
        df["recent_trade_freshness"] = (1.0 / (1.0 + mo_recent.clip(lower=0))).astype(np.float32)

    if "tot_hi_cred_lim" in df.columns and "loan_amnt" in df.columns:
        lim = pd.to_numeric(df["tot_hi_cred_lim"], errors="coerce")
        loan = pd.to_numeric(df["loan_amnt"], errors="coerce")
        df["credit_limit_to_loan"] = (lim / (loan + 1.0)).astype(np.float32)

    if "avg_cur_bal" in df.columns and "annual_inc" in df.columns:
        bal = pd.to_numeric(df["avg_cur_bal"], errors="coerce")
        inc = pd.to_numeric(df["annual_inc"], errors="coerce")
        df["avg_balance_to_income"] = (bal / (inc + 1.0)).astype(np.float32)

    if "credit_velocity_score" in df.columns and "dti" in df.columns:
        vel = pd.to_numeric(df["credit_velocity_score"], errors="coerce")
        dti = pd.to_numeric(df["dti"], errors="coerce")
        df["velocity_x_dti"] = (vel * (dti / 100.0)).astype(np.float32)

    if "acc_open_past_24mths" in df.columns:
        s = pd.to_numeric(df["acc_open_past_24mths"], errors="coerce")
        df["acc_open_past_24mths_high"] = (s.fillna(0) >= 7).astype(int)

    if "num_tl_op_past_12m" in df.columns:
        s = pd.to_numeric(df["num_tl_op_past_12m"], errors="coerce")
        df["num_tl_op_past_12m_high"] = (s.fillna(0) >= 3).astype(int)

    if "inq_last_12m" in df.columns:
        s = pd.to_numeric(df["inq_last_12m"], errors="coerce")
        df["inq_last_12m_high"] = (s.fillna(0) >= 3).astype(int)

    if "open_rv_24m" in df.columns:
        s = pd.to_numeric(df["open_rv_24m"], errors="coerce")
        df["open_rv_24m_high"] = (s.fillna(0) >= 4).astype(int)

    risk_flag_cols = [
        "term_is_60",
        "home_ownership_is_rent",
        "purpose_is_house",
        "purpose_is_small_business",
        "high_risk_state",
        "employment_lt_1yr",
        "recent_trade_within_6m",
        "dti_very_high",
        "many_recent_inquiries",
        "inq_last_12m_high",
        "revol_util_very_high",
        "bc_util_very_high",
        "avg_cur_bal_low",
        "tot_hi_cred_lim_low",
        "has_recent_delinq",
        "has_severe_delinq",
        "has_recent_chargeoff",
        "has_bankruptcy",
        "has_tax_lien",
        "currently_delinquent",
    ]
    present_flags = [c for c in risk_flag_cols if c in df.columns]
    if present_flags:
        flag_mat = df[present_flags].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["risk_score_heuristic"] = flag_mat.sum(axis=1).astype(np.float32)
        df["risk_score_heuristic_norm"] = (df["risk_score_heuristic"] / float(len(present_flags))).astype(np.float32)

    return df


def build_pipeline(df_train):
    numeric_features = df_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ["is_default", "borrower_id"]]

    categorical_features = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
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
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
                    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames
                    else OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
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
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )

    model = LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        n_jobs=1,
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
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
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
        tuning_jobs = st.slider(
            "Tuning parallel jobs (Thorough mode)",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
            disabled=training_mode != "Thorough (CV tuning)",
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
            st.dataframe(df_raw.head(20), width="stretch")

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
                st.plotly_chart(fig_class, width="stretch")

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
                    st.dataframe(report["missing"].head(30), width="stretch")

                with st.expander("Missingness differences: defaulters vs payers"):
                    if report["missing_diff"].empty:
                        st.info("No large missingness differences found.")
                    else:
                        st.dataframe(report["missing_diff"].head(50), width="stretch")

                with st.expander("Fraud indicators (engineered flags)"):
                    if report["fraud_flags"].empty:
                        st.info("No engineered fraud indicator columns found in this dataset.")
                    else:
                        st.dataframe(report["fraud_flags"].head(30), width="stretch")
                        fig_flags = px.bar(
                            report["fraud_flags"].head(15),
                            x="lift_vs_base",
                            y="indicator",
                            orientation="h",
                            title="Engineered fraud indicators (lift vs base default rate)",
                        )
                        st.plotly_chart(fig_flags, width="stretch")

                with st.expander("Numeric differences: defaulters vs payers"):
                    if report["numeric_diff"].empty:
                        st.info("Not enough numeric columns with sufficient non-missing values.")
                    else:
                        st.dataframe(report["numeric_diff"].head(50), width="stretch")
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
                            st.plotly_chart(fig_box, width="stretch")

                with st.expander("Categorical differences: high-risk categories"):
                    if report["categorical_risk"].empty:
                        st.info("No categorical columns with categories above the minimum count.")
                    else:
                        st.dataframe(report["categorical_risk"].head(50), width="stretch")

                analysis_bytes = analysis_report_to_excel_bytes(report)
                st.download_button(
                    "Download analysis report (Excel)",
                    data=analysis_bytes,
                    file_name="training_data_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                train_btn = st.button("Train model", type="primary")

                if train_btn:
                    try:
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
                        search = None
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
                                scoring={"ap": "average_precision", "roc": "roc_auc"},
                                refit="ap",
                                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                n_jobs=int(tuning_jobs),
                                pre_dispatch=1,
                                return_train_score=False,
                                verbose=0,
                                random_state=42,
                            )
                            search.fit(X_train, y_train)
                            pipeline = search.best_estimator_
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
                        best_params = None
                        if training_mode == "Thorough (CV tuning)" and search is not None:
                            try:
                                cv_pr_auc = float(search.best_score_)
                                cv_auc = float(search.cv_results_["mean_test_roc"][search.best_index_])
                                best_params = search.best_params_
                            except Exception:
                                cv_auc = None
                                cv_pr_auc = None
                                best_params = None
                        else:
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
                            "best_params": best_params,
                            "required_columns": numeric_features + categorical_features,
                        }

                        st.session_state.artifacts = artifacts
                        try:
                            joblib.dump(artifacts, "fraud_detection_model.joblib")
                        except Exception:
                            pass

                        progress.progress(100)
                        status.write("Done")
                    except MemoryError:
                        st.error(
                            "Training ran out of memory. Try lowering 'Tuning parallel jobs' to 1 (already default), "
                            "or use 'Quick' mode once to validate the pipeline, then retry tuning."
                        )
                        return
                    except Exception as e:
                        st.error(f"Training failed: {type(e).__name__}: {e}")
                        return

                    st.markdown("### Results")
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.write("Metrics")
                        metrics_table = pd.DataFrame([metrics])
                        if cv_auc is not None:
                            metrics_table["CV_ROC_AUC_Mean"] = cv_auc
                        if cv_pr_auc is not None:
                            metrics_table["CV_PR_AUC_Mean"] = cv_pr_auc
                        metrics_table["Threshold"] = threshold
                        st.dataframe(metrics_table, width="stretch")

                    with col2:
                        cm = confusion_matrix(y_val, y_pred_val)
                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual Fully Paid", "Actual Defaulted"],
                            columns=["Pred Fully Paid", "Pred Defaulted"],
                        )
                        fig_cm = px.imshow(
                            cm_df,
                            text_auto=True,
                            title="Confusion Matrix (Validation)",
                            color_continuous_scale="Blues",
                        )
                        st.plotly_chart(fig_cm, width="stretch")

                    with st.expander("ROC & Precision-Recall curves"):
                        try:
                            fpr, tpr, _ = roc_curve(y_val, y_proba_val)
                            fig_roc = go.Figure()
                            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                            fig_roc.add_trace(
                                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
                            )
                            fig_roc.update_layout(
                                title="ROC Curve (Validation)",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate",
                                height=380,
                            )
                            st.plotly_chart(fig_roc, width="stretch")

                            prec, rec, _ = precision_recall_curve(y_val, y_proba_val)
                            fig_pr = go.Figure()
                            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
                            fig_pr.update_layout(
                                title="Precision-Recall Curve (Validation)",
                                xaxis_title="Recall",
                                yaxis_title="Precision",
                                height=380,
                            )
                            st.plotly_chart(fig_pr, width="stretch")
                        except Exception as e:
                            st.info(f"Could not plot curves: {e}")

                    with st.expander("Performance by loan amount buckets"):
                        bucket_df = _loan_amount_bucket_table(X_val, y_val, y_proba_val, threshold=threshold, buckets=5)
                        if bucket_df.empty:
                            st.info("Loan amount buckets not available (missing loan_amnt or not enough data).")
                        else:
                            st.dataframe(bucket_df, width="stretch")

                    if feature_names is not None:
                        try:
                            importances = pipeline.named_steps["classifier"].feature_importances_
                            fi = pd.DataFrame({"feature": feature_names, "importance": importances})
                            fi = fi.sort_values("importance", ascending=False).head(30)
                            fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", title="Top feature importances")
                            st.plotly_chart(fig_fi, width="stretch")
                        except Exception:
                            pass

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
            st.dataframe(df_score_raw.head(20), width="stretch")

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
                st.plotly_chart(fig_hist, width="stretch")

                st.markdown("### Results")
                st.dataframe(output.head(50), width="stretch")

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
            st.dataframe(pd.DataFrame([summary]), width="stretch")
            

            pipeline = artifacts["pipeline"]
            feature_names = artifacts.get("feature_names")

            if feature_names is not None:
                try:
                    importances = pipeline.named_steps["classifier"].feature_importances_
                    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
                    fi = fi.sort_values("importance", ascending=False)
                    top = fi.head(20)
                    fig = px.bar(top, x="importance", y="feature", orientation="h", title="Top fraud indicators")
                    st.plotly_chart(fig, width="stretch")
                except Exception:
                    st.info("Feature importance not available.")


if __name__ == "__main__":
    main()
