
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import shap  # type: ignore
except Exception:
    shap = None

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "AB_NYC_2019.csv"
ARTIFACTS_DIR = APP_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
PLOT_DIR = ARTIFACTS_DIR / "plots"

st.set_page_config(page_title="NYC Airbnb Model Studio", layout="wide")

# === Assignment-configurable variables ===
DATASET_DESCRIPTION = (
    "NYC Airbnb listings dataset with property, location, and host attributes."
)
TARGET_COLUMN = "price"
TASK_TYPE = "regression"
DATASET_SOURCE_DESCRIPTION = (
    "NYC Airbnb Open Data (AB_NYC_2019) compiled from Inside Airbnb listings."
)
PREDICTION_TASK_DESCRIPTION = (
    "Predict nightly listing price to support pricing strategy and market analysis."
)
WHY_IT_MATTERS = (
    "Accurate price prediction helps hosts and platforms set competitive rates and improve market efficiency."
)
APPROACH_SUMMARY = (
    "We profile the data, compare multiple regression models, and explain tree-based models with SHAP."
)
KEY_FINDINGS_SUMMARY = (
    "Tree-based ensemble models typically perform best while highlighting location, room type, and host attributes."
)

DROP_COLS = ["id", "name", "host_id", "host_name"]
DATE_COL = "last_review"
RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_SPLITS = 2
TREE_MODEL_NAMES = {"CART / Decision Tree", "Random Forest", "LightGBM"}

MODEL_DISPLAY_ORDER = [
    "Linear Regression",
    "Lasso Regression",
    "Ridge Regression",
    "CART / Decision Tree",
    "Random Forest",
    "LightGBM",
]

MODEL_ALIASES = {
    "Decision Tree": "CART / Decision Tree",
    "CART": "CART / Decision Tree",
    "DecisionTree": "CART / Decision Tree",
    "CART / Decision Tree": "CART / Decision Tree",
    "Linear Regression": "Linear Regression",
    "Lasso Regression": "Lasso Regression",
    "Ridge Regression": "Ridge Regression",
    "Random Forest": "Random Forest",
    "LightGBM": "LightGBM",
}

BEST_PARAM_KEYS = {
    "Lasso Regression": ["alpha"],
    "Ridge Regression": ["alpha"],
    "CART / Decision Tree": ["max_depth", "min_samples_leaf", "min_samples_split"],
    "Random Forest": ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split"],
    "LightGBM": ["n_estimators", "max_depth", "learning_rate", "num_leaves"],
}

METADATA_FILES = {
    "manifest": "model_manifest.json",
    "metrics_summary": "metrics_summary.json",
    "per_model": "per_model_metrics.json",
    "cv_results": "cross_validation_results.json",
    "best_params": "best_params.json",
    "predictions": "predictions.json",
    "feature_defaults": "feature_defaults.json",
    "feature_columns": "feature_columns.json",
    "feature_importances": "feature_importances.json",
    "summary": "summary.json",
}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        max_date = df[DATE_COL].max()
        if pd.notna(max_date):
            df["days_since_last_review"] = (max_date - df[DATE_COL]).dt.days
        else:
            df["days_since_last_review"] = np.nan
        df["last_review_year"] = df[DATE_COL].dt.year
        df["last_review_month"] = df[DATE_COL].dt.month
        df["has_review"] = df[DATE_COL].notna().astype(int)
        df = df.drop(columns=[DATE_COL])
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = engineer_features(df)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
    y = df[TARGET_COLUMN].astype(float)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def get_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def evaluate_regression_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


@st.cache_data
def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def normalize_metadata_keys(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized[normalize_model_name(key)] = value
    return normalized


def load_manifest() -> dict[str, Any] | None:
    return load_json(METADATA_DIR / METADATA_FILES["manifest"])


def build_manifest_from_files() -> dict[str, Any]:
    variants = {
        "Linear Regression": ["linear_regression", "linear"],
        "Lasso Regression": ["lasso_regression", "lasso"],
        "Ridge Regression": ["ridge_regression", "ridge"],
        "CART / Decision Tree": ["cart", "decision_tree", "decision_tree_model"],
        "Random Forest": ["random_forest", "random_forest_model"],
        "LightGBM": ["lightgbm", "lightgbm_model", "boosted_model"],
    }
    models = []
    for label, names in variants.items():
        found = None
        for base in names:
            for ext in (".joblib", ".pkl"):
                candidate = MODELS_DIR / f"{base}{ext}"
                if candidate.exists():
                    found = candidate.name
                    break
            if found:
                break
        if found:
            models.append({"display_name": label, "file": found})
    return {"models": models}


def load_metadata_bundle() -> dict[str, Any]:
    raw_per_model = load_json(METADATA_DIR / METADATA_FILES["per_model"]) or {}
    if not raw_per_model:
        raw_per_model = load_json(METADATA_DIR / "metrics.json") or {}
    raw_cv = load_json(METADATA_DIR / METADATA_FILES["cv_results"]) or {}
    raw_params = load_json(METADATA_DIR / METADATA_FILES["best_params"]) or {}
    raw_predictions = load_json(METADATA_DIR / METADATA_FILES["predictions"]) or {}
    raw_importances = load_json(METADATA_DIR / METADATA_FILES["feature_importances"]) or {}

    return {
        "summary": load_json(METADATA_DIR / METADATA_FILES["summary"]),
        "feature_defaults": load_json(METADATA_DIR / METADATA_FILES["feature_defaults"]),
        "feature_columns": load_json(METADATA_DIR / METADATA_FILES["feature_columns"]),
        "feature_importances": normalize_metadata_keys(raw_importances),
        "metrics_summary": load_json(METADATA_DIR / METADATA_FILES["metrics_summary"]),
        "per_model_metrics": normalize_metadata_keys(raw_per_model),
        "cv_results": normalize_metadata_keys(raw_cv),
        "best_params": normalize_metadata_keys(raw_params),
        "predictions": normalize_metadata_keys(raw_predictions),
        "manifest": load_manifest(),
    }


@st.cache_resource
def load_saved_models_and_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    models_payload: dict[str, Any] = {}
    metadata = load_metadata_bundle()
    missing_models: list[str] = []

    if not MODELS_DIR.exists():
        metadata["load_status"] = {"missing_dir": True, "empty_dir": False, "found": [], "missing_models": []}
        return models_payload, metadata

    model_files = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        metadata["load_status"] = {"missing_dir": False, "empty_dir": True, "found": [], "missing_models": []}
        return models_payload, metadata

    manifest = metadata.get("manifest")
    if not manifest or "models" not in manifest:
        manifest = build_manifest_from_files()
        metadata["manifest"] = manifest
    else:
        for item in manifest.get("models", []):
            filename = item.get("file")
            display_name = normalize_model_name(item.get("display_name", "")) if item else ""
            if filename and not (MODELS_DIR / filename).exists():
                missing_models.append(display_name or filename)

    per_model_metrics = metadata.get("per_model_metrics", {})
    cv_results = metadata.get("cv_results", {})
    best_params = metadata.get("best_params", {})
    predictions = metadata.get("predictions", {})
    feature_importances = metadata.get("feature_importances", {})

    for item in manifest.get("models", []):
        display_name = normalize_model_name(item.get("display_name", ""))
        filename = item.get("file")
        if not display_name or not filename:
            continue
        path = MODELS_DIR / filename
        if not path.exists():
            continue
        try:
            model_obj = joblib.load(path)
        except Exception:
            continue
        models_payload[display_name] = {
            "model": model_obj,
            "artifact_path": str(path),
            "metrics": per_model_metrics.get(display_name),
            "best_params": best_params.get(display_name),
            "cv_results": cv_results.get(display_name),
            "predictions": predictions.get(display_name),
            "supports_shap": display_name in TREE_MODEL_NAMES,
            "supports_feature_importance": display_name in TREE_MODEL_NAMES,
            "feature_importances": feature_importances.get(display_name, []),
        }

    metadata["load_status"] = {
        "missing_dir": False,
        "empty_dir": False,
        "found": list(models_payload.keys()),
        "missing_models": missing_models,
    }
    return models_payload, metadata


@st.cache_data
def compute_cv_summary(model_pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    scores = cross_validate(
        clone(model_pipeline),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    rmse = -scores["test_rmse"]
    mae = -scores["test_mae"]
    r2 = scores["test_r2"]
    return {
        "n_splits": int(CV_SPLITS),
        "scoring": scoring,
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "mae_mean": float(np.mean(mae)),
        "mae_std": float(np.std(mae)),
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "fold_rmse": [float(v) for v in rmse],
        "fold_mae": [float(v) for v in mae],
        "fold_r2": [float(v) for v in r2],
    }



def build_model_comparison_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, payload in metrics.items():
        if not payload:
            continue
        cv_payload = payload.get("cv") or {}
        test_payload = payload.get("test") or {}
        rows.append(
            {
                "Model": model_name,
                "CV Mean RMSE": cv_payload.get("rmse_mean"),
                "CV Std RMSE": cv_payload.get("rmse_std"),
                "Test MAE": test_payload.get("mae"),
                "Test RMSE": test_payload.get("rmse"),
                "Test R²": test_payload.get("r2"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Model",
                "CV Mean RMSE",
                "CV Std RMSE",
                "Test MAE",
                "Test RMSE",
                "Test R²",
            ]
        )
    df = pd.DataFrame(rows)
    df["Model"] = pd.Categorical(df["Model"], MODEL_DISPLAY_ORDER, ordered=True)
    df = df.sort_values("Test RMSE", na_position="last")
    df["Model"] = df["Model"].astype(str)
    return df


def select_best_model(metrics: dict[str, Any]) -> str | None:
    if not metrics:
        return None
    candidates = {k: v for k, v in metrics.items() if v and v.get("test")}
    if not candidates:
        return None
    return min(candidates, key=lambda m: candidates[m]["test"]["rmse"])


def plot_predicted_vs_actual(y_true: pd.Series, y_pred: np.ndarray, title_suffix: str = ""):
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title=f"Predicted vs Actual{title_suffix}",
        opacity=0.7,
    )
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="red", dash="dash"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, title_suffix: str = ""):
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={"x": "Predicted", "y": "Residual"},
        title=f"Residuals vs Predicted{title_suffix}",
        opacity=0.7,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)


def load_shap_module():
    return shap


def compute_shap_values(model_pipeline, X_sample: pd.DataFrame, feature_names: list[str]):
    shap = load_shap_module()
    if shap is None:
        return None, "SHAP is not installed."

    if not hasattr(model_pipeline, "named_steps"):
        return None, "Model pipeline is missing preprocessing steps."

    preprocess = model_pipeline.named_steps.get("preprocess")
    model = model_pipeline.named_steps.get("model")
    if preprocess is None or model is None:
        return None, "Model pipeline is incomplete."

    X_trans = preprocess.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_trans)
        if hasattr(shap_values, "values"):
            values = shap_values.values
            base_values = getattr(shap_values, "base_values", None)
        else:
            values = shap_values
            base_values = None
        expected_value = getattr(explainer, "expected_value", None)
        return {
            "shap": shap,
            "explainer": explainer,
            "values": values,
            "base_values": base_values,
            "expected_value": expected_value,
            "X_trans": X_trans,
            "feature_names": feature_names,
        }, ""
    except Exception:
        return None, "SHAP values could not be computed for this model."


def get_transformed_feature_names(preprocess, input_df: pd.DataFrame) -> list[str]:
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        return list(input_df.columns)


def _select_output_value(value: Any, output_index: int = 0):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[output_index if output_index < len(value) else 0]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim == 1:
        if arr.size == 0:
            return None
        return arr[output_index if output_index < arr.size else 0]
    return arr


def _to_1d_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def build_waterfall_explanation(
    shap_payload: dict[str, Any],
    row_idx: int,
    feature_names: list[str],
    output_index: int = 0,
):
    if shap is None:
        return None
    values = shap_payload.get("values")
    base_values = shap_payload.get("base_values")
    expected_value = shap_payload.get("expected_value")
    X_trans = shap_payload.get("X_trans")

    if values is None or X_trans is None:
        return None

    values_arr = values
    if isinstance(values_arr, (list, tuple)):
        values_arr = values_arr[output_index if output_index < len(values_arr) else 0]
    values_arr = np.asarray(values_arr)

    if values_arr.ndim == 3:
        if output_index >= values_arr.shape[0]:
            output_index = 0
        row_shap = values_arr[output_index, row_idx, :]
    elif values_arr.ndim == 2:
        row_shap = values_arr[row_idx, :]
    elif values_arr.ndim == 1:
        row_shap = values_arr
    else:
        return None

    row_data = np.asarray(X_trans[row_idx]).reshape(-1)

    base = base_values
    if base is None:
        base = expected_value
    base = _select_output_value(base, output_index)
    if isinstance(base, np.ndarray):
        if base.ndim == 2:
            base = base[output_index if output_index < base.shape[0] else 0, row_idx]
        elif base.ndim == 1 and base.size == values_arr.shape[0]:
            base = base[output_index if output_index < base.size else 0]
        elif base.ndim == 1 and base.size == values_arr.shape[1]:
            base = base[row_idx]
    base = _select_output_value(base, output_index)
    if base is None:
        return None

    row_shap = _to_1d_array(row_shap)
    row_data = _to_1d_array(row_data)
    if row_shap is None or row_data is None:
        return None

    if len(feature_names) != len(row_shap):
        feature_names = [f"feature_{i}" for i in range(len(row_shap))]

    return shap.Explanation(
        values=row_shap,
        base_values=base,
        data=row_data,
        feature_names=feature_names,
    )


def render_shap_waterfall(shap_payload: dict[str, Any], row_idx: int, feature_names: list[str]):
    if shap is None:
        st.warning("SHAP is not installed, so explainability plots are unavailable.")
        return
    explanation = build_waterfall_explanation(shap_payload, row_idx, feature_names)
    if explanation is None:
        st.info("Waterfall plot is unavailable for this model output.")
        return
    try:
        shap_payload["shap"].plots.waterfall(explanation)
        st.pyplot(plt.gcf())
        plt.close()
    except Exception:
        st.info("Waterfall plot could not be rendered for this model output.")


def default_value_from_series(series: pd.Series):
    if series is None or series.empty:
        return None
    if pd.api.types.is_bool_dtype(series):
        mode = series.mode(dropna=True)
        return bool(mode.iloc[0]) if not mode.empty else False
    if pd.api.types.is_numeric_dtype(series):
        median = series.median(skipna=True)
        if pd.notna(median):
            return float(median)
        mean = series.mean(skipna=True)
        if pd.notna(mean):
            return float(mean)
        return 0.0
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "unknown"


def build_interactive_input_row(
    df: pd.DataFrame,
    expected_cols: list[str],
    feature_defaults: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = {}
    for col in expected_cols:
        if feature_defaults and col in feature_defaults:
            defaults[col] = feature_defaults[col]
        elif col in df.columns:
            defaults[col] = default_value_from_series(df[col])
        else:
            defaults[col] = 0.0

    user_values: dict[str, Any] = {}
    col_types = get_column_types(df[[c for c in df.columns if c in expected_cols]])
    numeric_cols = col_types["numeric"][:6]
    categorical_cols = col_types["categorical"][:4]

    st.markdown("**User-controlled features**")
    for col in numeric_cols:
        default_val = float(defaults.get(col, 0.0) or 0.0)
        user_values[col] = st.number_input(col, value=default_val)
    for col in categorical_cols:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        default_val = str(defaults.get(col, ""))
        if default_val not in options and options:
            default_val = options[0]
        user_values[col] = st.selectbox(col, options, index=options.index(default_val) if default_val in options else 0)

    auto_filled = {k: v for k, v in defaults.items() if k not in user_values}
    st.caption("Auto-filled features use dataset medians (numeric) or most frequent values (categorical).")
    st.dataframe(pd.DataFrame(auto_filled.items(), columns=["Feature", "Auto-filled value"]).head(10), use_container_width=True)
    return user_values, defaults


def build_prediction_dataframe(
    user_values: dict[str, Any],
    defaults: dict[str, Any],
    expected_cols: list[str],
) -> pd.DataFrame:
    row = {col: defaults.get(col, 0.0) for col in expected_cols}
    row.update(user_values)
    df_row = pd.DataFrame([row], columns=expected_cols)
    return df_row


def format_cv_summary(cv_payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "R²"],
            "Mean": [cv_payload.get("rmse_mean"), cv_payload.get("mae_mean"), cv_payload.get("r2_mean")],
            "Std": [cv_payload.get("rmse_std"), cv_payload.get("mae_std"), cv_payload.get("r2_std")],
        }
    )


def format_best_params(best_params_payload: Any) -> pd.DataFrame | None:
    if not best_params_payload:
        return None
    if isinstance(best_params_payload, dict) and "note" in best_params_payload:
        return None
    if isinstance(best_params_payload, dict):
        return pd.DataFrame(best_params_payload.items(), columns=["Hyperparameter", "Value"])
    return None


def infer_best_params(model_name: str, model_pipeline) -> dict[str, Any] | None:
    keys = BEST_PARAM_KEYS.get(model_name)
    if not keys:
        return None
    if hasattr(model_pipeline, "named_steps"):
        model_step = model_pipeline.named_steps.get("model")
    else:
        model_step = model_pipeline
    if model_step is None or not hasattr(model_step, "get_params"):
        return None
    params = model_step.get_params()
    inferred = {key: params.get(key) for key in keys if key in params}
    return inferred or None


def infer_feature_importances(model_pipeline, top_n: int = 12) -> list[dict[str, float]]:
    if not hasattr(model_pipeline, "named_steps"):
        return []
    preprocess = model_pipeline.named_steps.get("preprocess")
    model_step = model_pipeline.named_steps.get("model")
    if preprocess is None or model_step is None:
        return []
    if not hasattr(model_step, "feature_importances_"):
        return []
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        return []
    importances = model_step.feature_importances_
    if importances is None:
        return []
    top_features = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    return [
        {"feature": row.feature, "importance": float(row.importance)}
        for row in top_features.itertuples(index=False)
    ]


def build_interpretation_text(model_name: str, test_metrics: dict[str, Any] | None, cv_payload: dict[str, Any] | None) -> str:
    if not test_metrics:
        return "This model is available, but evaluation metrics were not recovered."
    rmse = test_metrics.get("rmse")
    r2 = test_metrics.get("r2")
    cv_rmse = cv_payload.get("rmse_mean") if cv_payload else None
    cv_std = cv_payload.get("rmse_std") if cv_payload else None
    stability = "stable" if cv_std is not None and cv_std < 10 else "variable"
    lines = [
        f"{model_name} achieves a test RMSE of {rmse:.2f} and an R² of {r2:.3f}.",
    ]
    if cv_rmse is not None and cv_std is not None:
        lines.append(
            f"Cross-validation RMSE averages {cv_rmse:.2f} with a std of {cv_std:.2f}, indicating {stability} performance across folds."
        )
    lines.append("Use this model when you want a balance of accuracy and generalization for price prediction.")
    return " ".join(lines)


def render_executive_summary_tab(df: pd.DataFrame, summary: dict | None, best_model_name: str | None):
    st.header("Executive Summary")
    st.markdown(
        f"""
        **Dataset:** {DATASET_DESCRIPTION}  
        **Prediction task:** {PREDICTION_TASK_DESCRIPTION}  
        **Why it matters:** {WHY_IT_MATTERS}
        """
    )
    st.info(APPROACH_SUMMARY)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 1}")
    c3.metric("Target", TARGET_COLUMN)

    if best_model_name:
        st.success(f"Best model: {best_model_name}")

    st.markdown(KEY_FINDINGS_SUMMARY)
    if summary:
        st.caption("Artifacts are loaded.")


def render_descriptive_analytics_tab(df: pd.DataFrame):
    st.header("Descriptive Analytics")
    st.subheader("Target Distribution")
    fig = px.histogram(df, x=TARGET_COLUMN, nbins=40, title="Target Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Distribution of nightly prices to understand scale and outliers.")

    st.subheader("Key Feature Distributions")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != TARGET_COLUMN]
    if numeric_cols:
        fig = px.box(df, y=numeric_cols[:6], title="Sample Numeric Features")
        st.plotly_chart(fig, use_container_width=True)

    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
    if categorical_cols:
        cat_col = categorical_cols[0]
        counts_df = (
            df[cat_col]
            .fillna("Missing")
            .value_counts()
            .head(10)
            .reset_index()
        )
        counts_df.columns = ["category", "count"]
        fig = px.bar(
            counts_df,
            x="category",
            y="count",
            title=f"Top Categories: {cat_col}",
            labels={"category": cat_col, "count": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns available for category frequency plots.")

    st.subheader("Correlation Heatmap")
    if numeric_cols:
        corr = df[numeric_cols + [TARGET_COLUMN]].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation heatmap highlights linear relationships among numeric features.")
    else:
        st.info("Not enough numeric features available to compute a correlation heatmap.")


def render_model_performance_tab(
    models_payload: dict[str, Any],
    comparison: pd.DataFrame,
    results: dict[str, Any],
    active_model_name: str | None,
    best_model_name: str | None,
):
    st.header("Model Performance")

    if not models_payload:
        st.warning("Some model artifacts are missing.")
        return

    st.subheader("Model Comparison Summary")
    if comparison.empty:
        st.info("Comparison table is unavailable. Metrics metadata may be missing.")
    else:
        def highlight_best(row):
            if best_model_name and row["Model"] == best_model_name:
                return ["background-color: #e6f4ea"] * len(row)
            return [""] * len(row)

        st.dataframe(
            comparison.style.apply(highlight_best, axis=1),
            use_container_width=True,
        )
        if best_model_name:
            st.caption(f"Best-performing model: {best_model_name}")

        st.subheader("Comparison Charts")
        chart_df = comparison.copy()
        chart_df["Best"] = np.where(chart_df["Model"] == best_model_name, "Best model", "Other")

        fig_rmse = px.bar(
            chart_df.sort_values("Test RMSE", na_position="last"),
            x="Model",
            y="Test RMSE",
            color="Best",
            title="Test RMSE by Model",
            color_discrete_map={"Best model": "#2ca02c", "Other": "#9ecae1"},
        )
        fig_r2 = px.bar(
            chart_df.sort_values("Test R²", ascending=False, na_position="last"),
            x="Model",
            y="Test R²",
            color="Best",
            title="Test R² by Model",
            color_discrete_map={"Best model": "#2ca02c", "Other": "#9ecae1"},
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.subheader("Model Details")
    if not active_model_name or active_model_name not in models_payload:
        active_model_name = list(models_payload.keys())[0]
    model_payload = models_payload[active_model_name]

    st.caption(f"Currently viewing: {active_model_name}")

    test_metrics = model_payload.get("metrics", {}).get("test")
    if active_model_name in results:
        test_metrics = results[active_model_name].get("test", test_metrics)

    if test_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Test MAE", f"{test_metrics['mae']:.2f}")
        c2.metric("Test RMSE", f"{test_metrics['rmse']:.2f}")
        c3.metric("Test R²", f"{test_metrics['r2']:.3f}")
    else:
        st.info("Test metrics unavailable for this model.")

    cv_payload = model_payload.get("cv_results")
    if cv_payload:
        st.markdown("**Cross-validation summary**")
        cv_table = format_cv_summary(cv_payload)
        st.table(cv_table)
        st.caption(
            f"Folds: {cv_payload.get('n_splits', CV_SPLITS)} | Scoring: RMSE (primary), MAE, R²"
        )
    else:
        st.info("Cross-validation summary missing for this model.")

    best_params_payload = model_payload.get("best_params")
    if best_params_payload:
        st.markdown("**Best hyperparameters**")
        if isinstance(best_params_payload, dict) and "note" in best_params_payload:
            st.info(best_params_payload["note"])
        else:
            params_table = format_best_params(best_params_payload)
            if params_table is not None:
                st.table(params_table)
            else:
                st.json(best_params_payload)
    else:
        st.info("Best hyperparameters not available for this model.")

    if active_model_name in results:
        y_true = results[active_model_name]["y_true"]
        y_pred = results[active_model_name]["y_pred"]
        plot_predicted_vs_actual(y_true, y_pred)
        plot_residuals(y_true, y_pred)
    else:
        st.info("Prediction plots are unavailable for this model.")

    if model_payload.get("supports_feature_importance"):
        features = model_payload.get("feature_importances") or []
        if features:
            st.markdown("**Top Features (importance)**")
            if features and isinstance(features[0], dict):
                importance_df = pd.DataFrame(features)
                fig_imp = px.bar(
                    importance_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Feature Importance",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.write(", ".join([str(item) for item in features]))
        else:
            st.info("Feature importance unavailable for this model.")

    st.markdown(build_interpretation_text(active_model_name, test_metrics, cv_payload))


def render_explainability_tab(
    df: pd.DataFrame,
    models_payload: dict[str, Any],
    metadata: dict[str, Any],
    active_model_name: str | None,
):
    st.header("Explainability & Interactive Prediction")

    if not models_payload:
        st.warning("Some model artifacts are missing.")
        return

    if not active_model_name or active_model_name not in models_payload:
        active_model_name = list(models_payload.keys())[0]
    model_payload = models_payload[active_model_name]
    model_pipeline = model_payload["model"]

    st.subheader("Explainability")
    st.caption(f"Currently viewing: {active_model_name}")

    if model_payload.get("supports_shap"):
        if shap is None:
            st.warning("SHAP is not installed, so explainability plots are unavailable.")
            return
        X, _ = split_features_target(df)
        sample = X.sample(n=min(300, len(X)), random_state=42)
        feature_columns = metadata.get("feature_columns")
        if feature_columns:
            sample = sample[feature_columns]
        preprocess = model_pipeline.named_steps.get("preprocess") if hasattr(model_pipeline, "named_steps") else None
        feature_names = get_transformed_feature_names(preprocess, sample) if preprocess is not None else list(sample.columns)

        shap_payload, error = compute_shap_values(model_pipeline, sample, feature_names)
        if shap_payload is None:
            st.info(error)
        else:
            shap_values = shap_payload["values"]
            X_trans = shap_payload["X_trans"]
            feature_names = shap_payload["feature_names"]

            st.markdown("**SHAP Summary (Beeswarm)**")
            shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
            st.pyplot(plt.gcf())
            plt.close()

            st.markdown("**SHAP Bar Plot**")
            shap.summary_plot(shap_values, X_trans, feature_names=feature_names, plot_type="bar", show=False)
            st.pyplot(plt.gcf())
            plt.close()

            st.markdown("**SHAP Waterfall (single prediction)**")
            row_idx = st.slider("Inspect single prediction index", 0, len(sample) - 1, 0)
            render_shap_waterfall(shap_payload, row_idx, feature_names)
    else:
        st.info(f"SHAP artifacts unavailable for {active_model_name}. Showing coefficient-based note instead.")
        st.caption("For linear models, coefficients describe the direction and magnitude of each feature's impact.")

    st.subheader("Interactive Prediction")
    X, _ = split_features_target(df)
    expected_cols = metadata.get("feature_columns") or list(X.columns)

    user_values, defaults = build_interactive_input_row(X, expected_cols, metadata.get("feature_defaults"))
    input_df = build_prediction_dataframe(user_values, defaults, expected_cols)

    try:
        prediction = float(model_pipeline.predict(input_df)[0])
    except Exception:
        st.warning("Prediction failed for the selected model.")
        return

    st.metric("Predicted price", f"${prediction:,.2f}")

    if model_payload.get("supports_shap"):
        st.caption("SHAP waterfall above explains the selected model's prediction.")
    else:
        tree_models = [name for name, payload in models_payload.items() if payload.get("supports_shap")]
        if tree_models:
            st.info("Tree-based SHAP explanations are available for tree models in the selector.")


st.title("NYC Airbnb Open Data: Complete Modeling Studio")

models_payload, metadata = load_saved_models_and_metadata()
load_status = metadata.get("load_status", {})

raw_df = load_dataset()
X, y = split_features_target(raw_df)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

results: dict[str, Any] = {}
metrics: dict[str, Any] = {}

for name, payload in models_payload.items():
    model = payload["model"]
    predictions_payload = payload.get("predictions")

    if predictions_payload and "y_true" in predictions_payload and "y_pred" in predictions_payload:
        y_true = pd.Series(predictions_payload["y_true"])
        y_pred = np.array(predictions_payload["y_pred"])
        test_metrics = evaluate_regression_model(y_true, y_pred)
    else:
        try:
            y_pred = model.predict(X_test)
            y_true = y_test
            test_metrics = evaluate_regression_model(y_true, y_pred)
        except Exception:
            y_true = None
            y_pred = None
            test_metrics = None

    cv_payload = payload.get("cv_results")
    if not cv_payload:
        try:
            cv_payload = compute_cv_summary(model, X_train, y_train)
        except Exception:
            cv_payload = None

    best_params_payload = payload.get("best_params")
    if not best_params_payload and name == "Linear Regression":
        best_params_payload = {"note": "No tuning applied for baseline linear regression."}
    if not best_params_payload:
        best_params_payload = infer_best_params(name, model)

    if payload.get("supports_feature_importance") and not payload.get("feature_importances"):
        payload["feature_importances"] = infer_feature_importances(model)

    payload["cv_results"] = cv_payload
    payload["best_params"] = best_params_payload

    if test_metrics:
        payload["metrics"] = payload.get("metrics") or {}
        payload["metrics"]["test"] = test_metrics

    if y_true is not None and y_pred is not None:
        results[name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "test": test_metrics,
        }

    if test_metrics:
        metrics[name] = {
            "test": test_metrics,
            "cv": cv_payload,
        }

comparison = build_model_comparison_table(metrics)
metrics_summary = metadata.get("metrics_summary") or {}
if metrics_summary and isinstance(metrics_summary, dict) and metrics_summary.get("rows"):
    summary_df = pd.DataFrame(metrics_summary["rows"])
    if "Test R2" in summary_df.columns and "Test R²" not in summary_df.columns:
        summary_df = summary_df.rename(columns={"Test R2": "Test R²"})
    if "Model" in summary_df.columns:
        summary_df["Model"] = summary_df["Model"].apply(normalize_model_name)
    if not summary_df.empty:
        comparison = (
            summary_df.set_index("Model")
            .combine_first(comparison.set_index("Model"))
            .reset_index()
        )
        comparison = comparison.sort_values("Test RMSE", na_position="last")

best_model_name = select_best_model(metrics)
manifest = metadata.get("manifest")
if manifest and manifest.get("best_model"):
    best_model_name = normalize_model_name(manifest["best_model"]) or best_model_name

st.sidebar.header("Explore Models")
if load_status.get("missing_dir") or load_status.get("empty_dir"):
    st.sidebar.warning("Some model artifacts are missing.")
    active_model_name = None
else:
    missing_models = load_status.get("missing_models") or []
    if missing_models:
        st.sidebar.info("Unavailable models: " + ", ".join(missing_models))
    model_names = list(models_payload.keys())
    st.sidebar.markdown(f"{len(model_names)} trained models available.")
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = model_names[0] if model_names else None
    active_model_name = st.sidebar.selectbox("Model Selector", model_names, key="selected_model")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

with tab1:
    render_executive_summary_tab(raw_df, metadata.get("summary"), best_model_name)

with tab2:
    render_descriptive_analytics_tab(raw_df)

with tab3:
    render_model_performance_tab(models_payload, comparison, results, active_model_name, best_model_name)

with tab4:
    render_explainability_tab(raw_df, models_payload, metadata, active_model_name)

