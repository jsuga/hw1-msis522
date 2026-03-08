from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "AB_NYC_2019.csv"
ARTIFACTS_DIR = APP_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

TARGET_COLUMN = "price"
DROP_COLS = ["id", "name", "host_id", "host_name"]
DATE_COL = "last_review"
RANDOM_STATE = 42
CV_SPLITS = 2
N_JOBS = 1
MAX_TRAIN_ROWS = 5000

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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

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


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_cv_summary(scores: dict[str, np.ndarray], n_splits: int) -> dict[str, Any]:
    rmse = -scores["test_rmse"]
    mae = -scores["test_mae"]
    r2 = scores["test_r2"]
    return {
        "n_splits": int(n_splits),
        "scoring": {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def compute_feature_defaults(X_train: pd.DataFrame) -> dict[str, Any]:
    defaults = {}
    for col in X_train.columns:
        series = X_train[col]
        if pd.api.types.is_numeric_dtype(series):
            median = series.median(skipna=True)
            if pd.notna(median):
                defaults[col] = float(median)
            else:
                mean = series.mean(skipna=True)
                defaults[col] = float(mean) if pd.notna(mean) else 0.0
        else:
            mode = series.mode(dropna=True)
            defaults[col] = mode.iloc[0] if not mode.empty else "unknown"
    return defaults


def trim_model_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if not params:
        return None
    clean = {}
    for key, value in params.items():
        clean[key.replace("model__", "")] = value
    return clean


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset()
    if len(raw_df) > MAX_TRAIN_ROWS:
        raw_df = raw_df.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_STATE)
    X, y = split_features_target(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_train)
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    per_model_metrics: dict[str, Any] = {}
    cv_results: dict[str, Any] = {}
    best_params: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    feature_importances: dict[str, list[dict[str, float]]] = {}

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "target": TARGET_COLUMN,
        "cv_splits": CV_SPLITS,
        "scoring": scoring,
        "models": [],
        "best_model": None,
    }

    model_specs = {
        "Linear Regression": {
            "model": LinearRegression(),
            "grid": None,
            "file": "linear_regression.joblib",
            "is_tree": False,
            "tuned": False,
        },
        "Lasso Regression": {
            "model": Lasso(max_iter=20000, random_state=RANDOM_STATE),
            "grid": {"model__alpha": [0.1]},
            "file": "lasso_regression.joblib",
            "is_tree": False,
            "tuned": True,
        },
        "Ridge Regression": {
            "model": Ridge(random_state=RANDOM_STATE),
            "grid": {"model__alpha": [1.0]},
            "file": "ridge_regression.joblib",
            "is_tree": False,
            "tuned": True,
        },
        "CART / Decision Tree": {
            "model": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "grid": {
                "model__max_depth": [8],
                "model__min_samples_leaf": [1],
                "model__min_samples_split": [2],
            },
            "file": "cart.joblib",
            "is_tree": True,
            "tuned": True,
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            "grid": {
                "model__n_estimators": [200],
                "model__max_depth": [None],
                "model__min_samples_leaf": [1],
            },
            "file": "random_forest.joblib",
            "is_tree": True,
            "tuned": True,
        },
        "LightGBM": {
            "model": None,
            "grid": {
                "model__n_estimators": [200],
                "model__max_depth": [-1],
                "model__learning_rate": [0.1],
                "model__num_leaves": [31],
            },
            "file": "lightgbm.joblib",
            "is_tree": True,
            "tuned": True,
        },
    }

    try:
        from lightgbm import LGBMRegressor  # type: ignore

        model_specs["LightGBM"]["model"] = LGBMRegressor(random_state=RANDOM_STATE)
    except Exception:
        model_specs["LightGBM"]["model"] = None

    for display_name, spec in model_specs.items():
        model = spec["model"]
        if model is None:
            continue

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        if spec["grid"]:
            grid = GridSearchCV(
                pipeline,
                spec["grid"],
                cv=cv,
                scoring=scoring,
                n_jobs=N_JOBS,
                refit="rmse",
            )
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            clean_params = trim_model_params(grid.best_params_)
            cv_scores = cross_validate(
                best_estimator,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=N_JOBS,
                return_train_score=False,
            )
            cv_payload = build_cv_summary(cv_scores, CV_SPLITS)
        else:
            cv_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=N_JOBS,
                return_train_score=False,
            )
            best_estimator = pipeline.fit(X_train, y_train)
            clean_params = {"note": "No tuning applied for baseline linear regression."}
            cv_payload = build_cv_summary(cv_scores, CV_SPLITS)

        y_pred = best_estimator.predict(X_test)
        test_metrics = evaluate_regression(y_test, y_pred)

        file_name = spec["file"]
        joblib.dump(best_estimator, MODELS_DIR / file_name)

        per_model_metrics[display_name] = {
            "test": test_metrics,
            "cv": cv_payload,
            "is_tree": spec["is_tree"],
        }
        cv_results[display_name] = cv_payload
        best_params[display_name] = clean_params
        predictions[display_name] = {
            "y_true": [float(v) for v in y_test.values[:3000]],
            "y_pred": [float(v) for v in y_pred[:3000]],
        }

        manifest["models"].append(
            {
                "display_name": display_name,
                "file": file_name,
                "tuned": bool(spec["tuned"]),
                "is_tree": bool(spec["is_tree"]),
            }
        )

        if spec["is_tree"]:
            model_step = best_estimator.named_steps.get("model")
            if hasattr(model_step, "feature_importances_"):
                preprocess = best_estimator.named_steps.get("preprocess")
                feature_names = preprocess.get_feature_names_out()
                importances = model_step.feature_importances_
                top_features = (
                    pd.DataFrame({"feature": feature_names, "importance": importances})
                    .sort_values("importance", ascending=False)
                    .head(12)
                )
                feature_importances[display_name] = [
                    {"feature": row.feature, "importance": float(row.importance)}
                    for row in top_features.itertuples(index=False)
                ]

    if per_model_metrics:
        best_model = min(per_model_metrics, key=lambda m: per_model_metrics[m]["test"]["rmse"])
        manifest["best_model"] = best_model

    summary = {
        "target": TARGET_COLUMN,
        "input_rows": int(len(raw_df)),
        "input_features": list(X_train.columns),
        "models_trained": list(per_model_metrics.keys()),
        "best_model": manifest.get("best_model"),
    }

    metrics_summary = []
    for model_name, payload in per_model_metrics.items():
        cv_payload = payload.get("cv", {})
        test_payload = payload.get("test", {})
        metrics_summary.append(
            {
                "Model": model_name,
                "CV Mean RMSE": cv_payload.get("rmse_mean"),
                "CV Std RMSE": cv_payload.get("rmse_std"),
                "Test MAE": test_payload.get("mae"),
                "Test RMSE": test_payload.get("rmse"),
                "Test R2": test_payload.get("r2"),
            }
        )

    feature_defaults = compute_feature_defaults(X_train)

    save_json(METADATA_DIR / METADATA_FILES["manifest"], manifest)
    save_json(METADATA_DIR / METADATA_FILES["metrics_summary"], {"rows": metrics_summary})
    save_json(METADATA_DIR / METADATA_FILES["per_model"], per_model_metrics)
    save_json(METADATA_DIR / METADATA_FILES["cv_results"], cv_results)
    save_json(METADATA_DIR / METADATA_FILES["best_params"], best_params)
    save_json(METADATA_DIR / METADATA_FILES["predictions"], predictions)
    save_json(METADATA_DIR / METADATA_FILES["feature_defaults"], feature_defaults)
    save_json(METADATA_DIR / METADATA_FILES["feature_columns"], list(X_train.columns))
    save_json(METADATA_DIR / METADATA_FILES["feature_importances"], feature_importances)
    save_json(METADATA_DIR / METADATA_FILES["summary"], summary)

    # Backward compatibility for older app loaders.
    save_json(METADATA_DIR / "metrics.json", per_model_metrics)


if __name__ == "__main__":
    main()
