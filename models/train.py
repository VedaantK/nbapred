"""
Model training pipeline.
Trains Linear Regression, Random Forest, and XGBoost on historical game data.
Uses time-based train/test split — never random split on time series.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config.settings import DB_PATH, MODELS_DIR
from config.logging_config import setup_logging
from features.engineer import build_training_dataset

try:
    from models.neural_net import TorchMLPRegressor
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = setup_logging("train")

MODELS_DIR = Path(MODELS_DIR) if not isinstance(MODELS_DIR, Path) else MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Metadata columns that are not features
_META = {"player_id", "player_name", "game_date", "target_points", "games_in_db"}

# ── model definitions ─────────────────────────────────────────────────────────

def _build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def _build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def _build_xgb_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )),
    ])


# ── evaluation helpers ────────────────────────────────────────────────────────

def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, lines: np.ndarray) -> float:
    """
    Percentage of games where the model correctly called over/under.
    Requires sportsbook lines — approximated by season avg when unavailable.
    """
    if lines is None or len(lines) != len(y_true):
        # Fallback: use median as the line
        lines = np.full(len(y_true), np.median(y_true))
    true_over = y_true > lines
    pred_over = y_pred > lines
    return float(np.mean(true_over == pred_over))


def get_feature_importance(model_pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract and rank feature importances from trained model pipeline."""
    model = model_pipeline.named_steps["model"]

    # Resolve the actual feature names the model saw, accounting for any
    # columns the imputer may have dropped (keep_empty_features=True prevents
    # this, but we guard here as a safety net).
    imputer = model_pipeline.named_steps.get("imputer")
    if imputer is not None and hasattr(imputer, "get_feature_names_out"):
        try:
            actual_names = list(imputer.get_feature_names_out(feature_names))
        except Exception:
            actual_names = feature_names
    else:
        actual_names = feature_names

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return pd.DataFrame({"feature": actual_names, "importance": [np.nan] * len(actual_names)})

    # Guard: if lengths still differ (shouldn't happen with keep_empty_features=True),
    # align by truncating to the shorter side rather than crashing.
    n = min(len(actual_names), len(importances))
    if n < len(actual_names):
        logger.warning(
            f"Feature name / importance length mismatch: {len(actual_names)} names vs "
            f"{len(importances)} importances. Truncating to {n}."
        )

    df = pd.DataFrame({"feature": actual_names[:n], "importance": list(importances[:n])})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ── main training function ────────────────────────────────────────────────────

def train_all_models(db_path: str | Path = DB_PATH) -> dict:
    """
    Full training pipeline.
    Returns dict with trained models, metrics, residuals, and feature lists.
    """
    db_path = Path(db_path)
    logger.info("Loading training dataset...")
    df = build_training_dataset(db_path)

    if df.empty or len(df) < 50:
        logger.error("Insufficient training data. Run backfill_data.py first.")
        return {}

    feature_cols = [c for c in df.columns if c not in _META]
    X = df[feature_cols]
    y = df["target_points"].values
    dates = pd.to_datetime(df["game_date"])

    # Time-based split: train on older 80%, test on most recent 20%
    cutoff_idx = int(len(df) * 0.80)
    sorted_idx = dates.argsort().values
    train_idx = sorted_idx[:cutoff_idx]
    test_idx = sorted_idx[cutoff_idx:]

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    models_config = {
        "linear_regression": _build_lr_pipeline(),
        "random_forest": _build_rf_pipeline(),
        "xgboost": _build_xgb_pipeline(),
    }

    if _TORCH_AVAILABLE:
        models_config["neural_net"] = TorchMLPRegressor(
            epochs=150,
            batch_size=256,
            lr=1e-3,
            weight_decay=1e-4,
            patience=15,
        )
    else:
        logger.warning("PyTorch not available — skipping neural_net model")

    results = {}

    for name, pipeline in models_config.items():
        logger.info(f"Training {name}...")
        try:
            pipeline.fit(X_train.values if hasattr(X_train, "values") else X_train, y_train)
            preds = pipeline.predict(X_test.values if hasattr(X_test, "values") else X_test)
        except Exception as e:
            logger.warning(f"Skipping {name} due to training error: {e}")
            continue

        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        dir_acc = _directional_accuracy(y_test, preds, lines=None)

        # Compute residuals on test set (actual - predicted)
        residuals = (y_test - preds).tolist()

        # Feature importance (sklearn pipelines only)
        if isinstance(pipeline, Pipeline):
            importance_df = get_feature_importance(pipeline, feature_cols)
        else:
            importance_df = pd.DataFrame({"feature": feature_cols, "importance": [np.nan] * len(feature_cols)})

        # Save model to disk
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, model_path)

        # Save residuals
        residuals_path = MODELS_DIR / f"{name}_residuals.json"
        with open(residuals_path, "w") as f:
            json.dump(residuals, f)

        # Save feature importance
        importance_path = MODELS_DIR / f"{name}_importance.json"
        importance_df.to_json(importance_path, orient="records")

        metrics = {"mae": round(mae, 3), "rmse": round(rmse, 3), "directional_accuracy": round(dir_acc, 3)}
        logger.info(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, Dir.Acc={dir_acc:.1%}")

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "residuals": residuals,
            "feature_names": feature_cols,
            "importance": importance_df,
        }

    # Save feature names for later use
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f)

    # Persist the metrics too. They used to exist only as a log line, so
    # nothing downstream — the API, the dashboard export — could read how the
    # models actually scored without retraining.
    from datetime import datetime as _dt
    metrics_blob = {
        "trained_at": _dt.now().isoformat(timespec="seconds"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": len(feature_cols),
        "attempted": list(models_config.keys()),
        "models": {name: info["metrics"] for name, info in results.items()},
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_blob, f, indent=2)
    logger.info(f"Metrics saved to {MODELS_DIR / 'metrics.json'}")

    logger.info("Training complete. Models saved to disk.")
    return results


_ALL_MODEL_NAMES = ("linear_regression", "random_forest", "xgboost", "neural_net")


def load_models() -> dict:
    """Load all trained models from disk."""
    models_dir = Path(MODELS_DIR)
    loaded = {}

    for name in _ALL_MODEL_NAMES:
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            logger.debug(f"Model not found (skipping): {model_path}")
            continue
        try:
            loaded[name] = joblib.load(model_path)
            logger.info(f"Loaded model: {name}")
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")

    return loaded


def save_model_weights(weights: dict[str, float]):
    """Persist dynamic ensemble weights to disk."""
    weights_path = Path(MODELS_DIR) / "model_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights, f)
    logger.info(f"Model weights saved: {weights}")


def load_model_weights() -> dict[str, float]:
    """Load persisted ensemble weights; returns {} if not found."""
    weights_path = Path(MODELS_DIR) / "model_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            return json.load(f)
    return {}


def load_residuals() -> dict:
    """Load residuals for each model (used for probability estimation)."""
    models_dir = Path(MODELS_DIR)
    residuals = {}

    for name in _ALL_MODEL_NAMES:
        path = models_dir / f"{name}_residuals.json"
        if path.exists():
            with open(path) as f:
                residuals[name] = np.array(json.load(f))

    return residuals


def load_feature_names() -> list[str]:
    """Load the feature names used during training."""
    path = Path(MODELS_DIR) / "feature_names.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []
