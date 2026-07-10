"""
IntelliLend — Engine A: Lead Scoring Model
============================================
Trains a LightGBM classifier to predict propensity-to-convert per customer.
Outputs a Lead Quality Score (0–100) calibrated from predicted probabilities.

Key metrics reported:
  - ROC-AUC
  - Precision@Top-20%
  - Simulated conversion rate at top-N cutoffs (must exceed 30%)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    classification_report, average_precision_score,
)
import lightgbm as lgb

from src.features import LEAD_SCORING_FEATURES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lead_scoring_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "lead_scoring_metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lead_scoring_model(
    features_df: pd.DataFrame,
    target_col: str = "loan_converted",
    feature_cols: list = None,
) -> dict:
    """Train a LightGBM lead scoring model with calibration.

    Args:
        features_df: Feature matrix from features.build_feature_matrix()
        target_col: Binary conversion label
        feature_cols: List of feature column names

    Returns:
        dict with model, metrics, feature importances, and predictions
    """
    if feature_cols is None:
        feature_cols = LEAD_SCORING_FEATURES

    # Filter to available features
    available_features = [f for f in feature_cols if f in features_df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"  ⚠️  Missing features (will skip): {missing}")

    X = features_df[available_features].copy()
    y = features_df[target_col].copy()

    print(f"\n  📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  📊 Class balance: {y.value_counts().to_dict()}")
    print(f"  📊 Base conversion rate: {y.mean()*100:.1f}%")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # LightGBM with tuned hyperparameters
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Calibrate probabilities using isotonic regression
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    # Predictions
    y_pred_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # ---------------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------------
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)

    # Precision@Top-20%
    n_top20 = max(int(len(y_test) * 0.20), 1)
    top20_idx = np.argsort(y_pred_proba)[::-1][:n_top20]
    precision_at_20 = y_test.iloc[top20_idx].mean()

    # Simulated conversion rate at various top-N cutoffs
    conversion_sim = {}
    for pct in [5, 10, 15, 20, 25, 30, 50]:
        n = max(int(len(y_test) * pct / 100), 1)
        top_idx = np.argsort(y_pred_proba)[::-1][:n]
        conv_rate = y_test.iloc[top_idx].mean() * 100
        conversion_sim[f"top_{pct}pct"] = round(conv_rate, 1)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(avg_prec, 4),
        "precision_at_top20": round(precision_at_20, 4),
        "base_conversion_rate": round(y.mean() * 100, 1),
        "conversion_rate_simulation": conversion_sim,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(available_features),
    }

    print(f"\n  📈 Model Performance:")
    print(f"     ROC-AUC:              {metrics['roc_auc']}")
    print(f"     Average Precision:    {metrics['average_precision']}")
    print(f"     Precision@Top-20%:    {metrics['precision_at_top20']:.1%}")
    print(f"\n  📈 Conversion Rate Simulation (vs. baseline {metrics['base_conversion_rate']:.1f}%):")
    for k, v in conversion_sim.items():
        label = k.replace("top_", "Top ").replace("pct", "%")
        marker = " ✅ >30%!" if v > 30 else ""
        print(f"     {label:12s}:  {v:.1f}%{marker}")

    # Feature importance
    importance = dict(zip(available_features, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  📊 Top 10 Feature Importances:")
    for feat, imp in list(importance.items())[:10]:
        print(f"     {feat:35s}: {imp}")

    # Full predictions on entire dataset
    X_all = features_df[available_features]
    all_proba = calibrated.predict_proba(X_all)[:, 1]
    features_df = features_df.copy()
    features_df["lead_score"] = (all_proba * 100).round(1)
    features_df["lead_score_raw"] = all_proba

    return {
        "model": calibrated,
        "base_model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_cols": available_features,
        "predictions_df": features_df,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(result: dict, model_path: str = MODEL_PATH, metrics_path: str = METRICS_PATH):
    """Save trained model and metrics to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model + feature list
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": result["model"],
            "base_model": result["base_model"],
            "feature_cols": result["feature_cols"],
        }, f)
    print(f"  ✅ Model saved: {model_path}")

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)
    print(f"  ✅ Metrics saved: {metrics_path}")


def load_model(model_path: str = MODEL_PATH) -> dict:
    """Load saved model from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_lead_scores(
    features_df: pd.DataFrame,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """Score new customers using saved model.

    Returns DataFrame with lead_score (0-100) column added.
    """
    saved = load_model(model_path)
    model = saved["model"]
    feature_cols = saved["feature_cols"]

    available = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available].fillna(0)

    proba = model.predict_proba(X)[:, 1]
    features_df = features_df.copy()
    features_df["lead_score"] = (proba * 100).round(1)
    features_df["lead_score_raw"] = proba

    return features_df
