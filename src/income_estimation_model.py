"""
IntelliLend — Engine B: Alternate Income Assessment
=====================================================
For thin-file / self-employed / gig customers where salary slips are
unreliable, this engine estimates actual disposable income from
transaction behavior.

Output:
  - Estimated monthly income
  - Confidence band (High / Medium / Low)
  - Income deviation flag (vs. declared income)

Model: XGBoost Regressor trained on transaction-derived features
against synthetic "true income" ground truth.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from src.features import INCOME_ESTIMATION_FEATURES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "income_estimation_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "income_estimation_metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Confidence band logic
# ---------------------------------------------------------------------------

def assign_confidence_band(
    estimated_income: np.ndarray,
    true_income: np.ndarray = None,
    occupation: pd.Series = None,
    cashflow_cv: pd.Series = None,
) -> np.ndarray:
    """Assign confidence bands based on multiple signals:

    High:   Salaried + low cash-flow CV + income estimate error <10%
    Medium: Semi-regular income + error 10-25%
    Low:    Irregular income + error >25% or thin data

    If true_income is not available (inference time), use heuristics
    based on occupation and cash-flow stability.
    """
    n = len(estimated_income)
    bands = np.array(["Medium"] * n)

    if true_income is not None:
        # Training/validation: use actual error
        abs_pct_error = np.abs(estimated_income - true_income) / np.maximum(true_income, 1) * 100
        bands[abs_pct_error < 10] = "High"
        bands[abs_pct_error > 25] = "Low"
    else:
        # Inference: use heuristics
        if occupation is not None:
            occ_values = occupation.values if hasattr(occupation, 'values') else occupation
            bands[occ_values == "salaried"] = "High"
            bands[occ_values == "gig_worker"] = "Low"

    # Cash-flow stability override
    if cashflow_cv is not None:
        cv_values = cashflow_cv.values if hasattr(cashflow_cv, 'values') else cashflow_cv
        # Very stable cash flow → upgrade confidence
        stable_mask = cv_values < 0.10
        bands[stable_mask & (bands == "Medium")] = "High"
        # Very unstable → downgrade
        unstable_mask = cv_values > 0.40
        bands[unstable_mask & (bands == "High")] = "Medium"
        bands[unstable_mask & (bands == "Medium")] = "Low"

    return bands


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_income_estimation_model(
    features_df: pd.DataFrame,
    target_col: str = "true_monthly_income",
    feature_cols: list = None,
) -> dict:
    """Train XGBoost regressor for income estimation.

    Args:
        features_df: Feature matrix from features.build_feature_matrix()
        target_col: Ground truth income column
        feature_cols: Feature column names

    Returns:
        dict with model, metrics, predictions, confidence bands
    """
    if feature_cols is None:
        feature_cols = INCOME_ESTIMATION_FEATURES

    available_features = [f for f in feature_cols if f in features_df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"  ⚠️  Missing features (skipping): {missing}")

    X = features_df[available_features].copy()
    y = features_df[target_col].copy()

    print(f"\n  📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  📊 Income range: ₹{y.min():,.0f} — ₹{y.max():,.0f}")
    print(f"  📊 Median income: ₹{y.median():,.0f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # XGBoost Regressor
    xgb_params = {
        "objective": "reg:squarederror",
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)

    # Clip negative predictions
    y_pred_test = np.maximum(y_pred_test, 0)
    y_pred_all = np.maximum(y_pred_all, 0)

    # ---------------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    # MAPE (excluding near-zero incomes)
    mask = y_test > 1000
    mape = np.mean(np.abs(y_test[mask] - y_pred_test[mask]) / y_test[mask]) * 100

    # Median Absolute Percentage Error
    mdape = np.median(np.abs(y_test[mask] - y_pred_test[mask]) / y_test[mask]) * 100

    # Confidence bands
    cashflow_cv_test = features_df.loc[X_test.index, "cashflow_cv"] if "cashflow_cv" in features_df.columns else None
    occupation_test = features_df.loc[X_test.index, "occupation"] if "occupation" in features_df.columns else None
    bands_test = assign_confidence_band(
        y_pred_test, y_test.values,
        occupation=occupation_test,
        cashflow_cv=cashflow_cv_test,
    )

    band_counts = pd.Series(bands_test).value_counts().to_dict()

    # Confidence band accuracy
    band_accuracy = {}
    for band in ["High", "Medium", "Low"]:
        band_mask = bands_test == band
        if band_mask.sum() > 0:
            band_mape = np.mean(np.abs(y_test.values[band_mask] - y_pred_test[band_mask]) / np.maximum(y_test.values[band_mask], 1)) * 100
            band_accuracy[band] = round(band_mape, 1)

    # Income deviation detection (declared vs estimated)
    if "declared_monthly_income" in features_df.columns:
        declared = features_df.loc[X_test.index, "declared_monthly_income"].values
        deviation = (declared - y_pred_test) / np.maximum(y_pred_test, 1) * 100
        n_flagged = int((np.abs(deviation) > 20).sum())
        pct_flagged = round(n_flagged / len(deviation) * 100, 1)
    else:
        n_flagged = 0
        pct_flagged = 0.0

    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "mape_pct": round(mape, 1),
        "mdape_pct": round(mdape, 1),
        "confidence_band_distribution": band_counts,
        "confidence_band_mape": band_accuracy,
        "income_deviation_flagged_n": n_flagged,
        "income_deviation_flagged_pct": pct_flagged,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print(f"\n  📈 Income Estimation Performance:")
    print(f"     MAE:   ₹{metrics['mae']:>10,.0f}")
    print(f"     RMSE:  ₹{metrics['rmse']:>10,.0f}")
    print(f"     R²:     {metrics['r2_score']:.4f}")
    print(f"     MAPE:   {metrics['mape_pct']:.1f}%")
    print(f"     MdAPE:  {metrics['mdape_pct']:.1f}%")
    print(f"\n  📊 Confidence Band Distribution (test set):")
    for band, count in band_counts.items():
        mape_val = band_accuracy.get(band, "N/A")
        print(f"     {band:8s}: {count:4d} customers  (MAPE: {mape_val}%)")
    print(f"\n  🚩 Income deviation flags (>20%): {n_flagged} ({pct_flagged}%)")

    # Full-dataset predictions
    features_df = features_df.copy()
    features_df["estimated_income"] = y_pred_all.round(2)

    # Assign confidence bands for all
    cashflow_cv_all = features_df["cashflow_cv"] if "cashflow_cv" in features_df.columns else None
    occupation_all = features_df["occupation"] if "occupation" in features_df.columns else None
    features_df["income_confidence_band"] = assign_confidence_band(
        y_pred_all, y.values,
        occupation=occupation_all,
        cashflow_cv=cashflow_cv_all,
    )

    # Income deviation flag
    if "declared_monthly_income" in features_df.columns:
        dev = (features_df["declared_monthly_income"] - features_df["estimated_income"]) / \
              features_df["estimated_income"].replace(0, 1) * 100
        features_df["income_deviation_pct"] = dev.round(1)
        features_df["income_deviation_flag"] = (np.abs(dev) > 20).astype(int)

    # Feature importance
    importance = dict(zip(available_features, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  📊 Top 10 Feature Importances:")
    for feat, imp in list(importance.items())[:10]:
        print(f"     {feat:35s}: {imp:.4f}")

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_cols": available_features,
        "predictions_df": features_df,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(result: dict, model_path: str = MODEL_PATH, metrics_path: str = METRICS_PATH):
    """Save trained model and metrics."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": result["model"],
            "feature_cols": result["feature_cols"],
        }, f)
    print(f"  ✅ Model saved: {model_path}")

    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)
    print(f"  ✅ Metrics saved: {metrics_path}")


def load_model(model_path: str = MODEL_PATH) -> dict:
    """Load saved model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_income(
    features_df: pd.DataFrame,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """Predict income for new customers using saved model."""
    saved = load_model(model_path)
    model = saved["model"]
    feature_cols = saved["feature_cols"]

    available = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available].fillna(0)

    predicted = np.maximum(model.predict(X), 0)
    features_df = features_df.copy()
    features_df["estimated_income"] = predicted.round(2)

    # Confidence bands (heuristic-only at inference)
    cashflow_cv = features_df.get("cashflow_cv")
    occupation = features_df.get("occupation")
    features_df["income_confidence_band"] = assign_confidence_band(
        predicted, occupation=occupation, cashflow_cv=cashflow_cv
    )

    return features_df
