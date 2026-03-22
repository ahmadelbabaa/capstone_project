"""Model loading and inference utilities."""
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH   = PROJECT_ROOT / "model" / "best_model_standardized.pkl"

# Human-readable feature labels
FEATURE_LABELS = {
    "obv_for_80":             "OBV For (first 80%)",
    "obv_against_80":         "OBV Against (first 80%)",
    "n_prog_passes_80":       "Progressive Passes",
    "n_prog_carries_80":      "Progressive Carries",
    "final_third_80":         "Final Third Entry",
    "press_compactness_80":   "Defensive Compactness (m)",
    "press_height_mean_x_80": "Def. Line Height — Mean X",
    "press_height_max_x_80":  "Def. Line Height — Max X",
    "def_line_depth_mean_x_80": "Def. Line Depth — Mean X",
    "avg_def_ball_dist_80":   "Avg Def-to-Ball Dist (m)",
    "min_def_ball_dist_80":   "Min Def-to-Ball Dist (m)",
    "n_frames_pressure_80":   "Frames Under Pressure",
    "press_fwd_speed_80":     "Def. Forward Speed (m/s)",
    "action_type_enc":        "Action Type at 80%",
    "action_x":               "Action Location X",
    "action_y":               "Action Location Y",
    "action_progressive":     "Action Progressive",
    "period":                 "Match Period",
}


@st.cache_resource
def load_model() -> dict:
    """Load the model bundle (Ridge + scaler + label_encoder + feature metadata)."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _encode_row(row: pd.Series, bundle: dict) -> np.ndarray:
    """Encode a single feature row into a scaled numpy array for prediction."""
    le = bundle["label_encoder"]
    feature_cols = bundle["feature_cols"]

    # Build a working copy with action_type_enc derived from action_type if needed
    row = row.copy()
    if "action_type_enc" not in row.index or pd.isna(row.get("action_type_enc")):
        raw = row.get("action_type", "Pass") or "Pass"
        try:
            row["action_type_enc"] = le.transform([raw])[0]
        except ValueError:
            row["action_type_enc"] = le.transform(["Pass"])[0]
    elif isinstance(row["action_type_enc"], str):
        try:
            row["action_type_enc"] = le.transform([row["action_type_enc"]])[0]
        except ValueError:
            row["action_type_enc"] = le.transform(["Pass"])[0]

    values = row[feature_cols].copy()

    # Fill any NaN with the scaler's learned means
    for i, col in enumerate(feature_cols):
        if pd.isna(values[col]):
            values[col] = bundle["scaler"].mean_[i]

    X = values.values.reshape(1, -1).astype(float)
    return bundle["scaler"].transform(X)


def predict_single(row: pd.Series, bundle: dict) -> dict:
    """
    Run model prediction for one goal kick sequence row.
    Returns predicted value, actual value, residual, and per-feature contributions.
    """
    X_scaled = _encode_row(row, bundle)
    predicted = float(bundle["model"].predict(X_scaled)[0])
    actual    = float(row["obv_remaining"]) if "obv_remaining" in row.index else float("nan")

    # Ridge: contribution of feature i = coef[i] * scaled_x[i]
    contributions = pd.Series(
        bundle["model"].coef_ * X_scaled[0],
        index=bundle["feature_cols"],
    )

    return {
        "predicted":           predicted,
        "actual":              actual,
        "residual":            actual - predicted,
        "feature_contributions": contributions,
    }


@st.cache_data
def predict_batch(_bundle_key, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions over the full model features DataFrame.
    _bundle_key is unused but forces Streamlit to re-cache if the model changes.
    Returns df with added columns: predicted_obv, residual.
    """
    bundle = load_model()
    le     = bundle["label_encoder"]
    feature_cols = bundle["feature_cols"]

    result = df.copy()

    # Encode action_type column
    def safe_encode(val):
        if isinstance(val, str):
            try:
                return le.transform([val])[0]
            except ValueError:
                return le.transform(["Pass"])[0]
        return val

    result = result.copy()
    result["action_type_enc"] = result["action_type"].apply(safe_encode)

    # Fill any remaining NaN values with the scaler's learned means
    X_df = result[feature_cols].copy()
    for i, col in enumerate(feature_cols):
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(bundle["scaler"].mean_[i])

    X = X_df.values.astype(float)
    X_scaled = bundle["scaler"].transform(X)
    result["predicted_obv"] = bundle["model"].predict(X_scaled)
    result["residual"]      = result["obv_remaining"] - result["predicted_obv"]
    return result
