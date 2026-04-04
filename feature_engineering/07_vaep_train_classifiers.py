"""
07_vaep_train_classifiers.py
-----------------------------
Trains two XGBoost binary classifiers on all on-ball actions:

  clf_scores   — P(team scores within next K actions | game state)
  clf_concedes — P(team concedes within next K actions | game state)

Both classifiers use the same 30-feature game-state input (a0_*, a1_*, a2_*)
and are trained on the FULL dataset (all 376 matches). There is no train/val
split here — these classifiers learn general football probabilities from every
action in the dataset. The 80/20 evaluation split lives in the final goal kick
model (Phase 2).

Why train on all data?
-----------------------
In goal kick sequences specifically, P(concede) ≈ 0 because sequences always
end when the opposing team wins the ball — so a goal kick sequence never
contains an opposition goal. Training only on goal kick data would collapse
the concedes classifier. Training on all match data gives the classifiers
proper calibration across the full range of football contexts.

Outputs
-------
  data/feature_engineering/value_classifiers.pkl
      Bundle: {clf_scores, clf_concedes, feature_cols, k}

  model/value_classifier_diagnostics.png
      Calibration curves + mean p_score by action type

Input  : data/feature_engineering/on_ball_actions.parquet
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from vaep_utils import GAME_STATE_FEATURES, ACTION_TYPE_MAP, K

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
INPUT_FILE     = PROJECT_ROOT / "data" / "feature_engineering" / "on_ball_actions.parquet"
OUTPUT_MODEL   = PROJECT_ROOT / "data" / "feature_engineering" / "value_classifiers.pkl"
OUTPUT_DIAG    = PROJECT_ROOT / "model" / "value_classifier_diagnostics.png"

# ── XGBoost hyperparameters ────────────────────────────────────────────────────
XGBOOST_PARAMS = dict(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    n_estimators     = 200,
    max_depth        = 3,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1,
)


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading on-ball actions...")
    actions = pd.read_parquet(INPUT_FILE)
    print(f"  Rows             : {len(actions):,}")
    print(f"  Matches          : {actions['skc_match_id'].nunique()}")

    # ── Feature matrix ────────────────────────────────────────────────────────
    X = actions[GAME_STATE_FEATURES].values
    y_scores   = actions["scores"].values
    y_concedes = actions["concedes"].values

    print(f"\n  scores=1 rate    : {y_scores.mean():.3%}")
    print(f"  concedes=1 rate  : {y_concedes.mean():.3%}")

    # ── Train classifiers (no scale_pos_weight) ───────────────────────────────
    # We do NOT use scale_pos_weight here. The classifiers need well-calibrated
    # probabilities so that differences between game states are meaningful in the
    # value formula. scale_pos_weight inflates all probabilities (mean p~0.39
    # instead of ~0.01), which amplifies noise and distorts the value signal.
    # The natural class imbalance is intentional — rare events (goals) should
    # produce small but meaningful probability shifts.
    print("\nTraining clf_scores...")
    clf_scores = XGBClassifier(**XGBOOST_PARAMS)
    clf_scores.fit(X, y_scores)
    print("  Done.")

    print("Training clf_concedes...")
    clf_concedes = XGBClassifier(**XGBOOST_PARAMS)
    clf_concedes.fit(X, y_concedes)
    print("  Done.")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\n-- Diagnostics --------------------------------------------------")

    p_score   = clf_scores.predict_proba(X)[:, 1]
    p_concede = clf_concedes.predict_proba(X)[:, 1]

    print(f"  p_score   : mean={p_score.mean():.4f}  std={p_score.std():.4f}  "
          f"min={p_score.min():.4f}  max={p_score.max():.4f}")
    print(f"  p_concede : mean={p_concede.mean():.4f}  std={p_concede.std():.4f}  "
          f"min={p_concede.min():.4f}  max={p_concede.max():.4f}")

    # Mean p_score by action type (sanity: shots should be highest)
    inv_map = {v: k for k, v in ACTION_TYPE_MAP.items()}
    actions["p_score"]   = p_score
    actions["p_concede"] = p_concede
    type_diag = (
        actions.groupby("event_type")[["p_score", "p_concede"]]
        .mean()
        .sort_values("p_score", ascending=False)
    )
    print(f"\n  Mean probabilities by action type:\n{type_diag.to_string()}")
    print("-----------------------------------------------------------------")

    # ── Calibration plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y, p, label in [
        (axes[0], y_scores,   p_score,   "Scores"),
        (axes[1], y_concedes, p_concede, "Concedes"),
    ]:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label="Classifier")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration — clf_{label.lower()}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    OUTPUT_DIAG.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIAG, dpi=150)
    plt.close()
    print(f"\nCalibration plot saved -> {OUTPUT_DIAG}")

    # ── Save model bundle ─────────────────────────────────────────────────────
    bundle = {
        "clf_scores":   clf_scores,
        "clf_concedes": clf_concedes,
        "feature_cols": GAME_STATE_FEATURES,
        "k":            K,
    }
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model bundle saved -> {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
