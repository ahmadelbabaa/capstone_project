"""
08_score_actions.py
--------------------
Applies the possession value formula to every on-ball action.

Formula
-------
action_prog_value = (p_score_after  - p_score_before)
                  - (p_concede_after - p_concede_before)

Where "before" = game state before action i (action i-1 is most recent):
  a0 ← current a1
  a1 ← current a2
  a2 ← zeros (no information three actions ago)

And "after" = game state after action i (current a0, a1, a2).

Positive value: the action improved the team's position (increased scoring
chance and/or reduced conceding chance).
Negative value: the action worsened the team's position.

Output columns (all from input, plus):
  p_score           — P(score in next K actions | state after action i)
  p_concede         — P(concede in next K actions | state after action i)
  action_prog_value — the possession value of this action

Input  : data/feature_engineering/on_ball_actions.parquet
         data/feature_engineering/value_classifiers.pkl
Output : data/feature_engineering/scored_actions.parquet
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from vaep_utils import GAME_STATE_FEATURES, BASE_FEATURES

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
INPUT_ACTIONS = PROJECT_ROOT / "data" / "feature_engineering" / "on_ball_actions.parquet"
INPUT_MODEL   = PROJECT_ROOT / "data" / "feature_engineering" / "value_classifiers.pkl"
OUTPUT_FILE   = PROJECT_ROOT / "data" / "feature_engineering" / "scored_actions.parquet"


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading on-ball actions...")
    actions = pd.read_parquet(INPUT_ACTIONS)
    print(f"  Rows             : {len(actions):,}")

    print("Loading value classifiers...")
    with open(INPUT_MODEL, "rb") as f:
        bundle = pickle.load(f)
    clf_scores   = bundle["clf_scores"]
    clf_concedes = bundle["clf_concedes"]
    feature_cols = bundle["feature_cols"]   # == GAME_STATE_FEATURES
    print(f"  Classifiers loaded (k={bundle['k']})")

    # ── Build X_after (state after action i) ──────────────────────────────────
    # Current game state: a0=action i, a1=action i-1, a2=action i-2
    X_after = actions[feature_cols].values

    # ── Build X_before (state before action i) ────────────────────────────────
    # Before action i was taken, action i-1 was the most recent.
    # Shift the game state back one step:
    #   new a0 = current a1  (action i-1 becomes most recent)
    #   new a1 = current a2  (action i-2 shifts back)
    #   new a2 = zeros       (no information three actions ago)
    n_features = len(BASE_FEATURES)   # 10

    a1_cols = [f"a1_{f}" for f in BASE_FEATURES]
    a2_cols = [f"a2_{f}" for f in BASE_FEATURES]

    X_before = np.hstack([
        actions[a1_cols].values,              # becomes new a0
        actions[a2_cols].values,              # becomes new a1
        np.zeros((len(actions), n_features)), # new a2 = zeros
    ])

    # ── Predict probabilities ─────────────────────────────────────────────────
    print("\nPredicting P(score) and P(concede)...")
    p_score_after    = clf_scores.predict_proba(X_after)[:, 1]
    p_score_before   = clf_scores.predict_proba(X_before)[:, 1]
    p_concede_after  = clf_concedes.predict_proba(X_after)[:, 1]
    p_concede_before = clf_concedes.predict_proba(X_before)[:, 1]

    # ── Apply value formula ───────────────────────────────────────────────────
    actions["p_score"]           = p_score_after
    actions["p_concede"]         = p_concede_after
    actions["action_prog_value"] = (
        (p_score_after  - p_score_before) -
        (p_concede_after - p_concede_before)
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\n-- Diagnostics --------------------------------------------------")
    v = actions["action_prog_value"]
    print(f"  action_prog_value : mean={v.mean():.5f}  std={v.std():.5f}")
    print(f"                      min={v.min():.4f}   max={v.max():.4f}")
    print(f"  Positive actions  : {(v > 0).mean():.1%}")
    print(f"  Negative actions  : {(v < 0).mean():.1%}")

    print(f"\n  Mean action_prog_value by event type:")
    type_vals = (
        actions.groupby("event_type")["action_prog_value"]
        .mean()
        .sort_values(ascending=False)
    )
    print(type_vals.to_string())

    print(f"\n  Goal kick actions : {actions['goal_kick_id'].notna().sum():,}")
    gk = actions[actions["goal_kick_id"].notna()]
    if not gk.empty:
        print(f"  GK mean value     : {gk['action_prog_value'].mean():.5f}")
    print("-----------------------------------------------------------------")

    # ── Save ──────────────────────────────────────────────────────────────────
    actions.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")
    print(f"Shape  : {actions.shape}")


if __name__ == "__main__":
    main()
