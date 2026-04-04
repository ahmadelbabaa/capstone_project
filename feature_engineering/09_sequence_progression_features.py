"""
09_sequence_progression_features.py
-------------------------------------
Phase 2, Step 1 — aggregate action-level progression values into
sequence-level features for each goal kick, split by the 80/20 window.

Outputs
-------
prog_value_80               : sum of action_prog_value in first 80% of sequence
                              → used as a MODEL FEATURE
prog_value_20               : sum of action_prog_value in last 20% of sequence
                              → used as the MODEL TARGET (replaces obv_remaining)
sequence_progression_value  : total (80% + 20%), kept for reference
mean_action_prog_value      : mean over all actions in the sequence
max_action_prog_value       : peak single-action value
n_prog_actions              : total on-ball actions in sequence

The 80/20 split is derived from the same frame cutoff used in
extract_80_20_features.py — first 80% of unique frames per sequence.

Input  : data/feature_engineering/scored_actions.parquet
         data/feature_engineering/goal_kick_sequences.parquet  (for in_80 flags)
         data/feature_engineering/model_features_80_20.parquet (coverage check)
Output : data/feature_engineering/sequence_progression_features.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
INPUT_SCORED  = PROJECT_ROOT / "data" / "feature_engineering" / "scored_actions.parquet"
GK_SEQ_FILE   = PROJECT_ROOT / "data" / "feature_engineering" / "goal_kick_sequences.parquet"
INPUT_MODEL   = PROJECT_ROOT / "data" / "feature_engineering" / "model_features_80_20.parquet"
OUTPUT_FILE   = PROJECT_ROOT / "data" / "feature_engineering" / "sequence_progression_features.parquet"


def assign_window(g):
    """Return in_80 flag (1 = first 80% of frames, 0 = last 20%) per row."""
    frames = sorted(g["frame"].unique())
    n      = len(frames)
    cut    = int(np.floor(n * 0.8))
    frames_80 = set(frames[:cut])
    return g["frame"].isin(frames_80).astype(int)


def main():
    # ── Load scored actions ────────────────────────────────────────────────────
    print("Loading scored actions...")
    actions = pd.read_parquet(INPUT_SCORED)
    print(f"  Total actions     : {len(actions):,}")

    # ── Filter to goal kick actions by the possession team only ───────────────
    gk_actions = actions[
        actions["goal_kick_id"].notna() &
        (actions["team_id"] == actions["possession_team_id"])
    ].copy()
    print(f"  Goal kick actions (possession team only): {len(gk_actions):,}")
    print(f"  Goal kick seqs    : {gk_actions['goal_kick_id'].nunique():,}")

    # ── Derive in_80 flag from goal_kick_sequences frame cutoff ───────────────
    print("\nLoading frame window flags from goal_kick_sequences...")
    seq_frames = (
        pd.read_parquet(GK_SEQ_FILE, columns=["goal_kick_id", "frame"])
        .drop_duplicates(["goal_kick_id", "frame"])
    )
    # Recompute the same 80/20 cutoff used in extract_80_20_features.py
    seq_frames["in_80"] = (
        seq_frames.groupby("goal_kick_id", group_keys=False)
        .apply(assign_window, include_groups=False)
    )

    # Join window flag onto scored actions via (goal_kick_id, frame)
    gk_actions = gk_actions.merge(
        seq_frames[["goal_kick_id", "frame", "in_80"]],
        on=["goal_kick_id", "frame"],
        how="left",
    )
    # Actions whose frame doesn't appear in sequences file fall back to in_80=1
    gk_actions["in_80"] = gk_actions["in_80"].fillna(1).astype(int)

    matched = gk_actions["in_80"].notna().sum()
    print(f"  Actions with window flag : {matched:,}")

    # ── Split into 80% and 20% windows ────────────────────────────────────────
    gk_80 = gk_actions[gk_actions["in_80"] == 1]
    gk_20 = gk_actions[gk_actions["in_80"] == 0]

    print(f"  Actions in first 80%     : {len(gk_80):,}")
    print(f"  Actions in last 20%      : {len(gk_20):,}")

    # ── Aggregate per sequence ─────────────────────────────────────────────────
    prog_80 = (
        gk_80.groupby("goal_kick_id")["action_prog_value"]
        .sum()
        .rename("prog_value_80")
    )

    prog_20 = (
        gk_20.groupby("goal_kick_id")["action_prog_value"]
        .sum()
        .rename("prog_value_20")
    )

    # Total and other aggregates (all actions, kept for reference)
    seq_agg = (
        gk_actions
        .groupby("goal_kick_id")["action_prog_value"]
        .agg(
            sequence_progression_value = "sum",
            mean_action_prog_value     = "mean",
            max_action_prog_value      = "max",
            n_prog_actions             = "count",
        )
    )

    seq_features = (
        seq_agg
        .join(prog_80, how="left")
        .join(prog_20, how="left")
        .reset_index()
    )

    # Fill zeros for sequences with no actions in a window
    for col in ["prog_value_80", "prog_value_20"]:
        seq_features[col] = seq_features[col].fillna(0)

    # ── Sanity check: all model goal_kick_ids must be covered ─────────────────
    print("\nVerifying coverage against model_features_80_20...")
    model_df  = pd.read_parquet(INPUT_MODEL, columns=["goal_kick_id"])
    model_ids = set(model_df["goal_kick_id"].dropna().unique())
    output_ids = set(seq_features["goal_kick_id"].unique())

    missing = model_ids - output_ids
    extra   = output_ids - model_ids

    print(f"  Model sequences    : {len(model_ids):,}")
    print(f"  Output sequences   : {len(output_ids):,}")
    print(f"  Missing from output: {len(missing)}")
    print(f"  Extra in output    : {len(extra)}")

    if missing:
        print(f"  NOTE: {len(missing)} model goal_kick_ids have no on-ball actions — filling with 0")
        missing_df = pd.DataFrame({
            "goal_kick_id":               list(missing),
            "sequence_progression_value": 0.0,
            "mean_action_prog_value":     0.0,
            "max_action_prog_value":      0.0,
            "n_prog_actions":             0,
            "prog_value_80":              0.0,
            "prog_value_20":              0.0,
        })
        seq_features = pd.concat([seq_features, missing_df], ignore_index=True)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    for col, label in [
        ("prog_value_80", "prog_value_80  (feature — first 80%)"),
        ("prog_value_20", "prog_value_20  (target  — last  20%)"),
        ("sequence_progression_value", "sequence_progression_value (total)"),
    ]:
        v = seq_features[col]
        print(f"\n-- {label} --")
        print(f"  mean : {v.mean():.4f}  std : {v.std():.4f}")
        print(f"  min  : {v.min():.4f}  max : {v.max():.4f}")
        print(f"  >0   : {(v > 0).mean():.1%}  <0 : {(v < 0).mean():.1%}")

    print(f"\n  Correlation prog_value_80 vs prog_value_20 : "
          f"{seq_features['prog_value_80'].corr(seq_features['prog_value_20']):.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    seq_features.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")
    print(f"Shape  : {seq_features.shape}")
    print(f"Columns: {list(seq_features.columns)}")


if __name__ == "__main__":
    main()
