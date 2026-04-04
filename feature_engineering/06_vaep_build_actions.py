"""
06_vaep_build_actions.py
------------------------
Extracts on-ball actions from all 376 merged match parquets, computes
per-action game-state features, derives lags, and assigns binary labels.

Output: one row per on-ball action (all 376 matches, periods 1 & 2 only).

Columns
-------
Identifiers : skc_match_id, sb_match_id, period, frame, event_id,
              team_id, possession_team_id, event_type, shot_outcome,
              goal_kick_id
Features    : a0_* (current action, 10 cols)
              a1_* (lag-1,          10 cols)
              a2_* (lag-2,          10 cols)
Labels      : scores   — 1 if same team scores within next K on-ball actions
              concedes — 1 if opposing team scores within next K on-ball actions

Game-state lags (PLAN.md §2.4)
-------------------------------
Lags copy the previous actions' raw feature values into the current row so
the classifier sees trajectory, not just a snapshot. They are computed via
pandas shift() within each (match, period) group. Period boundaries and the
first two actions of each period are filled with 0.

Binary labels (k=10)
---------------------
For each action by team T:
  scores   = 1  iff any of the next K on-ball actions (same match) is a
             Shot by team T with shot_outcome == 'Goal'
  concedes = 1  iff any of the next K on-ball actions is a goal by the
             opposing team

Labels are computed with a reverse rolling-sum trick (no Python loops).

Prerequisites
-------------
Run data_prep/enrich_merged_with_shot_outcomes.py first to add the
shot_outcome column to the merged parquets.

Input  : data/merged_j1_2024/match_*.parquet
         data/feature_engineering/goal_kick_sequences.parquet
Output : data/feature_engineering/on_ball_actions.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vaep_utils import (
    ON_BALL_TYPES, ACTION_TYPE_MAP, BASE_FEATURES, GAME_STATE_FEATURES,
    PITCH_LENGTH, PITCH_WIDTH, GOAL_X, GOAL_Y, K,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MERGED_FOLDER = PROJECT_ROOT / "data" / "merged_j1_2024"
GK_SEQ_FILE   = PROJECT_ROOT / "data" / "feature_engineering" / "goal_kick_sequences.parquet"
OUTPUT_FILE   = PROJECT_ROOT / "data" / "feature_engineering" / "on_ball_actions.parquet"

OUTPUT_COLS = [
    "skc_match_id", "sb_match_id", "period", "frame", "event_id",
    "team_id", "possession_team_id", "event_type", "shot_outcome",
    "goal_kick_id",
    "scores", "concedes",
] + GAME_STATE_FEATURES


# ── Per-match processing ───────────────────────────────────────────────────────

def process_match(df: pd.DataFrame, gk_lookup: dict) -> pd.DataFrame | None:
    """
    Process one match's merged parquet DataFrame.
    Returns a DataFrame with one row per on-ball action, or None if empty.
    """
    # ── 1. Deduplicate to one row per event; keep periods 1 & 2 only ──────────
    events = (
        df.drop_duplicates(subset="event_id")
        .dropna(subset=["event_id", "event_type"])
        .query("period in [1, 2]")
        .sort_values(["period", "frame"])
        .reset_index(drop=True)
        .copy()
    )

    if events.empty:
        return None

    # ── 2. Identify home team's StatsBomb ID ───────────────────────────────────
    home_name = events["home_team"].iloc[0]
    home_rows = events.loc[
        (events["event_team_name"] == home_name) & events["event_sb_team_id"].notna()
    ]
    home_team_id = home_rows["event_sb_team_id"].iloc[0] if not home_rows.empty else None

    # ── 3. Compute running score differential (vectorized) ────────────────────
    is_goal = (events["event_type"] == "Shot") & (events["shot_outcome"] == "Goal")

    if home_team_id is not None:
        home_goals = (is_goal & (events["event_sb_team_id"] == home_team_id)).astype(int)
        away_goals = (is_goal & (events["event_sb_team_id"] != home_team_id)).astype(int)
    else:
        home_goals = pd.Series(0, index=events.index)
        away_goals = pd.Series(0, index=events.index)

    # Shift by 1: the goal event itself sees the pre-goal score
    home_cum = home_goals.cumsum().shift(1, fill_value=0)
    away_cum = away_goals.cumsum().shift(1, fill_value=0)

    is_home_poss = events["possession_team_id"] == home_team_id
    events["score_diff"] = np.where(
        is_home_poss,
        home_cum - away_cum,
        away_cum - home_cum,
    ).clip(-5, 5)

    # ── 4. Filter to on-ball action types ─────────────────────────────────────
    actions = (
        events[events["event_type"].isin(ON_BALL_TYPES)]
        .copy()
        .reset_index(drop=True)
    )

    if actions.empty:
        return None

    # Rename for output clarity
    actions = actions.rename(columns={"event_sb_team_id": "team_id"})

    # ── 5. Build a0_* features (fully vectorized) ─────────────────────────────
    loc_x = actions["event_location_x"].fillna(PITCH_LENGTH / 2)
    loc_y = actions["event_location_y"].fillna(PITCH_WIDTH / 2)
    end_x = actions["pass_end_x"].fillna(actions["carry_end_x"]).fillna(loc_x)
    end_y = actions["pass_end_y"].fillna(actions["carry_end_y"]).fillna(loc_y)

    actions["a0_action_type"]   = actions["event_type"].map(ACTION_TYPE_MAP).fillna(9).astype(int)
    actions["a0_start_x"]       = (loc_x / PITCH_LENGTH).clip(0, 1)
    actions["a0_start_y"]       = (loc_y / PITCH_WIDTH).clip(0, 1)
    actions["a0_end_x"]         = (end_x / PITCH_LENGTH).clip(0, 1)
    actions["a0_end_y"]         = (end_y / PITCH_WIDTH).clip(0, 1)
    actions["a0_dist_to_goal"]  = (
        np.sqrt((loc_x - GOAL_X) ** 2 + (loc_y - GOAL_Y) ** 2) / PITCH_LENGTH
    )
    actions["a0_angle_to_goal"] = np.arctan2(loc_y - GOAL_Y, GOAL_X - loc_x) / np.pi
    actions["a0_score_diff"]    = actions["score_diff"].clip(-5, 5)
    actions["a0_minute_norm"]   = (actions["event_minute"].fillna(0) / 90.0).clip(0, 1)
    actions["a0_period"]        = actions["period"].astype(int)

    # ── 6. Game-state lags within each period ─────────────────────────────────
    # shift(1) and shift(2) within (period) groups; NaN → 0 at boundaries
    for lag, prefix in [(1, "a1"), (2, "a2")]:
        for col in BASE_FEATURES:
            actions[f"{prefix}_{col}"] = (
                actions.groupby("period")[f"a0_{col}"]
                .shift(lag)
                .fillna(0)
            )

    # ── 7. Binary labels: scores / concedes (k=10) ────────────────────────────
    # Initialise to 0; will be filled per team below
    actions["scores"]   = 0
    actions["concedes"] = 0

    is_goal_action = (
        (actions["event_type"] == "Shot") & (actions["shot_outcome"] == "Goal")
    ).astype(int)

    def fwd_k_positive(series: pd.Series) -> pd.Series:
        """Return 1 where the sum of the next K values (exclusive of self) > 0."""
        return (
            series.iloc[::-1]
            .rolling(K, min_periods=1)
            .sum()
            .iloc[::-1]
            .shift(-1)
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    for tid in actions["team_id"].dropna().unique():
        team_mask = actions["team_id"] == tid
        team_goals = is_goal_action.where(team_mask, other=0)
        opp_goals  = is_goal_action.where(~team_mask, other=0)

        actions.loc[team_mask, "scores"]   = fwd_k_positive(team_goals)[team_mask].values
        actions.loc[team_mask, "concedes"] = fwd_k_positive(opp_goals)[team_mask].values

    # ── 8. Attach goal_kick_id via (sb_match_id, event_id) lookup ─────────────
    actions["goal_kick_id"] = (
        actions[["sb_match_id", "event_id"]]
        .apply(lambda r: gk_lookup.get((r["sb_match_id"], r["event_id"]), np.nan), axis=1)
    )

    return actions[OUTPUT_COLS]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Load goal kick ID lookup ───────────────────────────────────────────────
    print("Loading goal kick sequences...")
    gk_seq = (
        pd.read_parquet(GK_SEQ_FILE)
        .drop_duplicates(subset="event_id")
        .dropna(subset=["event_id", "goal_kick_id"])
        [["sb_match_id", "event_id", "goal_kick_id"]]
    )
    gk_lookup = {
        (row.sb_match_id, row.event_id): row.goal_kick_id
        for row in gk_seq.itertuples(index=False)
    }
    print(f"  Goal kick events  : {len(gk_lookup):,}")

    # ── Process each match ────────────────────────────────────────────────────
    parquet_files = sorted(MERGED_FOLDER.glob("match_*.parquet"))
    print(f"\nProcessing {len(parquet_files)} match files...")

    all_actions = []
    skipped = []
    for i, fp in enumerate(parquet_files, 1):
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"  [SKIP] {fp.name} — {e}")
            skipped.append(fp.name)
            continue
        result = process_match(df, gk_lookup)
        if result is not None:
            all_actions.append(result)

        if i % 50 == 0 or i == len(parquet_files):
            print(f"  [{i}/{len(parquet_files)}] {fp.name}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} corrupted file(s): {skipped}")

    # ── Concatenate ───────────────────────────────────────────────────────────
    print("\nConcatenating all matches...")
    full = pd.concat(all_actions, ignore_index=True)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n-- Sanity Checks ------------------------------------------------")
    print(f"  Total actions     : {len(full):,}")
    print(f"  Matches           : {full['skc_match_id'].nunique()}")
    print(f"  scores=1 rate     : {full['scores'].mean():.3%}  (expected ~2–5%)")
    print(f"  concedes=1 rate   : {full['concedes'].mean():.3%}  (expected ~2–5%)")
    print(f"  Goal kick actions : {full['goal_kick_id'].notna().sum():,}")

    print(f"\n  Action type breakdown:")
    print(full["event_type"].value_counts().to_string())

    print(f"\n  Top event types when scores=1:")
    print(full[full["scores"] == 1]["event_type"].value_counts().head(6).to_string())

    print(f"\n  score_diff distribution:")
    print(full["a0_score_diff"].value_counts().sort_index().to_string())
    print("-----------------------------------------------------------------")

    # ── Save ──────────────────────────────────────────────────────────────────
    full.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")
    print(f"Shape  : {full.shape}")


if __name__ == "__main__":
    main()
