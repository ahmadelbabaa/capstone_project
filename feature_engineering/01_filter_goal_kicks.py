"""
01_filter_goal_kicks.py
-----------------------
Filters all merged match parquet files to extract only rows where
play_pattern = "From Goal Kick" and assigns a unique goal_kick_id
to each possession sequence.

Output
------
data/feature_engineering/goal_kick_sequences.parquet
    All tracking rows belonging to goal kick sequences, with goal_kick_id added.

goal_kick_id logic
------------------
A new sequence begins when, in the sorted (match, period, timestamp) order:
  - The match changes, OR
  - The period changes, OR
  - The possession_team_id changes, OR
  - There is a time gap > GAP_THRESHOLD seconds between consecutive rows
    (catches multiple goal kicks from the same team in the same period)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MERGED_DIR     = PROJECT_ROOT / "data" / "merged_j1_2024"
OUTPUT_DIR     = PROJECT_ROOT / "data" / "feature_engineering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE    = OUTPUT_DIR / "goal_kick_sequences.parquet"

# A gap larger than this (seconds) within the same match/period/team
# is treated as the start of a new goal kick sequence.
GAP_THRESHOLD  = 30.0

# ── Load & filter ──────────────────────────────────────────────────────────────
parquet_files = sorted(MERGED_DIR.glob("match_*.parquet"))
print(f"Found {len(parquet_files)} match files.")

chunks = []
for fp in parquet_files:
    df = pd.read_parquet(fp)
    gk = df[df["play_pattern"] == "From Goal Kick"]
    if not gk.empty:
        chunks.append(gk)

if not chunks:
    raise RuntimeError("No 'From Goal Kick' rows found across all match files.")

goal_kicks = pd.concat(chunks, ignore_index=True)
print(f"Rows after filtering: {len(goal_kicks):,}")

# ── Sort ───────────────────────────────────────────────────────────────────────
goal_kicks = goal_kicks.sort_values(
    ["skc_match_id", "period", "timestamp_seconds", "frame"],
    ignore_index=True
)

# ── Assign goal_kick_id ────────────────────────────────────────────────────────
# Build a boolean mask that is True at the start of every new sequence.
match_change   = goal_kicks["skc_match_id"]      != goal_kicks["skc_match_id"].shift(1)
period_change  = goal_kicks["period"]             != goal_kicks["period"].shift(1)
team_change    = goal_kicks["possession_team_id"] != goal_kicks["possession_team_id"].shift(1)
time_gap       = goal_kicks["timestamp_seconds"].diff().abs() > GAP_THRESHOLD

new_sequence   = match_change | period_change | team_change | time_gap
new_sequence.iloc[0] = True          # first row always starts a sequence

goal_kicks.insert(0, "goal_kick_id", new_sequence.cumsum().astype(int))

# ── Summary ────────────────────────────────────────────────────────────────────
n_sequences  = goal_kicks["goal_kick_id"].nunique()
n_matches    = goal_kicks["skc_match_id"].nunique()
frames_per   = goal_kicks.groupby("goal_kick_id")["frame"].nunique()
events_per   = goal_kicks.groupby("goal_kick_id")["event_id"].nunique()

print(f"\n-- Goal Kick Filtering Summary ------------------------------")
print(f"  Matches with goal kicks : {n_matches}")
print(f"  Total sequences         : {n_sequences}")
print(f"  Total rows              : {len(goal_kicks):,}")
print(f"  Avg frames / sequence   : {frames_per.mean():.1f}")
print(f"  Avg events / sequence   : {events_per.mean():.1f}")
print(f"  Possession teams        : {goal_kicks['possession_team_name'].nunique()}")
print(f"-------------------------------------------------------------\n")

# ── Column check ──────────────────────────────────────────────────────────────
expected_cols = [
    # Core identifiers
    "goal_kick_id", "event_id", "skc_match_id", "sb_match_id",
    "possession_team_id", "period", "timestamp_seconds",
    "event_location_x", "event_location_y",
    # OBV
    "obv_total_net", "obv_for_net", "obv_against_net",
    # Progression
    "pass_end_x", "carry_end_x",
    # Tracking
    "player_x", "player_y", "ball_x", "ball_y",
    "player_id", "skc_team_id", "player_role", "is_home",
]
missing = [c for c in expected_cols if c not in goal_kicks.columns]
if missing:
    print(f"WARNING: expected columns not found: {missing}")

# ── Save ───────────────────────────────────────────────────────────────────────
goal_kicks.to_parquet(OUTPUT_FILE, index=False)
print(f"Saved -> {OUTPUT_FILE}")
print(f"Shape : {goal_kicks.shape}")