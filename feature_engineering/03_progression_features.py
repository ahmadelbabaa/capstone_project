"""
03_progression_features.py
--------------------------
Computes progression-based features for each goal kick sequence.

Features produced
-----------------
n_progressive_passes  : Number of passes where pass_end_x > event_location_x
                        (ball moves forward along the pitch)
n_progressive_carries : Number of carries where carry_end_x > event_location_x
                        (same forward-movement logic)
final_third_entry     : Boolean — True if any pass or carry ends in the
                        attacking final third (x > 80 in StatsBomb coords)

Coordinate system
-----------------
StatsBomb coords: x = 0 (own goal) → 120 (opponent goal), y = 0 → 80.
The pitch is split into three equal thirds along x:
  Defensive third  : x  0 – 40
  Middle third     : x 40 – 80
  Attacking third  : x 80 – 120   ← final third threshold = 80

The possession team always attacks left-to-right in StatsBomb, so
x > 80 is the final third from their perspective for all sequences.

Input  : data/feature_engineering/goal_kick_sequences.parquet
Output : data/feature_engineering/progression_features.parquet
"""

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE   = PROJECT_ROOT / "data" / "feature_engineering" / "goal_kick_sequences.parquet"
OUTPUT_FILE  = PROJECT_ROOT / "data" / "feature_engineering" / "progression_features.parquet"

FINAL_THIRD_X = 80.0   # StatsBomb x threshold for the attacking final third

# ── Load & deduplicate to event level ─────────────────────────────────────────
print("Loading goal kick sequences...")
df = pd.read_parquet(INPUT_FILE)
print(f"  Rows loaded   : {len(df):,}")
print(f"  Sequences     : {df['goal_kick_id'].nunique():,}")

events = df.drop_duplicates(subset="event_id").copy()
print(f"  Unique events : {len(events):,}")

# ── Filter by event type ───────────────────────────────────────────────────────
passes  = events[events["event_type"] == "Pass"].copy()
carries = events[events["event_type"] == "Carry"].copy()

print(f"  Pass events   : {len(passes):,}")
print(f"  Carry events  : {len(carries):,}")

# ── Progressive passes ─────────────────────────────────────────────────────────
# A pass is progressive if it moves the ball forward (end x > start x)
passes["is_progressive"] = passes["pass_end_x"] > passes["event_location_x"]

prog_passes = (
    passes.groupby("goal_kick_id")["is_progressive"]
    .sum()
    .reset_index()
    .rename(columns={"is_progressive": "n_progressive_passes"})
)

# ── Progressive carries ────────────────────────────────────────────────────────
carries["is_progressive"] = carries["carry_end_x"] > carries["event_location_x"]

prog_carries = (
    carries.groupby("goal_kick_id")["is_progressive"]
    .sum()
    .reset_index()
    .rename(columns={"is_progressive": "n_progressive_carries"})
)

# ── Final third entry ──────────────────────────────────────────────────────────
# True if any pass or carry ends with x > 80 in the sequence
passes["enters_final_third"]  = passes["pass_end_x"]  > FINAL_THIRD_X
carries["enters_final_third"] = carries["carry_end_x"] > FINAL_THIRD_X

final_third_passes  = passes.groupby("goal_kick_id")["enters_final_third"].any()
final_third_carries = carries.groupby("goal_kick_id")["enters_final_third"].any()

final_third = (
    final_third_passes
    .combine(final_third_carries, lambda a, b: a | b, fill_value=False)
    .reset_index()
    .rename(columns={"enters_final_third": "final_third_entry"})
)

# ── Combine all sequences (include those with zero passes/carries) ─────────────
all_ids = df[["goal_kick_id"]].drop_duplicates()

features = (
    all_ids
    .merge(prog_passes,  on="goal_kick_id", how="left")
    .merge(prog_carries, on="goal_kick_id", how="left")
    .merge(final_third,  on="goal_kick_id", how="left")
)

# Sequences with no passes or carries get 0, not NaN
features["n_progressive_passes"]  = features["n_progressive_passes"].fillna(0).astype(int)
features["n_progressive_carries"] = features["n_progressive_carries"].fillna(0).astype(int)
features["final_third_entry"]     = features["final_third_entry"].fillna(False).astype(bool)

# ── Sanity checks ──────────────────────────────────────────────────────────────
assert len(features) == df["goal_kick_id"].nunique(), \
    "Row count mismatch — every sequence must have exactly one feature row."
assert features.isna().sum().sum() == 0, \
    "Unexpected nulls in progression features."

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("-- Progression Feature Summary --------------------------")
print(f"  Sequences                      : {len(features):,}")
print(f"  n_progressive_passes  mean     : {features['n_progressive_passes'].mean():.2f}")
print(f"  n_progressive_passes  max      : {features['n_progressive_passes'].max()}")
print(f"  n_progressive_carries mean     : {features['n_progressive_carries'].mean():.2f}")
print(f"  n_progressive_carries max      : {features['n_progressive_carries'].max()}")
print(f"  final_third_entry = True       : {features['final_third_entry'].sum():,} "
      f"({features['final_third_entry'].mean()*100:.1f}%)")
print(f"  final_third_entry = False      : {(~features['final_third_entry']).sum():,} "
      f"({(~features['final_third_entry']).mean()*100:.1f}%)")
print("---------------------------------------------------------")
print()

# ── Save ───────────────────────────────────────────────────────────────────────
features.to_parquet(OUTPUT_FILE, index=False)
print(f"Saved -> {OUTPUT_FILE}")
print(f"Shape  : {features.shape}")
