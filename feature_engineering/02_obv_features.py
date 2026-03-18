"""
02_obv_features.py
------------------
Computes OBV (On-Ball Value) sequence-level features for each goal kick.

OBV values are attached per StatsBomb event. Since the goal kick sequences
parquet has one row per (frame, player), we first deduplicate to one row per
event, then aggregate per goal_kick_id.

Features produced
-----------------
obv_total_seq   : Cumulative net OBV over the entire sequence
                  (sum of obv_total_net across all events)
obv_for_seq     : Cumulative OBV for the possession team over the sequence
                  (sum of obv_for_net)
obv_against_seq : Cumulative OBV against the possession team over the sequence
                  (sum of obv_against_net)

Note: Events with no OBV value (Ball Recovery, Pressure, Duel, etc.) are
      treated as contributing 0 to the cumulative total.

Input  : data/feature_engineering/goal_kick_sequences.parquet
Output : data/feature_engineering/obv_features.parquet
"""

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE   = PROJECT_ROOT / "data" / "feature_engineering" / "goal_kick_sequences.parquet"
OUTPUT_FILE  = PROJECT_ROOT / "data" / "feature_engineering" / "obv_features.parquet"

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading goal kick sequences...")
df = pd.read_parquet(INPUT_FILE)
print(f"  Rows loaded     : {len(df):,}")
print(f"  Sequences       : {df['goal_kick_id'].nunique():,}")

# ── Deduplicate to one row per event ──────────────────────────────────────────
# The sequences file has one row per (frame, player). OBV values are identical
# across all rows sharing the same event_id, so we keep just the first.
events = df.drop_duplicates(subset="event_id").copy()
print(f"  Unique events   : {len(events):,}")

# ── Core identifiers (one value per sequence) ─────────────────────────────────
# Take the first event of each sequence for the sequence-level identifiers.
identifiers = (
    events.sort_values("timestamp_seconds")
    .groupby("goal_kick_id", sort=False)
    .first()
    .reset_index()[
        [
            "goal_kick_id",
            "skc_match_id",
            "sb_match_id",
            "possession_team_id",
            "possession_team_name",
            "period",
            "timestamp_seconds",
            "event_location_x",
            "event_location_y",
        ]
    ]
    .rename(columns={"timestamp_seconds": "sequence_start_time"})
)

# ── OBV aggregation ───────────────────────────────────────────────────────────
# NaN → 0 before summing (events without OBV contribute nothing)
obv_cols = ["obv_total_net", "obv_for_net", "obv_against_net"]
events[obv_cols] = events[obv_cols].fillna(0)

obv_agg = (
    events.groupby("goal_kick_id")[obv_cols]
    .sum()
    .reset_index()
    .rename(columns={
        "obv_total_net"   : "obv_total_seq",
        "obv_for_net"     : "obv_for_seq",
        "obv_against_net" : "obv_against_seq",
    })
)

# ── Merge identifiers + OBV features ─────────────────────────────────────────
features = identifiers.merge(obv_agg, on="goal_kick_id", how="left")

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert len(features) == df["goal_kick_id"].nunique(), \
    "Row count mismatch — every sequence must have exactly one feature row."

assert features["obv_total_seq"].isna().sum() == 0, \
    "Unexpected nulls in obv_total_seq after fillna."

print()
print("-- OBV Feature Summary ------------------------------")
print(f"  Sequences         : {len(features):,}")
print(f"  obv_total_seq     : mean={features['obv_total_seq'].mean():.4f}  "
      f"std={features['obv_total_seq'].std():.4f}")
print(f"  obv_for_seq       : mean={features['obv_for_seq'].mean():.4f}  "
      f"std={features['obv_for_seq'].std():.4f}")
print(f"  obv_against_seq   : mean={features['obv_against_seq'].mean():.4f}  "
      f"std={features['obv_against_seq'].std():.4f}")
print(f"  Nulls in any col  : {features.isna().sum().sum()}")
print("-----------------------------------------------------")
print()

# ── Save ───────────────────────────────────────────────────────────────────────
features.to_parquet(OUTPUT_FILE, index=False)
print(f"Saved -> {OUTPUT_FILE}")
print(f"Shape  : {features.shape}")
