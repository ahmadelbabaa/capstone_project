"""
05_merge_features.py
---------------------
Merges all feature files into a single table with one row per goal_kick_id.

Sources
-------
obv_features.parquet              → core identifiers + OBV features
progression_features.parquet      → progression features
defensive_pressure_features.parquet → defensive pressure features

Output
------
data/feature_engineering/goal_kick_features.parquet
    One row per goal_kick_id, all features joined.
"""

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
FE_DIR = Path(__file__).resolve().parent.parent / "data" / "feature_engineering"

# ── Load ───────────────────────────────────────────────────────────────────────
obv  = pd.read_parquet(FE_DIR / "obv_features.parquet")
prog = pd.read_parquet(FE_DIR / "progression_features.parquet")
def_ = pd.read_parquet(FE_DIR / "defensive_pressure_features.parquet")

print(f"OBV features        : {obv.shape}")
print(f"Progression features: {prog.shape}")
print(f"Defensive features  : {def_.shape}")

# ── Merge ──────────────────────────────────────────────────────────────────────
# OBV is the base (it carries the core identifiers)
features = (
    obv
    .merge(prog, on="goal_kick_id", how="left")
    .merge(def_,  on="goal_kick_id", how="left")
)

# ── Sanity checks ──────────────────────────────────────────────────────────────
assert len(features) == len(obv), "Row count changed after merge — check for duplicates."
assert features["goal_kick_id"].is_unique,  "goal_kick_id is not unique after merge."

null_summary = features.isna().sum()
null_summary = null_summary[null_summary > 0]
if not null_summary.empty:
    print("\nNulls after merge:")
    print(null_summary.to_string())

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nFinal feature table : {features.shape}")
print(f"Columns             : {list(features.columns)}")

# ── Save ───────────────────────────────────────────────────────────────────────
OUT = FE_DIR / "goal_kick_features.parquet"
features.to_parquet(OUT, index=False)
print(f"\nSaved -> {OUT}")