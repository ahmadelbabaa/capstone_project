"""
enrich_merged_with_shot_outcomes.py
------------------------------------
Adds shot_outcome column to all merged parquet files.

The merged parquets have event_type == 'Shot' rows but lack the outcome
(Goal / Saved / Blocked / Off T / etc.) since that detail lives in the
raw StatsBomb events JSON. This script:

  1. Loads sb_events.json once and extracts {event_id: shot_outcome} for
     all Shot events.
  2. Iterates over every merged parquet, joins the outcome, and saves in-place.

New column
----------
shot_outcome : str | NaN
    Outcome of a Shot event (e.g. 'Goal', 'Saved', 'Blocked', 'Off T',
    'Wayward', 'Post'). NaN for all non-Shot events.

Input  : data/sb_data/sb_events.json
         data/merged_j1_2024/match_*.parquet  (376 files)
Output : same parquet files, updated in-place
"""

import json
import glob
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
EVENTS_FILE   = PROJECT_ROOT / "data" / "sb_data" / "sb_events.json"
MERGED_FOLDER = PROJECT_ROOT / "data" / "merged_j1_2024"

# ── Step 1: Build shot outcome lookup from raw JSON ────────────────────────────
print("Loading StatsBomb events JSON...")
with open(EVENTS_FILE, "r", encoding="utf-8") as f:
    raw_events = json.load(f)

shot_outcomes = {
    e["id"]: e["shot.outcome.name"]
    for e in raw_events
    if e.get("type.name") == "Shot" and e.get("shot.outcome.name") is not None
}

print(f"  Shot events found : {len(shot_outcomes):,}")
print(f"  Unique outcomes   : {set(shot_outcomes.values())}")
print(f"  Goals             : {sum(1 for v in shot_outcomes.values() if v == 'Goal')}")

# ── Step 2: Patch each merged parquet ─────────────────────────────────────────
parquet_files = sorted(MERGED_FOLDER.glob("match_*.parquet"))
print(f"\nPatching {len(parquet_files)} parquet files...")

for i, fp in enumerate(parquet_files, 1):
    df = pd.read_parquet(fp)

    # Map shot outcomes via event_id; NaN for non-Shot events
    df["shot_outcome"] = df["event_id"].map(shot_outcomes)

    df.to_parquet(fp, index=False)

    if i % 50 == 0 or i == len(parquet_files):
        print(f"  [{i}/{len(parquet_files)}] {fp.name}")

# ── Verification ───────────────────────────────────────────────────────────────
print("\nVerification on last file:")
sample = pd.read_parquet(parquet_files[-1])
shot_rows = sample[sample["event_type"] == "Shot"].drop_duplicates("event_id")
print(f"  Shot events       : {len(shot_rows)}")
print(f"  shot_outcome dist :\n{shot_rows['shot_outcome'].value_counts().to_string()}")
print(f"  Non-shot nulls    : {sample[sample['event_type'] != 'Shot']['shot_outcome'].isna().all()}")

print("\nDone.")
