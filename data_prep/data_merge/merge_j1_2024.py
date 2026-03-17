"""
Merge J1 League 2024 SkillCorner tracking data with StatsBomb event data.

Data sources:
  - Tracking: data/tracking_j1_2024/{skc_id}_tracking_extrapolated.jsonl
  - Events:   data/sb_data/sb_events.json
  - Mapping:  data/mapping_ids/skc_sb_match_mapping.csv
  - Metadata: data/metadata_j1_2024/{skc_id}_metadata.json

Output: data/merged_j1_2024/match_{skc_id}.parquet  (one file per match)
"""

import json
import os
import glob
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR          = ROOT / "data"
TRACKING_DIR      = DATA_DIR / "tracking_j1_2024"
METADATA_DIR      = DATA_DIR / "metadata_j1_2024"
SB_DATA_DIR       = DATA_DIR / "sb_data"
MAPPING_FILE      = DATA_DIR / "mapping_ids" / "skc_sb_match_mapping.csv"
EVENTS_FILE       = SB_DATA_DIR / "sb_events.json"
LINEUPS_FILE      = SB_DATA_DIR / "sb_lineups.json"
OUTPUT_DIR        = DATA_DIR / "merged_j1_2024"

# ── Constants ─────────────────────────────────────────────────────────────────
# Priority event types for conflict resolution when multiple events share a timestamp window
PRIORITY_EVENTS = ["Pass", "Carry", "Shot", "Duel", "Ball Recovery"]

# Subsample: keep one tracking frame every N seconds (0 = keep all)
DEFAULT_SUBSAMPLE_SECONDS = 0.2


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS.ss' or 'HH:MM:SS.sss' to total seconds."""
    if ts is None:
        return float("nan")
    try:
        parts = ts.split(":")
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:
        return float("nan")


def load_match_mapping() -> pd.DataFrame:
    """Load SkillCorner -> StatsBomb match ID mapping."""
    df = pd.read_csv(MAPPING_FILE)
    # Expected columns: skc_match_id, sb_match_id, sb_home_team_id, date, home_team, away_team
    return df


def load_events(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load all StatsBomb events and add a timestamp_seconds column.
    Filter to only matches that exist in the mapping.
    """
    print("Loading StatsBomb events...")
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)

    events_df = pd.DataFrame(events)

    # Keep only matches present in our mapping
    sb_ids = set(mapping_df["sb_match_id"].tolist())
    events_df = events_df[events_df["match_id"].isin(sb_ids)].copy()
    print(f"  Loaded {len(events_df):,} events across {events_df['match_id'].nunique()} matches")

    # Parse timestamp to seconds for merge_asof
    events_df["timestamp_seconds"] = events_df["timestamp"].apply(parse_timestamp)

    # Extract nested coordinates to flat columns
    events_df["event_location_x"] = events_df["location"].apply(
        lambda loc: float(loc[0]) if isinstance(loc, list) and len(loc) >= 2 else None
    )
    events_df["event_location_y"] = events_df["location"].apply(
        lambda loc: float(loc[1]) if isinstance(loc, list) and len(loc) >= 2 else None
    )
    events_df["pass_end_x"] = events_df["pass.end_location"].apply(
        lambda loc: float(loc[0]) if isinstance(loc, list) and len(loc) >= 2 else None
    )
    events_df["pass_end_y"] = events_df["pass.end_location"].apply(
        lambda loc: float(loc[1]) if isinstance(loc, list) and len(loc) >= 2 else None
    )
    events_df["carry_end_x"] = events_df["carry.end_location"].apply(
        lambda loc: float(loc[0]) if isinstance(loc, list) and len(loc) >= 2 else None
    )
    events_df["carry_end_y"] = events_df["carry.end_location"].apply(
        lambda loc: float(loc[1]) if isinstance(loc, list) and len(loc) >= 2 else None
    )
    events_df["shot_end_x"] = events_df["shot.end_location"].apply(
        lambda loc: float(loc[0]) if isinstance(loc, list) and len(loc) >= 3 else None
    )
    events_df["shot_end_y"] = events_df["shot.end_location"].apply(
        lambda loc: float(loc[1]) if isinstance(loc, list) and len(loc) >= 3 else None
    )

    return events_df


def load_lineups() -> dict:
    """Load StatsBomb lineups keyed by match_id (string)."""
    with open(LINEUPS_FILE, "r", encoding="utf-8") as f:
        lineups = json.load(f)
    return lineups  # {str(sb_match_id): [{team_id, lineup: [{player_id, player_name, ...}]}]}


def build_player_lookup(skc_match_id: int) -> dict:
    """
    Build a dict {player_id (SkillCorner): {name, team_id, jersey_number, position, is_home}}
    from the SkillCorner metadata JSON for a single match.
    """
    meta_file = METADATA_DIR / f"{skc_match_id}_metadata.json"
    if not meta_file.exists():
        print(f"  Warning: metadata not found for match {skc_match_id}")
        return {}

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    home_team_id = meta["home_team"]["id"]

    lookup = {}
    for p in meta.get("players", []):
        pid = p.get("id")
        if pid is None:
            continue
        lookup[pid] = {
            "player_name":    p.get("short_name"),
            "skc_team_id":    p.get("team_id"),
            "jersey_number":  p.get("number"),
            "player_role":    p.get("player_role", {}).get("name") if p.get("player_role") else None,
            "is_home":        p.get("team_id") == home_team_id,
        }
    return lookup


def load_tracking_frames(skc_match_id: int) -> list:
    """Load raw tracking frames from JSONL (actually a JSON array) file."""
    tracking_file = TRACKING_DIR / f"{skc_match_id}_tracking_extrapolated.jsonl"
    if not tracking_file.exists():
        print(f"  Warning: tracking file not found for match {skc_match_id}")
        return []
    with open(tracking_file, "r", encoding="utf-8") as f:
        return json.load(f)


def explode_tracking(frames: list, player_lookup: dict) -> pd.DataFrame:
    """
    Convert raw tracking frames into a flat DataFrame with one row per
    (frame, player).  Ball data is duplicated onto every player row.
    """
    rows = []
    for frame in frames:
        ts_str   = frame.get("timestamp")
        period   = frame.get("period")
        frame_no = frame.get("frame")

        if ts_str is None or period is None:
            continue

        ts_sec = parse_timestamp(ts_str)

        ball       = frame.get("ball_data") or {}
        possession = frame.get("possession") or {}
        player_list = frame.get("player_data") or []

        for player in player_list:
            pid = player.get("player_id")
            meta_info = player_lookup.get(pid, {})
            rows.append({
                "frame":              frame_no,
                "period":             period,
                "timestamp_seconds":  ts_sec,
                "ball_x":             ball.get("x"),
                "ball_y":             ball.get("y"),
                "ball_z":             ball.get("z"),
                "ball_detected":      ball.get("is_detected"),
                "possession_player_id": possession.get("player_id"),
                "possession_group":   possession.get("group"),
                "player_id":          pid,
                "player_x":           player.get("x"),
                "player_y":           player.get("y"),
                "player_detected":    player.get("is_detected"),
                # From metadata
                "player_name":        meta_info.get("player_name"),
                "skc_team_id":        meta_info.get("skc_team_id"),
                "jersey_number":      meta_info.get("jersey_number"),
                "player_role":        meta_info.get("player_role"),
                "is_home":            meta_info.get("is_home"),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def subsample_tracking(tracking_df: pd.DataFrame, subsample_seconds: float) -> pd.DataFrame:
    """
    Keep only frames whose timestamp is approximately divisible by subsample_seconds.
    Applied per period to avoid cross-period bleed.
    """
    if subsample_seconds <= 0:
        return tracking_df

    kept_frames = []
    for period, group in tracking_df.groupby("period"):
        unique_ts = sorted(group["timestamp_seconds"].unique())
        selected = [
            ts for ts in unique_ts
            if (ts % subsample_seconds < 0.001 or abs(ts % subsample_seconds - subsample_seconds) < 0.001)
        ]
        kept_frames.append(group[group["timestamp_seconds"].isin(selected)])

    result = pd.concat(kept_frames, ignore_index=True)
    orig_frames  = tracking_df["frame"].nunique()
    final_frames = result["frame"].nunique()
    print(f"  Subsampling: {orig_frames:,} -> {final_frames:,} unique frames "
          f"({final_frames/orig_frames*100:.1f}% preserved)")
    return result


def filter_priority_events(events_df: pd.DataFrame,
                            priority_types: list = PRIORITY_EVENTS) -> pd.DataFrame:
    """
    When multiple events share the same 1-second window, keep only the highest-
    priority type.  Single-event windows are always kept untouched.
    """
    events_df = events_df.copy()
    events_df["_time_window"] = events_df["timestamp_seconds"].round(0)

    filtered = []
    for period, pgrp in events_df.groupby("period"):
        for tw, wgrp in pgrp.groupby("_time_window"):
            if len(wgrp) == 1:
                filtered.append(wgrp)
            else:
                priority = wgrp[wgrp["type.name"].isin(priority_types)]
                if len(priority) > 0:
                    filtered.append(priority)
                else:
                    filtered.append(wgrp.iloc[:1])  # keep first event

    result = pd.concat(filtered, ignore_index=True).drop(columns=["_time_window"])
    print(f"  Event filtering: {len(events_df):,} -> {len(result):,} events")
    return result


def select_event_columns(events_df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename StatsBomb event columns for the merged output."""
    col_map = {
        "id":                   "event_id",
        "type.name":            "event_type",
        "minute":               "event_minute",
        "second":               "event_second",
        "timestamp":            "event_timestamp",
        "timestamp_seconds":    "timestamp_seconds",   # keep for merge key
        "period":               "period",              # keep for merge key
        "duration":             "event_duration",
        "team.id":              "event_sb_team_id",
        "team.name":            "event_team_name",
        "player.id":            "event_sb_player_id",
        "player.name":          "event_player_name",
        "possession_team.id":   "possession_team_id",
        "possession_team.name": "possession_team_name",
        "play_pattern.name":    "play_pattern",
        "event_location_x":     "event_location_x",
        "event_location_y":     "event_location_y",
        "pass_end_x":           "pass_end_x",
        "pass_end_y":           "pass_end_y",
        "carry_end_x":          "carry_end_x",
        "carry_end_y":          "carry_end_y",
        "shot_end_x":           "shot_end_x",
        "shot_end_y":           "shot_end_y",
        # On-Ball Value (OBV) metrics from StatsBomb
        "obv_for_net":          "obv_for_net",
        "obv_against_net":      "obv_against_net",
        "obv_total_net":        "obv_total_net",
        "obv_for_before":       "obv_for_before",
        "obv_for_after":        "obv_for_after",
        "obv_against_before":   "obv_against_before",
        "obv_against_after":    "obv_against_after",
    }
    available = {k: v for k, v in col_map.items() if k in events_df.columns}
    return events_df[list(available.keys())].rename(columns=available)


def merge_match(
    skc_match_id: int,
    sb_match_id:  int,
    home_team:    str,
    away_team:    str,
    tracking_df:  pd.DataFrame,
    events_df:    pd.DataFrame,
    subsample_seconds: float = DEFAULT_SUBSAMPLE_SECONDS,
    tolerance_seconds: float = 1.0,
) -> pd.DataFrame | None:
    """
    Merge tracking and event data for a single match.

    SkillCorner timestamps are match-cumulative (period 2 starts where period 1
    ended), while StatsBomb timestamps reset to 0 at the start of each period.
    We therefore merge per period using period-relative timestamps.

    Returns a DataFrame or None if no data is available.
    """
    if tracking_df.empty:
        print(f"  Skipping match {skc_match_id}: empty tracking data")
        return None

    if events_df.empty:
        print(f"  Skipping match {skc_match_id}: no events found")
        return None

    # Subsample tracking frames
    if subsample_seconds > 0:
        tracking_df = subsample_tracking(tracking_df, subsample_seconds)

    # Filter events
    match_events = filter_priority_events(events_df)

    # Prepare event columns for merge
    events_slim = select_event_columns(match_events)

    # Merge period-by-period to handle timestamp systems that differ between
    # providers (SkillCorner is match-cumulative; StatsBomb resets per period).
    period_results = []
    for period in sorted(tracking_df["period"].unique()):
        trk = tracking_df[tracking_df["period"] == period].copy()
        evt = events_slim[events_slim["period"] == period].copy()

        if trk.empty:
            continue
        if evt.empty:
            # No events for this period; still keep tracking rows (event cols = NaN)
            trk["skc_match_id"] = skc_match_id
            trk["sb_match_id"]  = sb_match_id
            trk["home_team"]    = home_team
            trk["away_team"]    = away_team
            period_results.append(trk)
            continue

        # Period-relative timestamps:
        #   SkillCorner: subtract period start so period 2 begins at 0
        #   StatsBomb: already period-relative
        trk_start = trk["timestamp_seconds"].min()
        trk = trk.copy()
        trk["ts_rel"] = trk["timestamp_seconds"] - trk_start

        evt = evt.copy()
        evt["ts_rel"] = evt["timestamp_seconds"]  # already period-relative

        trk_sorted = trk.sort_values("ts_rel")
        evt_sorted = evt.sort_values("ts_rel")

        period_merged = pd.merge_asof(
            trk_sorted,
            evt_sorted,
            on="ts_rel",
            direction="nearest",
            tolerance=tolerance_seconds,
        )

        # Drop the temporary merge key; keep original timestamp_seconds
        period_merged.drop(columns=["ts_rel"], inplace=True)

        period_results.append(period_merged)

    if not period_results:
        return None

    merged = pd.concat(period_results, ignore_index=True)

    # Clean up duplicate columns produced by merge_asof when both sides had
    # 'period' and 'timestamp_seconds'. Keep the tracking (left) versions.
    if "period_x" in merged.columns:
        merged.rename(columns={"period_x": "period"}, inplace=True)
    if "period_y" in merged.columns:
        merged.drop(columns=["period_y"], inplace=True)
    if "timestamp_seconds_x" in merged.columns:
        merged.rename(columns={"timestamp_seconds_x": "timestamp_seconds"}, inplace=True)
    if "timestamp_seconds_y" in merged.columns:
        merged.drop(columns=["timestamp_seconds_y"], inplace=True)

    # Add match-level metadata
    merged["skc_match_id"] = skc_match_id
    merged["sb_match_id"]  = sb_match_id
    merged["home_team"]    = home_team
    merged["away_team"]    = away_team

    print(f"  Merged: {len(merged):,} rows, {merged['frame'].nunique():,} frames, "
          f"{merged['player_id'].nunique()} unique players")
    return merged


def get_already_processed() -> set:
    """Return set of skc_match_ids already saved in the output directory."""
    pattern = str(OUTPUT_DIR / "match_*.parquet")
    files   = glob.glob(pattern)
    ids = set()
    for f in files:
        name = os.path.basename(f)
        try:
            ids.add(int(name.replace("match_", "").replace(".parquet", "")))
        except ValueError:
            pass
    return ids


def save_match(merged_df: pd.DataFrame, skc_match_id: int) -> None:
    """Save a single merged match file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"match_{skc_match_id}.parquet"
    merged_df.to_parquet(out_path, index=False, compression="snappy", engine="pyarrow")
    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"  Saved -> {out_path.name}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    force_reprocess:    bool  = False,
    subsample_seconds:  float = DEFAULT_SUBSAMPLE_SECONDS,
    tolerance_seconds:  float = 1.0,
    match_ids:          list  = None,   # limit to specific skc_match_ids for testing
):
    print("=" * 70)
    print("J1 League 2024 — Tracking × Events Merge Pipeline")
    print("=" * 70)

    # ── Load shared data ─────────────────────────────────────────────────────
    mapping_df = load_match_mapping()
    print(f"Mapping loaded: {len(mapping_df)} matches")

    all_events_df = load_events(mapping_df)

    # ── Determine which matches to process ───────────────────────────────────
    already_done = set() if force_reprocess else get_already_processed()
    if already_done:
        print(f"Skipping {len(already_done)} already-processed matches "
              f"(use --force to reprocess)")

    to_process = mapping_df.copy()
    if not force_reprocess:
        to_process = to_process[~to_process["skc_match_id"].isin(already_done)]
    if match_ids:
        to_process = to_process[to_process["skc_match_id"].isin(match_ids)]

    print(f"\nMatches to process: {len(to_process)}")
    print("=" * 70)

    # ── Per-match processing ─────────────────────────────────────────────────
    success, failed = 0, []
    total = len(to_process)

    for i, row in enumerate(to_process.itertuples(), start=1):
        skc_id     = row.skc_match_id
        sb_id      = row.sb_match_id
        home_team  = row.home_team
        away_team  = row.away_team
        match_date = row.date

        print(f"\n[{i}/{total}] Match {skc_id}  ({home_team} vs {away_team}, {match_date})")

        t0 = time.time()
        try:
            # 1. Load and explode tracking data
            frames = load_tracking_frames(skc_id)
            if not frames:
                print("  No tracking frames — skipping")
                failed.append((skc_id, "no tracking frames"))
                continue

            player_lookup = build_player_lookup(skc_id)
            tracking_df   = explode_tracking(frames, player_lookup)
            print(f"  Tracking: {len(frames):,} frames -> {len(tracking_df):,} player-frame rows")

            # 2. Filter events for this match
            match_events = all_events_df[all_events_df["match_id"] == sb_id].copy()
            print(f"  Events for this match: {len(match_events):,}")

            # 3. Merge
            merged = merge_match(
                skc_id, sb_id, home_team, away_team,
                tracking_df, match_events,
                subsample_seconds=subsample_seconds,
                tolerance_seconds=tolerance_seconds,
            )

            if merged is None or merged.empty:
                failed.append((skc_id, "empty merge result"))
                continue

            # 4. Save
            save_match(merged, skc_id)
            success += 1

        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            failed.append((skc_id, str(exc)))

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Done.  Success: {success}/{total}   Failed: {len(failed)}")
    if failed:
        print("Failed matches:")
        for mid, reason in failed:
            print(f"  {mid}: {reason}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge J1 2024 SkillCorner tracking with StatsBomb events."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess matches that already have output files."
    )
    parser.add_argument(
        "--subsample", type=float, default=DEFAULT_SUBSAMPLE_SECONDS,
        help="Keep one tracking frame every N seconds (0 = keep all). Default: 0.2"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.0,
        help="Max seconds between a tracking frame and its matched event. Default: 1.0"
    )
    parser.add_argument(
        "--match-ids", nargs="+", type=int, default=None,
        help="Process only specific SkillCorner match IDs (for testing)."
    )
    args = parser.parse_args()

    main(
        force_reprocess   = args.force,
        subsample_seconds = args.subsample,
        tolerance_seconds = args.tolerance,
        match_ids         = args.match_ids,
    )
