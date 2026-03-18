"""
04_defensive_pressure_features.py
----------------------------------
Computes defensive pressure features for each goal kick sequence using
SkillCorner tracking data (player_x, player_y in metres, pitch-centre origin).

Team identification
-------------------
possession_team_id is a StatsBomb ID while skc_team_id is a SkillCorner ID —
they cannot be compared directly. Instead we use:
    possession_team_name == home_team  →  possession team is the home side
    is_home flag on tracking rows      →  reliable team split for each frame

Features produced
-----------------
Compactness
  press_compactness_mean    : Mean pairwise distance between all defending
                              players per frame, averaged over all frames.

Line height & depth  (SkillCorner x, metres from pitch centre)
  press_height_mean_x       : Mean x of all defending players across all frames.
  press_height_max_x        : Max frame-wise mean x (highest the line ever pushes).
  def_line_depth_mean_x     : Per frame, mean x of the 4 deepest defenders
                              (lowest x = furthest from ball), averaged over frames.
  def_line_depth_min_x      : Min of the above over frames (deepest they ever drop).

Intensity / movement
  press_forward_speed_mean  : Mean Δx/Δt across all defending players and frames.
                              Positive = moving in the +x direction (up the pitch).
  press_forward_speed_max   : Max frame-wise average forward speed.

Proximity to ball
  avg_def_to_ball_dist      : Per frame, mean distance from ball to nearest 3
                              defenders, averaged over all frames.
  min_def_to_ball_dist      : Minimum such distance across all frames.
  n_frames_ball_under_pressure : Frames where at least one defender is within
                              5 metres of the ball.

Coordinates: SkillCorner metres, pitch-centre origin. x: -52.5 → +52.5.
N_NEAREST = 3  (nearest defenders for ball distance)
PRESSURE_RADIUS = 5  metres

Input  : data/feature_engineering/goal_kick_sequences.parquet
Output : data/feature_engineering/defensive_pressure_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# ── Config ─────────────────────────────────────────────────────────────────────
N_NEAREST       = 3    # defenders used for avg_def_to_ball_dist
PRESSURE_RADIUS = 5.0  # metres for n_frames_ball_under_pressure
N_DEEP          = 4    # defenders used for def_line_depth

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE   = PROJECT_ROOT / "data" / "feature_engineering" / "goal_kick_sequences.parquet"
OUTPUT_FILE  = PROJECT_ROOT / "data" / "feature_engineering" / "defensive_pressure_features.parquet"

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading goal kick sequences...")
df = pd.read_parquet(INPUT_FILE)
print(f"  Rows      : {len(df):,}")
print(f"  Sequences : {df['goal_kick_id'].nunique():,}")

# ── Helper functions ───────────────────────────────────────────────────────────

def mean_pairwise_dist(positions):
    """Mean pairwise Euclidean distance between all pairs in an (n, 2) array."""
    n = len(positions)
    if n < 2:
        return np.nan
    # Vectorised pairwise distances using broadcasting
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n, n, 2)
    dists = np.sqrt((diff ** 2).sum(axis=2))                           # (n, n)
    # Upper triangle only (exclude self-distances and duplicates)
    idx = np.triu_indices(n, k=1)
    return dists[idx].mean()


def ball_to_def_distances(ball, positions):
    """Euclidean distances from ball to each defender. Returns sorted array."""
    diffs = positions - ball          # (n, 2)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    return np.sort(dists)


# ── Main loop ──────────────────────────────────────────────────────────────────
print("Computing defensive pressure features...")

records = []

for gk_id, gk_df in df.groupby("goal_kick_id"):

    # Identify defending team via is_home flag
    poss_is_home = gk_df["possession_team_name"].iloc[0] == gk_df["home_team"].iloc[0]
    def_is_home  = not poss_is_home

    def_df = gk_df[gk_df["is_home"] == def_is_home].copy()

    if def_df.empty:
        records.append({"goal_kick_id": gk_id})
        continue

    # ── Frame-level accumulators ───────────────────────────────────────────────
    compactness_per_frame     = []
    mean_x_per_frame          = []
    depth_mean_x_per_frame    = []   # mean x of N_DEEP deepest defenders
    ball_near_dist_per_frame  = []   # mean dist to nearest N_NEAREST defenders
    min_dist_per_frame        = []   # closest single defender to ball
    under_pressure_per_frame  = []   # bool: any defender within PRESSURE_RADIUS

    # For forward speed we need consecutive frames per player
    # Store (player_id, frame, x, t) tuples
    player_frame_records = []

    frames_sorted = sorted(def_df["frame"].unique())

    for frame_id in frames_sorted:
        fdf = def_df[def_df["frame"] == frame_id]

        pos   = fdf[["player_x", "player_y"]].values.astype(float)
        xs    = pos[:, 0]
        ball  = fdf[["ball_x", "ball_y"]].iloc[0].values.astype(float)
        t     = fdf["timestamp_seconds"].iloc[0]

        # -- Compactness --
        compactness_per_frame.append(mean_pairwise_dist(pos))

        # -- Line height --
        mean_x_per_frame.append(xs.mean())

        # -- Line depth: N_DEEP defenders with smallest x --
        if len(xs) >= N_DEEP:
            deep_xs = np.sort(xs)[:N_DEEP]
            depth_mean_x_per_frame.append(deep_xs.mean())

        # -- Ball proximity --
        dists = ball_to_def_distances(ball, pos)
        n     = min(N_NEAREST, len(dists))
        ball_near_dist_per_frame.append(dists[:n].mean())
        min_dist_per_frame.append(dists[0])

        # -- Under pressure --
        under_pressure_per_frame.append(bool(dists[0] <= PRESSURE_RADIUS))

        # -- Forward speed records --
        for _, prow in fdf.iterrows():
            player_frame_records.append({
                "player_id": prow["player_id"],
                "frame":     frame_id,
                "x":         prow["player_x"],
                "t":         t,
            })

    # ── Aggregate frame-level stats ────────────────────────────────────────────
    compactness_arr    = np.array(compactness_per_frame,    dtype=float)
    mean_x_arr         = np.array(mean_x_per_frame,         dtype=float)
    depth_mean_x_arr   = np.array(depth_mean_x_per_frame,   dtype=float)
    ball_near_arr      = np.array(ball_near_dist_per_frame,  dtype=float)
    min_dist_arr       = np.array(min_dist_per_frame,        dtype=float)
    under_pressure_arr = np.array(under_pressure_per_frame,  dtype=bool)

    press_compactness_mean  = np.nanmean(compactness_arr)
    press_height_mean_x     = np.nanmean(mean_x_arr)
    press_height_max_x      = np.nanmax(mean_x_arr)
    def_line_depth_mean_x   = np.nanmean(depth_mean_x_arr) if len(depth_mean_x_arr) > 0 else np.nan
    def_line_depth_min_x    = np.nanmin(depth_mean_x_arr)  if len(depth_mean_x_arr) > 0 else np.nan
    avg_def_to_ball_dist    = np.nanmean(ball_near_arr)
    min_def_to_ball_dist    = float(np.nanmin(min_dist_arr))
    n_frames_under_pressure = int(under_pressure_arr.sum())

    # ── Forward speed: Δx/Δt per player across consecutive frames ─────────────
    speed_records = pd.DataFrame(player_frame_records)
    speed_records = speed_records.sort_values(["player_id", "frame"])

    # Diff within each player
    speed_records["dx"] = speed_records.groupby("player_id")["x"].diff()
    speed_records["dt"] = speed_records.groupby("player_id")["t"].diff()
    speed_records = speed_records.dropna(subset=["dx", "dt"])
    speed_records = speed_records[speed_records["dt"] > 0]   # skip duplicate timestamps

    if not speed_records.empty:
        speed_records["forward_speed"] = speed_records["dx"] / speed_records["dt"]
        # Frame-wise mean speed (average across players per frame transition)
        frame_mean_speed = speed_records.groupby("frame")["forward_speed"].mean()
        press_forward_speed_mean = float(frame_mean_speed.mean())
        press_forward_speed_max  = float(frame_mean_speed.max())
    else:
        press_forward_speed_mean = np.nan
        press_forward_speed_max  = np.nan

    records.append({
        "goal_kick_id"               : gk_id,
        "press_compactness_mean"     : press_compactness_mean,
        "press_height_mean_x"        : press_height_mean_x,
        "press_height_max_x"         : press_height_max_x,
        "def_line_depth_mean_x"      : def_line_depth_mean_x,
        "def_line_depth_min_x"       : def_line_depth_min_x,
        "press_forward_speed_mean"   : press_forward_speed_mean,
        "press_forward_speed_max"    : press_forward_speed_max,
        "avg_def_to_ball_dist"       : avg_def_to_ball_dist,
        "min_def_to_ball_dist"       : min_def_to_ball_dist,
        "n_frames_ball_under_pressure": n_frames_under_pressure,
    })

# ── Build output DataFrame ─────────────────────────────────────────────────────
features = pd.DataFrame(records)

# ── Sanity checks ──────────────────────────────────────────────────────────────
assert len(features) == df["goal_kick_id"].nunique(), \
    "Row count mismatch — every sequence must have exactly one feature row."

null_counts = features.isna().sum()
if null_counts.sum() > 0:
    print("\nWARNING: nulls detected:")
    print(null_counts[null_counts > 0])

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("-- Defensive Pressure Feature Summary ---------------------------")
for col in features.columns[1:]:
    s = features[col]
    print(f"  {col:<35}: mean={s.mean():.3f}  std={s.std():.3f}")
print("------------------------------------------------------------------")
print()

# ── Save ───────────────────────────────────────────────────────────────────────
features.to_parquet(OUTPUT_FILE, index=False)
print(f"Saved -> {OUTPUT_FILE}")
print(f"Shape  : {features.shape}")
