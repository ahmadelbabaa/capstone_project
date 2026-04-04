"""
extract_80_20_features.py
--------------------------
For each goal kick sequence, splits frames into first 80% (features) and
last 20% (target). Computes all features on the 80% window and saves to parquet.

Optimisations vs notebook version
- Ball distances fully vectorised (no per-frame Python loop)
- Forward speed uses pandas groupby diff (no list-of-dicts construction)
- Frame loop only used for compactness (pairwise distances), with numpy broadcasting
- Defending team flag pre-computed once per sequence, not inside the frame loop
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MIN_FRAMES      = 10
N_DEEP          = 4
N_NEAREST       = 3
PRESSURE_RADIUS = 5.0
FINAL_THIRD_X   = 80.0   # StatsBomb x

PROJECT_ROOT       = Path(__file__).resolve().parent.parent
INPUT_FILE         = PROJECT_ROOT / "data" / "feature_engineering" / "goal_kick_sequences.parquet"
PROG_FEATURES_FILE = PROJECT_ROOT / "data" / "feature_engineering" / "sequence_progression_features.parquet"
OUTPUT_FILE        = PROJECT_ROOT / "data" / "feature_engineering" / "model_features_80_20.parquet"

print("Loading sequences...")
seqs = pd.read_parquet(INPUT_FILE)
print(f"  Rows      : {len(seqs):,}")
print(f"  Sequences : {seqs['goal_kick_id'].nunique():,}")

# ── Pre-compute defending team flag ───────────────────────────────────────────
poss_is_home_map = (
    seqs.groupby("goal_kick_id")
    .apply(lambda g: g["possession_team_name"].iloc[0] == g["home_team"].iloc[0])
    .rename("poss_is_home")
    .reset_index()
)
seqs = seqs.merge(poss_is_home_map, on="goal_kick_id")
seqs["is_def"] = seqs["is_home"] != seqs["poss_is_home"]

# ── Assign 80 / 20 window flags ───────────────────────────────────────────────
def assign_window(g):
    frames  = sorted(g["frame"].unique())
    n       = len(frames)
    cut     = int(np.floor(n * 0.8))
    frames_80 = set(frames[:cut])
    return g["frame"].isin(frames_80).astype(int)   # 1 = in 80%, 0 = in 20%

print("Assigning 80/20 window flags...")
seqs["in_80"] = seqs.groupby("goal_kick_id", group_keys=False).apply(assign_window)

# Drop sequences that are too short
n_frames = seqs.groupby("goal_kick_id")["frame"].nunique()
valid_ids = n_frames[n_frames >= MIN_FRAMES].index
seqs = seqs[seqs["goal_kick_id"].isin(valid_ids)].copy()
print(f"  Sequences after MIN_FRAMES filter : {seqs['goal_kick_id'].nunique():,}")

df_80 = seqs[seqs["in_80"] == 1].copy()
df_20 = seqs[seqs["in_80"] == 0].copy()
def_80 = df_80[df_80["is_def"]].copy()

# ── TARGET: OBV in last 20% ───────────────────────────────────────────────────
target = (
    df_20.drop_duplicates(["goal_kick_id", "event_id"])
    .groupby("goal_kick_id")["obv_total_net"]
    .apply(lambda x: x.fillna(0).sum())
    .rename("obv_remaining")
)

# ── OBV features from first 80% ───────────────────────────────────────────────
events_80 = df_80.drop_duplicates(["goal_kick_id", "event_id"])
obv_feats = events_80.groupby("goal_kick_id").agg(
    obv_for_80     = ("obv_for_net",     lambda x: x.fillna(0).sum()),
    obv_against_80 = ("obv_against_net", lambda x: x.fillna(0).sum()),
)

# ── Progression features ───────────────────────────────────────────────────────
passes_80  = events_80[events_80["event_type"] == "Pass"]
carries_80 = events_80[events_80["event_type"] == "Carry"]

n_prog_passes = (
    passes_80[passes_80["pass_end_x"] > passes_80["event_location_x"]]
    .groupby("goal_kick_id").size().rename("n_prog_passes_80")
)
n_prog_carries = (
    carries_80[carries_80["carry_end_x"] > carries_80["event_location_x"]]
    .groupby("goal_kick_id").size().rename("n_prog_carries_80")
)

ft_pass  = passes_80[passes_80["pass_end_x"]   > FINAL_THIRD_X].groupby("goal_kick_id").size() > 0
ft_carry = carries_80[carries_80["carry_end_x"] > FINAL_THIRD_X].groupby("goal_kick_id").size() > 0
final_third = (ft_pass | ft_carry).astype(int).rename("final_third_80")

# ── Ball distance features (vectorised) ───────────────────────────────────────
def_80["ball_dist"] = np.sqrt(
    (def_80["player_x"] - def_80["ball_x"])**2 +
    (def_80["player_y"] - def_80["ball_y"])**2
)

# Per frame: nearest N defenders to ball
nearest = (
    def_80.sort_values("ball_dist")
    .groupby(["goal_kick_id", "frame"])
    .head(N_NEAREST)
    .groupby(["goal_kick_id", "frame"])["ball_dist"]
    .mean()
)
avg_def_ball_dist   = nearest.groupby("goal_kick_id").mean().rename("avg_def_ball_dist_80")
min_def_ball_dist   = def_80.groupby("goal_kick_id")["ball_dist"].min().rename("min_def_ball_dist_80")
n_frames_pressure   = (
    def_80.groupby(["goal_kick_id", "frame"])["ball_dist"]
    .min()
    .lt(PRESSURE_RADIUS)
    .groupby("goal_kick_id").sum()
    .rename("n_frames_pressure_80")
)

# ── Line height features (vectorised) ─────────────────────────────────────────
frame_mean_x = def_80.groupby(["goal_kick_id", "frame"])["player_x"].mean()
press_height_mean_x = frame_mean_x.groupby("goal_kick_id").mean().rename("press_height_mean_x_80")
press_height_max_x  = frame_mean_x.groupby("goal_kick_id").max().rename("press_height_max_x_80")

# Deepest N defenders per frame: smallest x values
def_line_depth = (
    def_80.sort_values("player_x")
    .groupby(["goal_kick_id", "frame"])
    .head(N_DEEP)
    .groupby(["goal_kick_id", "frame"])["player_x"]
    .mean()
    .groupby("goal_kick_id").mean()
    .rename("def_line_depth_mean_x_80")
)

# ── Forward speed (pandas diff, no list building) ─────────────────────────────
# Select only needed columns to avoid memory error on large DataFrame copy
def_80_s = (
    def_80[["goal_kick_id", "player_id", "frame", "player_x", "timestamp_seconds"]]
    .sort_values(["goal_kick_id", "player_id", "frame"])
    .copy()
)
def_80_s["dx"] = def_80_s.groupby(["goal_kick_id", "player_id"])["player_x"].diff()
def_80_s["dt"] = def_80_s.groupby(["goal_kick_id", "player_id"])["timestamp_seconds"].diff()
valid_speed = def_80_s.dropna(subset=["dx", "dt"])
valid_speed = valid_speed[valid_speed["dt"] > 0].copy()
valid_speed["fwd_speed"] = valid_speed["dx"] / valid_speed["dt"]

press_fwd_speed = (
    valid_speed.groupby(["goal_kick_id", "frame"])["fwd_speed"]
    .mean()
    .groupby("goal_kick_id").mean()
    .rename("press_fwd_speed_80")
)

# ── Compactness (pairwise distances — requires frame loop) ────────────────────
print("Computing compactness (pairwise distances)...")
compactness_records = []

for (gk_id, fr), fdf in def_80.groupby(["goal_kick_id", "frame"]):
    pos = fdf[["player_x", "player_y"]].values.astype(float)
    n   = len(pos)
    if n < 2:
        continue
    diff  = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt((diff**2).sum(axis=2))
    idx   = np.triu_indices(n, k=1)
    compactness_records.append({"goal_kick_id": gk_id, "compactness": dists[idx].mean()})

compactness_df = pd.DataFrame(compactness_records)
press_compactness = (
    compactness_df.groupby("goal_kick_id")["compactness"]
    .mean()
    .rename("press_compactness_80")
)

# ── Action at the 80% mark ────────────────────────────────────────────────────
last_event_80 = (
    events_80.sort_values("timestamp_seconds")
    .groupby("goal_kick_id")
    .last()
    [["event_type", "event_location_x", "event_location_y",
      "pass_end_x", "carry_end_x", "event_team_name", "possession_team_name"]]
)
last_event_80["action_type"] = last_event_80["event_type"]
last_event_80["action_x"]    = last_event_80["event_location_x"]
last_event_80["action_y"]    = last_event_80["event_location_y"]

last_event_80["action_progressive"] = np.where(
    last_event_80["action_type"] == "Pass",
    (last_event_80["pass_end_x"] > last_event_80["event_location_x"]).astype(int),
    np.where(
        last_event_80["action_type"] == "Carry",
        (last_event_80["carry_end_x"] > last_event_80["event_location_x"]).astype(int),
        0
    )
)

# 1 = action by possession team, 0 = action by defending team
last_event_80["action_team_poss"] = (
    last_event_80["event_team_name"] == last_event_80["possession_team_name"]
).astype(int)

action_feats = last_event_80[["action_type", "action_x", "action_y",
                               "action_progressive", "action_team_poss"]]

# ── Context ───────────────────────────────────────────────────────────────────
context = (
    seqs.groupby("goal_kick_id")
    .first()
    [["period", "possession_team_name", "skc_match_id"]]
)

# ── Merge all features ────────────────────────────────────────────────────────
print("Merging features...")
model_df = (
    target.to_frame()
    .join(obv_feats, how="left")
    .join(n_prog_passes, how="left")
    .join(n_prog_carries, how="left")
    .join(final_third, how="left")
    .join(press_height_mean_x, how="left")
    .join(press_height_max_x, how="left")
    .join(def_line_depth, how="left")
    .join(avg_def_ball_dist, how="left")
    .join(min_def_ball_dist, how="left")
    .join(n_frames_pressure, how="left")
    .join(press_fwd_speed, how="left")
    .join(press_compactness, how="left")
    .join(action_feats, how="left")
    .join(context, how="left")
    .reset_index()
)

# Fill progression count nulls with 0 (sequences with no passes/carries)
for col in ["n_prog_passes_80", "n_prog_carries_80", "final_third_80"]:
    model_df[col] = model_df[col].fillna(0).astype(int)

# ── Join sequence_progression_value (custom possession value metric) ──────────
prog_feats = pd.read_parquet(PROG_FEATURES_FILE)[
    ["goal_kick_id", "sequence_progression_value",
     "mean_action_prog_value", "max_action_prog_value", "n_prog_actions",
     "prog_value_80", "prog_value_20"]
]
model_df = model_df.merge(prog_feats, on="goal_kick_id", how="left")
# Sequences with no on-ball actions (Ball Receipt only) → fill with 0
for col in ["sequence_progression_value", "mean_action_prog_value",
            "max_action_prog_value", "n_prog_actions",
            "prog_value_80", "prog_value_20"]:
    model_df[col] = model_df[col].fillna(0)

print()
print("-- Feature table summary --------------------------------------------------")
print(f"  Shape  : {model_df.shape}")
print(f"  Nulls  :")
null_counts = model_df.isna().sum()
print(null_counts[null_counts > 0].to_string())
print()
print("  Target (obv_remaining):")
print(model_df["obv_remaining"].describe().round(4).to_string())
print()
print("  Action type at 80% mark:")
print(model_df["action_type"].value_counts().head(8).to_string())

model_df.to_parquet(OUTPUT_FILE, index=False)
print()
print(f"Saved -> {OUTPUT_FILE}")
