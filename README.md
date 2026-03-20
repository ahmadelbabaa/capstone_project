# Goal-Kick Pressing Analysis — J1 League 2024

Capstone thesis project applying machine learning to football set pieces. The research question is:

> **How do opponent out-of-possession pressing structures during goal kicks predict build-up success, as measured by StatsBomb On-Ball Value (OBV)?**

---

## Research Overview

Goal kicks represent a critical transition moment where the defending team's pressing shape is briefly observable and quantifiable. This project takes the opponent team's structure at the moment of each goal kick and models their effect on the kicking team's subsequent build-up quality.

---

## Data Sources

| Source | Dataset | Scope |
|--------|---------|-------|
| **SkillCorner** | Tracking data (10 fps) | J1 League 2024 — 376 matches |
| **StatsBomb** | Event data + On-Ball Value (OBV) | J1 League 2024 — 376 matches |

Both datasets cover the same 376 matches and have been fully merged.

---

## Project Structure

```
capstone_project/
├── data/
│   ├── merged_j1_2024/               # 376 per-match parquet files (tracking + events merged)
│   ├── feature_engineering/          # Processed feature parquet files
│   │   ├── goal_kick_sequences.parquet       # All goal kick frames (54 MB)
│   │   ├── obv_features.parquet              # OBV aggregations per sequence
│   │   ├── progression_features.parquet      # Pass/carry/final-third features
│   │   ├── defensive_pressure_features.parquet # Tracking-based defensive features
│   │   ├── goal_kick_features.parquet        # All features merged (one row per sequence)
│   │   └── model_features_80_20.parquet      # 80/20 temporal split features for modelling
│   ├── tracking_j1_2024/             # Raw SkillCorner tracking JSONL files
│   ├── metadata_j1_2024/             # SkillCorner match metadata JSON files
│   ├── sb_data/                      # StatsBomb event data JSON
│   └── mapping_ids/
│       └── skc_sb_match_mapping.csv  # SkillCorner ↔ StatsBomb match ID mapping
│
├── data_prep/
│   ├── create_match_mapping.py       # Creates skc_sb_match_mapping.csv
│   ├── statsbomb.py                  # StatsBomb data loading helpers
│   ├── extract_lineups.py            # Extracts lineup data from StatsBomb
│   ├── paths.py                      # Centralised path constants
│   ├── tracking_data_extract.ipynb   # SkillCorner tracking extraction
│   ├── prepare_sync_data.ipynb       # Sync preparation
│   ├── data_mapping.ipynb            # Match mapping exploration
│   ├── analyze_j1_merged.ipynb       # EDA on merged data
│   └── data_merge/
│       └── merge_j1_2024.py          # Main merge pipeline (tracking + events)
│
├── feature_engineering/
│   ├── 01_filter_goal_kicks.py       # Filters play_pattern = "From Goal Kick"; assigns goal_kick_id
│   ├── 02_obv_features.py            # Sequence-level OBV aggregations
│   ├── 03_progression_features.py    # Progressive passes, carries, final third entry
│   └── 04_defensive_pressure_features.py  # Tracking-based defensive metrics
│
├── model/
│   ├── extract_80_20_features.py     # Recomputes features on first 80% of each sequence
│   ├── goal_kick_model.ipynb         # Model training (base)
│   ├── goal_kick_model_standardized.ipynb  # Model training with StandardScaler
│   ├── best_model.pkl                # Saved best model (Ridge Regression)
│   ├── best_model_standardized.pkl   # Saved best model with scaler pipeline
│   ├── feature_importance.png        # Feature importance chart (raw)
│   ├── feature_importance_standardized.png # Feature importance chart (standardised)
│   ├── actual_vs_predicted_standardized.png
│   └── recommendation_and_fit.png
│
├── testing/
│   ├── test_goal_kick_animation.ipynb      # Animation: single goal kick sequence
│   ├── test_goal_kick_animation2.ipynb     # Animation: merged feature dataset + cumulative OBV overlay
│   └── test_goal_kick_short.ipynb          # Animation: short goal kick sequence
│
├── requirements.txt
└── written_thesis/                   # Thesis write-up (in progress)
```

---

## What Has Been Done

### 1. Data Acquisition
- Downloaded 376 matches of J1 League 2024 tracking data from SkillCorner (extrapolated JSONL format, 10 fps)
- Downloaded corresponding StatsBomb event data including OBV columns (`obv_for_net`, `obv_against_net`, `obv_total_net`)

### 2. Match ID Mapping (`data_prep/create_match_mapping.py`)
- Matched SkillCorner matches to StatsBomb matches by joining on **date + home team + away team**
- Applied team name normalisation for divergent naming conventions (e.g. `"Tokyo"` → `"FC Tokyo"`)
- Produced `data/mapping_ids/skc_sb_match_mapping.csv` — 376 rows covering all J1 2024 matches

### 3. Data Merge (`data_prep/data_merge/merge_j1_2024.py`)
- For each of the 376 matched games:
  1. Loaded SkillCorner tracking frames (player x/y positions, ball position, roles, team IDs)
  2. Loaded StatsBomb events (event type, play pattern, OBV, possession team)
  3. Temporally aligned events to the nearest tracking frame
  4. Joined on `(skc_match_id, sb_match_id)` to produce a unified row per tracking frame
- Output: `data/merged_j1_2024/match_{skc_id}.parquet` — one file per match

**Merged parquet schema (key columns):**

| Column | Description |
|--------|-------------|
| `frame`, `timestamp_seconds` | Tracking frame identifier and time |
| `player_id`, `player_x`, `player_y` | Player tracking position (metres from pitch centre) |
| `player_role` | Goalkeeper / Defender / Midfielder / Striker |
| `skc_team_id`, `is_home` | SkillCorner team identifier and home/away flag |
| `ball_x`, `ball_y`, `ball_z` | Ball position |
| `event_id`, `event_type` | StatsBomb event identifier and type |
| `play_pattern` | e.g. `"From Goal Kick"`, `"Regular Play"` |
| `possession_team_id`, `possession_team_name` | Team in possession |
| `event_location_x/y` | StatsBomb event location (0–120, 0–80) |
| `pass_end_x/y`, `carry_end_x/y` | Pass / carry destination |
| `obv_for_net`, `obv_against_net`, `obv_total_net` | On-Ball Value per event |
| `home_team`, `away_team` | Team names |

### 4. Goal Kick Filtering (`feature_engineering/01_filter_goal_kicks.py`)
- Filtered all 376 merged parquet files for rows where `play_pattern == "From Goal Kick"`
- Assigned a unique `goal_kick_id` per possession sequence by detecting changes in `possession_team_id` across match/period boundaries
- Output: `data/feature_engineering/goal_kick_sequences.parquet`
  - **5,593 unique goal kick sequences** across 376 matches
  - 54 MB, one row per tracking frame × player

### 5. Feature Engineering

All feature scripts produce one row per `goal_kick_id` and are stored in `data/feature_engineering/`.

#### OBV Features (`02_obv_features.py`)
Deduplicates by `event_id` then sums OBV across all events in the sequence:

| Feature | Description |
|---------|-------------|
| `obv_total_seq` | Cumulative net OBV over the entire sequence |
| `obv_for_seq` | Cumulative OBV for the possession team |
| `obv_against_seq` | Cumulative OBV for the defending team |

#### Progression Features (`03_progression_features.py`)
Event-level features computed from StatsBomb coordinates (0–120 x-axis):

| Feature | Description |
|---------|-------------|
| `n_progressive_passes` | Passes where `pass_end_x > event_location_x` |
| `n_progressive_carries` | Carries where `carry_end_x > event_location_x` |
| `final_third_entry` | Boolean — any action ended with x > 80 (final third) |

#### Defensive Pressure Features (`04_defensive_pressure_features.py`)
Frame-level tracking features computed on the defending team (identified via `is_home` flag, since `possession_team_id` and `skc_team_id` use different ID systems):

| Feature | Description |
|---------|-------------|
| `press_compactness_mean` | Mean pairwise distance between all defending players, averaged over frames |
| `press_height_mean_x` | Mean x of defending team across all frames |
| `press_height_max_x` | Max frame-wise mean x (how high the line ever pushes) |
| `def_line_depth_mean_x` | Mean x of 4 deepest defenders, averaged over frames |
| `def_line_depth_min_x` | Minimum of the above over frames (how deep they ever drop) |
| `press_forward_speed_mean` | Mean Δx/Δt across all defending players (positive = moving up pitch) |
| `press_forward_speed_max` | Max frame-wise average forward speed |
| `avg_def_to_ball_dist` | Mean distance from ball to nearest 3 defenders, averaged over frames |
| `min_def_to_ball_dist` | Minimum such distance across all frames |
| `n_frames_ball_under_pressure` | Frames where at least one defender is within 5 m of the ball |

#### Merged Feature Table
All feature files joined on `goal_kick_id` into `goal_kick_features.parquet`:
- **5,593 rows × 25 columns** — one row per goal kick sequence
- Includes core identifiers: `goal_kick_id`, `skc_match_id`, `sb_match_id`, `possession_team_id`, `possession_team_name`, `period`, `sequence_start_time`, `event_location_x/y`

### 6. Visualisation & Testing (`testing/`)
Three Jupyter notebooks were produced to validate the data pipeline and explore sequences visually:

| Notebook | Description |
|----------|-------------|
| `test_goal_kick_animation.ipynb` | Pitch animation of a single goal kick sequence with player/ball trails |
| `test_goal_kick_animation2.ipynb` | Two-panel animation: pitch + real-time cumulative OBV chart, using the merged feature table to select a high-OBV sequence with final-third entry |
| `test_goal_kick_short.ipynb` | Same two-panel animation for a short goal kick sequence (goalkeeper plays to a nearby defender) |

All animations use SkillCorner → StatsBomb coordinate conversion, colour players by team using the `is_home` flag, and run at 10 fps with ball and player trails.

### 7. Predictive Modelling (`model/`)

#### 80/20 Temporal Split (`extract_80_20_features.py`)
Rather than a standard train/test split, the model uses a **temporal split within each sequence**:
- **Input features**: computed from the first 80% of frames in each sequence (game state so far)
- **Target**: `obv_remaining_20` — OBV accumulated in the final 20% of the sequence
- **Action feature**: event type, location (x, y), and direction at the 80% mark

Features recomputed on the 80% window:

| Feature | Description |
|---------|-------------|
| `obv_for_80`, `obv_against_80` | Cumulative OBV in first 80% |
| `n_prog_passes_80`, `n_prog_carries_80` | Progressive actions in first 80% |
| `final_third_80` | Whether final third was entered in first 80% |
| `press_compactness_80` | Defensive compactness in first 80% |
| `press_height_mean/max_x_80` | Defensive line height in first 80% |
| `def_line_depth_mean_x_80` | Defensive depth in first 80% |
| `avg/min_def_ball_dist_80` | Defender-to-ball distances in first 80% |
| `n_frames_pressure_80` | Frames under pressure in first 80% |
| `press_fwd_speed_80` | Defensive forward speed in first 80% |
| `action_type_enc` | Event type at the 80% mark (label encoded) |
| `action_x`, `action_y` | Pitch location at the 80% mark |
| `action_progressive` | Whether the action at 80% is forward-moving |
| `period` | Match period |

Sequences shorter than 10 frames were excluded → **5,113 sequences** used for modelling.

#### Model Training (`goal_kick_model_standardized.ipynb`)
Three models compared with `StandardScaler` applied to all features:

| Model | RMSE | MAE | R² (test) | R² (CV) |
|-------|------|-----|-----------|---------|
| **Ridge Regression** ✓ | 0.0651 | 0.0172 | **0.177** | 0.035 |
| Gradient Boosting | 0.0684 | 0.0178 | 0.091 | 0.017 |
| Random Forest | 0.0690 | 0.0166 | 0.076 | 0.019 |

Ridge Regression selected as best model. Saved to `model/best_model_standardized.pkl` (includes `StandardScaler` pipeline and `LabelEncoder`).

**Key finding from standardised feature importance:** OBV momentum in the first 80% of the sequence (`obv_for_80`, `obv_against_80`) is the dominant predictor of remaining OBV — sequences that are going well tend to continue going well. Defensive pressure features and action type contribute modestly on top of this momentum signal.

---

## Key Dependencies

```
pandas==3.0.1
numpy==2.4.3
matplotlib==3.10.8
requests==2.32.5
pyarrow
mplsoccer
skillcorner
scikit-learn
```