# Goal-Kick Pressing Analysis — J1 League 2024

Capstone thesis project applying machine learning to football set pieces. The research question is:

> **How do opponent out-of-possession pressing structures during goal kicks predict build-up success, as measured by StatsBomb On-Ball Value (OBV)?**

---

## Research Overview

Goal kicks represent a critical transition moment where the defending team's pressing shape is briefly observable and quantifiable. This project takes opponent team's structure at the moment of each goal kick and models their effect on the kicking team's subsequent build-up quality.


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
│   ├── merged_j1_2024/          # 376 per-match parquet files (primary output)
│   ├── tracking_j1_2024/        # Raw SkillCorner tracking JSONL files
│   ├── metadata_j1_2024/        # SkillCorner match metadata JSON files
│   ├── sb_data/                 # StatsBomb event data JSON
│   ├── mapping_ids/
│   │   └── skc_sb_match_mapping.csv  # SkillCorner ↔ StatsBomb match ID mapping
│   └── matches_mapping.csv      # StatsBomb ↔ Wyscout match ID mapping
│
├── data_prep/
│   ├── create_match_mapping.py  # Creates skc_sb_match_mapping.csv
│   ├── statsbomb.py             # StatsBomb data loading helpers
│   ├── extract_lineups.py       # Extracts lineup data from StatsBomb
│   ├── paths.py                 # Centralised path constants
│   ├── tracking_data_extract.ipynb   # SkillCorner tracking extraction
│   ├── prepare_sync_data.ipynb       # Sync preparation
│   ├── data_mapping.ipynb            # Match mapping exploration
│   ├── analyze_j1_merged.ipynb       # EDA on merged data (match 1410827)
│   └── data_merge/
│       └── merge_j1_2024.py     # Main merge pipeline (tracking + events)
│
└── written_thesis/              # Thesis write-up (in progress)
```

---

## What Has Been Done

### 1. Data Acquisition
- Downloaded 376 matches of J1 League 2024 tracking data from SkillCorner using API(extrapolated JSONL format, 10 fps)
- Downloaded corresponding StatsBomb event data including OBV columns (`obv_for_net`, `obv_against_net`, `obv_total_net`)

### 2. Match ID Mapping (`create_match_mapping.py`)
- Matched SkillCorner matches to StatsBomb matches by joining on **date + home team + away team**
- Applied team name normalisation for 4 divergent naming conventions (e.g. `"Tokyo"` → `"FC Tokyo"`)
- Produced `data/mapping_ids/skc_sb_match_mapping.csv` — 376 rows covering all J1 2024 matches

### 3. Data Merge (`data_merge/merge_j1_2024.py`)
- For each of the 376 matched games:
  1. Loaded SkillCorner tracking frames (player x/y positions, velocities, roles, team IDs)
  2. Loaded StatsBomb events (event type, play pattern, OBV, possession team)
  3. Temporally aligned events to the nearest tracking frame
  4. Joined on `(skc_match_id, sb_match_id)` to produce a unified row per tracking frame
- Output: `data/merged_j1_2024/match_{skc_id}.parquet` — one file per match

**Merged parquet schema (key columns):**

| Column | Description |
|--------|-------------|
| `frame`, `timestamp_seconds` | Tracking frame identifier and time |
| `player_id`, `player_x`, `player_y` | Player tracking position (metres from centre) |
| `player_role` | Goalkeeper / Defender / Midfielder / Striker |
| `skc_team_id`, `is_home` | Team identifier and home/away flag |
| `ball_x`, `ball_y`, `ball_z` | Ball position |
| `event_id`, `event_type` | StatsBomb event identifier and type |
| `play_pattern` | e.g. `"From Goal Kick"`, `"Regular Play"` |
| `possession_team_id` | Team in possession during the event |
| `event_location_x/y` | StatsBomb event location on pitch |
| `pass_end_x/y`, `carry_end_x/y` | Pass/carry destination |
| `obv_for_net`, `obv_against_net`, `obv_total_net` | On-Ball Value per event |
| `sb_match_id`, `skc_match_id` | Cross-dataset match IDs |
| `home_team`, `away_team` | Team names |

Sample validation (match 1410827): 484,484 rows × 50 columns; 22,022 unique frames; 63.6% of rows have a matched event; median temporal alignment gap 0.36 s.

---

## Next Steps

1. **Extract goal-kick sequences** — filter `play_pattern == "From Goal Kick"` and group into possession chains
2. **Classify pressing phase** — at each goal kick, compute the avg x-position of the defending team's 3 deepest players and assign High / Mid / Low block label
3. **Compute sequence-level OBV** — sum `obv_total_net` per possession sequence for the kicking team
4. **Feature engineering** — compactness, pressing height, defensive line depth, spread over the 10-second post-kick window
5. **Regression model** — GradientBoostingRegressor with SHAP interpretability, trained per-match to avoid data leakage
6. **Streamlit dashboard** — visualise pressing phase distributions and OBV outcomes per team

---

## Key Dependencies

- `pandas`, `pyarrow` — parquet I/O and data manipulation
- `scikit-learn` — regression modelling
- `shap` — model interpretability
- `mplsoccer` — pitch visualisation
- `streamlit` — interactive dashboard (planned)
