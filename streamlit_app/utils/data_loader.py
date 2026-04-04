"""Cached data loading functions for the Streamlit app."""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
MODEL_FEATURES = PROJECT_ROOT / "data" / "feature_engineering" / "model_features_80_20.parquet"
GK_FEATURES    = PROJECT_ROOT / "data" / "feature_engineering" / "goal_kick_features.parquet"
SEQUENCES_FILE = PROJECT_ROOT / "data" / "feature_engineering" / "goal_kick_sequences.parquet"


@st.cache_data
def load_model_features() -> pd.DataFrame:
    """Load the 5,113-row model training/test feature table."""
    return pd.read_parquet(MODEL_FEATURES)


@st.cache_data
def load_goal_kick_features() -> pd.DataFrame:
    """Load the full 5,593-row goal kick feature table (richer metadata)."""
    return pd.read_parquet(GK_FEATURES)


@st.cache_data
def load_sequence_metadata() -> pd.DataFrame:
    """
    Load a lightweight per-sequence metadata table from the large sequences file.
    Reads only necessary columns via PyArrow to avoid loading the full ~8 GB dataset.
    Joins against goal_kick_features for OBV and match context.
    """
    cols = ["goal_kick_id", "home_team", "away_team", "possession_team_name", "skc_match_id"]
    table = pq.read_table(SEQUENCES_FILE, columns=cols)
    meta = table.to_pandas().drop_duplicates("goal_kick_id").reset_index(drop=True)

    # Enrich with sequence-level info from goal_kick_features
    gk = load_goal_kick_features()[
        ["goal_kick_id", "obv_total_seq", "final_third_entry", "period",
         "n_progressive_passes", "n_progressive_carries",
         "obv_for_seq", "obv_against_seq"]
    ]
    meta = meta.merge(gk, on="goal_kick_id", how="left")

    # Add prog_value_20 from model features
    mf_prog = pd.read_parquet(MODEL_FEATURES, columns=["goal_kick_id", "prog_value_20"])
    meta = meta.merge(mf_prog, on="goal_kick_id", how="left")
    return meta.sort_values("goal_kick_id").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_sequence_for_id(goal_kick_id: int) -> pd.DataFrame:
    """
    Load all tracking frames for a single goal kick sequence.
    Uses PyArrow filter pushdown — never loads the full sequences file.
    """
    table = pq.read_table(
        SEQUENCES_FILE,
        filters=[("goal_kick_id", "=", goal_kick_id)],
    )
    return table.to_pandas()
