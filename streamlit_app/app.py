"""
J1 League 2024 — Goal Kick Pressing Analysis
Streamlit app entry point.

Run from project root:
    streamlit run streamlit_app/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="J1 2024 Goal Kick Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("⚽ J1 League 2024")
st.sidebar.caption(
    "Gradient Boosting model predicting goal kick build-up quality (Sequence Progression Value) "
    "from defensive pressing structures — 376 matches, 5,113 sequences."
)
st.sidebar.markdown("---")

st.title("Goal Kick Pressing Analysis")
st.markdown(
    """
    This app demonstrates a machine learning model trained on J1 League 2024 data
    to predict **build-up quality** (measured by a custom **Sequence Progression Value** metric
    inspired by VAEP) in the **final 20%** of goal kick sequences, using tracking and event data from the first 80%.

    **Navigate using the sidebar:**
    - **Overview** — Dataset statistics and model performance
    - **Sequence Explorer** — Browse goal kick sequences with pitch visualisations
    - **Model Predictions** — Interactive model demo with feature contributions
    """
)
