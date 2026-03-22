"""Page 2 — Goal Kick Sequence Explorer with pitch snapshots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from utils.data_loader import load_sequence_metadata, load_sequence_for_id, load_model_features
from utils.pitch_viz import draw_three_snapshots, build_animation
from utils.model_inference import load_model, predict_single

st.set_page_config(page_title="Sequence Explorer", layout="wide")
st.title("Goal Kick Sequence Explorer")
st.caption("Browse individual goal kick sequences with pitch snapshots at key moments.")

# ── Load metadata (lightweight) ────────────────────────────────────────────────
meta = load_sequence_metadata()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

teams = sorted(meta["possession_team_name"].dropna().unique())
team_filter = st.sidebar.selectbox("Team", ["All"] + teams)

period_filter = st.sidebar.selectbox("Period", ["Both", "1", "2"])

final_third_only = st.sidebar.checkbox("Final third only")

min_obv = float(meta["obv_total_seq"].min())
max_obv = float(meta["obv_total_seq"].max())
obv_threshold = st.sidebar.slider(
    "Min OBV total", min_value=round(min_obv, 2), max_value=round(max_obv, 2),
    value=round(min_obv, 2), step=0.01,
)

# Apply filters
filtered = meta.copy()
if team_filter != "All":
    filtered = filtered[filtered["possession_team_name"] == team_filter]
if period_filter != "Both":
    filtered = filtered[filtered["period"] == int(period_filter)]
if final_third_only:
    filtered = filtered[filtered["final_third_entry"] == True]
filtered = filtered[filtered["obv_total_seq"] >= obv_threshold]

st.sidebar.caption(f"{len(filtered):,} sequences match")

if filtered.empty:
    st.warning("No sequences match the current filters.")
    st.stop()

# ── Sequence selector ──────────────────────────────────────────────────────────
def seq_label(row):
    opponent = row["away_team"] if row["possession_team_name"] == row["home_team"] else row["home_team"]
    return (
        f"#{int(row['goal_kick_id'])} — {row['possession_team_name']} vs {opponent} "
        f"| OBV: {row['obv_total_seq']:.4f}"
    )

filtered = filtered.copy()
filtered["_label"] = filtered.apply(seq_label, axis=1)
label_to_id = dict(zip(filtered["_label"], filtered["goal_kick_id"]))

# Default to highest OBV sequence
default_label = filtered.sort_values("obv_total_seq", ascending=False).iloc[0]["_label"]
default_idx   = list(label_to_id.keys()).index(default_label)

selected_label = st.selectbox("Select a goal kick sequence", list(label_to_id.keys()), index=default_idx)
selected_id    = label_to_id[selected_label]
selected_meta  = filtered[filtered["goal_kick_id"] == selected_id].iloc[0]

# ── Summary metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("OBV Total",           f"{selected_meta['obv_total_seq']:.4f}")
c2.metric("Progressive Passes",  int(selected_meta["n_progressive_passes"]))
c3.metric("Progressive Carries", int(selected_meta["n_progressive_carries"]))
c4.metric("Final Third",         "Yes" if selected_meta["final_third_entry"] else "No")
c5.metric("Period",              int(selected_meta["period"]))

st.markdown("---")

# ── Pitch snapshots ────────────────────────────────────────────────────────────
st.subheader("Pitch Snapshots — Kick / 40% / 80%")

with st.spinner("Loading sequence frames..."):
    seq_df = load_sequence_for_id(selected_id)

if seq_df.empty:
    st.error(f"No tracking data found for sequence #{selected_id}.")
    st.stop()

poss_team = selected_meta["possession_team_name"]
figs = draw_three_snapshots(seq_df, possession_team=poss_team)
labels = ["At Kick-off", "At 40% Mark", "At 80% Mark"]

cols = st.columns(3)
for col, fig, label in zip(cols, figs, labels):
    with col:
        st.caption(label)
        st.pyplot(fig)
        plt.close(fig)

st.markdown("---")

# ── Animation ─────────────────────────────────────────────────────────────────
st.subheader("Build-up Animation (first 80%)")
st.caption("Two-panel: pitch tracking (left) + cumulative OBV chart (right). Animates up to the 80% mark. Generation takes ~5–15 seconds.")

n_frames    = seq_df["frame"].nunique()
n_frames_80 = max(1, int(n_frames * 0.8))

# Use session_state to persist animation HTML across button/action clicks
anim_key = f"anim_html_{selected_id}"
if st.button(f"Generate animation ({n_frames_80} of {n_frames} frames)"):
    with st.spinner("Building animation..."):
        st.session_state[anim_key] = build_animation(seq_df, possession_team=poss_team, cutoff_frac=0.8)

if anim_key in st.session_state:
    components.html(st.session_state[anim_key], height=1050, scrolling=False)

st.markdown("---")

# ── What-if prediction buttons ────────────────────────────────────────────────
st.subheader("What happens next? Predict the remaining OBV")
st.caption(
    "At the 80% mark of this sequence, choose an action type. "
    "The model predicts how much OBV the team accumulates in the final 20%."
)

mf     = load_model_features()
bundle = load_model()

seq_features = mf[mf["goal_kick_id"] == selected_id]

if seq_features.empty:
    st.info("This sequence is not in the model dataset — prediction unavailable.")
else:
    base_row      = seq_features.iloc[0].copy()
    actual_action = base_row.get("action_type", "Unknown")
    base_pred     = predict_single(base_row, bundle)["predicted"]

    ACTION_BUTTONS = [
        ("Pass",          "Pass"),
        ("Carry",         "Carry"),
        ("Ball Receipt*", "Ball Receipt"),
        ("Duel",          "Duel"),
        ("Pressure",      "Pressure"),
        ("Ball Recovery", "Ball Recovery"),
        ("Clearance",     "Clearance"),
        ("Shot",          "Shot"),
    ]

    st.markdown(f"**Actual action in this sequence:** {actual_action}")

    # Precompute all predictions so results display without a second rerun
    all_predictions = {}
    for action, _ in ACTION_BUTTONS:
        row = base_row.copy()
        row["action_type"] = action
        all_predictions[action] = predict_single(row, bundle)["predicted"]

    # Show buttons — selected action stored in session_state
    btn_key = f"selected_action_{selected_id}"
    cols = st.columns(len(ACTION_BUTTONS))
    for col, (action, label) in zip(cols, ACTION_BUTTONS):
        if col.button(label, key=f"action_{selected_id}_{action}"):
            st.session_state[btn_key] = action

    selected_action = st.session_state.get(btn_key)

    if selected_action:
        predicted = all_predictions[selected_action]
        delta     = predicted - base_pred
        actual    = base_row["obv_remaining"]

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Action", selected_action)
        c2.metric("Predicted OBV (final 20%)", f"{predicted:+.4f}",
                  delta=f"{delta:+.4f} vs actual action")
        c3.metric("True OBV Remaining", f"{actual:+.4f}")

        # Bar chart comparing all action predictions
        import plotly.graph_objects as go
        actions   = [label for _, label in ACTION_BUTTONS]
        act_keys  = [action for action, _ in ACTION_BUTTONS]
        pred_vals = [all_predictions[a] for a in act_keys]
        colors    = [
            "#ffbe0b" if a == selected_action else
            ("#3a86ff" if v >= 0 else "#ff006e")
            for a, v in zip(act_keys, pred_vals)
        ]

        fig = go.Figure(go.Bar(
            x=actions, y=pred_vals,
            marker_color=colors,
            text=[f"{v:+.4f}" for v in pred_vals],
            textposition="outside",
        ))
        fig.add_hline(y=actual, line_dash="dash", line_color="white",
                      annotation_text=f"True: {actual:+.4f}", annotation_font_color="white")
        fig.update_layout(
            title="Predicted OBV by Action Type at 80% Mark",
            yaxis_title="Predicted OBV (final 20%)",
            paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
            font_color="white", title_font_color="white",
            height=350, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Raw event data ─────────────────────────────────────────────────────────────
with st.expander("Sequence event data"):
    event_cols = [
        "event_id", "event_type", "event_player_name", "event_team_name",
        "event_minute", "event_second",
        "obv_for_net", "obv_against_net", "obv_total_net",
    ]
    available = [c for c in event_cols if c in seq_df.columns]
    events_df = (
        seq_df[available]
        .drop_duplicates("event_id")
        .sort_values(["event_minute", "event_second"])
        .reset_index(drop=True)
    )
    st.dataframe(events_df, use_container_width=True)
