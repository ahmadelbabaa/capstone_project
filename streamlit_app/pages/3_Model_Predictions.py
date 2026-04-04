"""Page 3 — Model Predictions (primary demo page)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.data_loader import load_model_features, load_sequence_metadata
from utils.model_inference import load_model, predict_batch, predict_single, FEATURE_LABELS

st.set_page_config(page_title="Model Predictions", layout="wide")
st.title("Model Predictions")
st.caption("Gradient Boosting — Sequence Progression Value of goal kick build-up sequences")

# ── Load data ─────────────────────────────────────────────────────────────────
mf     = load_model_features()
meta   = load_sequence_metadata()
bundle = load_model()
mf_pred = predict_batch(id(bundle), mf)

# ── Sidebar: filters and sequence selector ────────────────────────────────────
st.sidebar.header("Select a Sequence")

teams = sorted(mf_pred["possession_team_name"].dropna().unique())
team_filter = st.sidebar.selectbox("Filter by team", ["All"] + teams)

filtered = mf_pred.copy()
if team_filter != "All":
    filtered = filtered[filtered["possession_team_name"] == team_filter]

# Enrich with opponent name from metadata
filtered = filtered.merge(
    meta[["goal_kick_id", "home_team", "away_team"]],
    on="goal_kick_id", how="left",
)

def seq_label(row):
    opponent = row["away_team"] if row["possession_team_name"] == row["home_team"] else row["home_team"]
    return (
        f"#{int(row['goal_kick_id'])} — {row['possession_team_name']} vs {opponent} "
        f"| Progression value: {row['prog_value_20']:.4f}"
    )

filtered["_label"] = filtered.apply(seq_label, axis=1)
label_to_id = dict(zip(filtered["_label"], filtered["goal_kick_id"]))

if filtered.empty:
    st.warning("No sequences match the current filter.")
    st.stop()

selected_label = st.sidebar.selectbox("Goal kick sequence", list(label_to_id.keys()))
selected_id    = label_to_id[selected_label]
selected_row   = filtered[filtered["goal_kick_id"] == selected_id].iloc[0]

st.sidebar.caption(
    f"Showing {len(filtered):,} sequences"
    + (f" for {team_filter}" if team_filter != "All" else "")
)
st.sidebar.caption("Only sequences included in the model dataset are shown (5,113 total).")

# ── Section 1: Prediction result ──────────────────────────────────────────────
st.subheader("Prediction Result")

result = predict_single(selected_row, bundle)
actual    = result["actual"]
predicted = result["predicted"]
residual  = result["residual"]

c1, c2, c3 = st.columns(3)
c1.metric("Actual Progression Value", f"{actual:.4f}")
c2.metric("Predicted Progression Value", f"{predicted:.4f}", delta=f"{predicted - actual:+.4f}")
c3.metric("Residual (actual − predicted)", f"{residual:+.4f}")

if residual > 0.01:
    st.success(
        f"The model **underestimated** this sequence — the build-up was better than expected "
        f"(residual = +{residual:.4f})."
    )
elif residual < -0.01:
    st.warning(
        f"The model **overestimated** this sequence — the build-up underperformed expectations "
        f"(residual = {residual:.4f})."
    )
else:
    st.info(f"The model predicted this sequence accurately (residual ≈ 0).")

st.markdown("---")

# ── Section 2: Feature importance ─────────────────────────────────────────────
st.subheader("Feature Importance")

contributions    = result["feature_contributions"].copy()
is_directional   = result.get("is_directional", False)

# Sort by absolute value descending
contributions = contributions.reindex(
    contributions.abs().sort_values(ascending=False).index
)

labels = [
    f"{FEATURE_LABELS.get(f, f)}<br>({selected_row.get(f, 'N/A'):.3g})"
    if isinstance(selected_row.get(f, ""), (int, float, np.integer, np.floating))
    else f"{FEATURE_LABELS.get(f, f)}<br>({selected_row.get(f, 'N/A')})"
    for f in contributions.index
]

if is_directional:
    st.caption(
        "Each bar shows how much a feature pushed the prediction up (blue) or down (red). "
        "Raw feature value shown in parentheses."
    )
    measure  = ["relative"] * len(contributions) + ["total"]
    x_labels = labels + ["<b>Predicted Progression Value</b>"]
    y_values = list(contributions.values) + [bundle["model"].intercept_]

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=x_labels,
        y=y_values,
        connector=dict(line=dict(color="grey", width=0.5)),
        increasing=dict(marker=dict(color="#3a86ff")),
        decreasing=dict(marker=dict(color="#ff006e")),
        totals=dict(marker=dict(color="#ffbe0b")),
        texttemplate="%{y:.4f}",
        textposition="outside",
    ))
    fig_wf.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
        font_color="white", title_font_color="white",
        height=450,
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(title="Contribution to Predicted Progression Value"),
        showlegend=False,
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig_wf, use_container_width=True)
else:
    st.caption(
        "Relative importance of each feature in the model (larger = more influential). "
        "Raw feature value shown in parentheses."
    )
    bar_colors = ["#3a86ff"] * len(contributions)
    fig_wf = go.Figure(go.Bar(
        x=list(contributions.values),
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        texttemplate="%{x:.4f}",
        textposition="outside",
    ))
    fig_wf.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
        font_color="white", title_font_color="white",
        height=450,
        xaxis=dict(title="Feature Importance"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        showlegend=False,
        margin=dict(t=30, b=20, l=220),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

st.markdown("---")

# ── Section 3: Feature values vs dataset ──────────────────────────────────────
st.subheader("Feature Values vs Dataset")

feature_cols  = bundle["feature_cols"]
dataset_means = mf[feature_cols].mean()
dataset_stds  = mf[feature_cols].std()

# Build comparison table
rows_data = []
for f in feature_cols:
    val = selected_row.get(f, float("nan"))
    mu  = dataset_means[f]
    sd  = dataset_stds[f]
    z   = (val - mu) / sd if sd > 0 else 0.0
    rows_data.append({
        "Feature":        FEATURE_LABELS.get(f, f),
        "This Sequence":  round(float(val), 4),
        "Dataset Mean":   round(float(mu), 4),
        "Dataset Std":    round(float(sd), 4),
        "Z-score":        round(float(z), 2),
    })
table_df = pd.DataFrame(rows_data)

c1, c2 = st.columns([1.2, 1])

with c1:
    st.dataframe(table_df, use_container_width=True, height=420)

with c2:
    # Radar chart: top 8 features by importance, z-scores clipped to ±3
    model = bundle["model"]
    if hasattr(model, "coef_"):
        importances = pd.Series(model.coef_, index=feature_cols)
    else:
        importances = pd.Series(model.feature_importances_, index=feature_cols)
    top8  = importances.abs().nlargest(8).index.tolist()
    z_scores = {}
    for f in top8:
        val = selected_row.get(f, float("nan"))
        mu, sd = dataset_means[f], dataset_stds[f]
        z = (val - mu) / sd if sd > 0 else 0.0
        z_scores[f] = float(np.clip(z, -3, 3))

    radar_labels = [FEATURE_LABELS.get(f, f) for f in top8]
    radar_values = list(z_scores.values())
    # Close the loop
    radar_labels_closed = radar_labels + [radar_labels[0]]
    radar_values_closed = radar_values + [radar_values[0]]

    fig_radar = go.Figure()
    # Dataset mean baseline (all zeros in z-score space)
    fig_radar.add_trace(go.Scatterpolar(
        r=[0] * (len(top8) + 1), theta=radar_labels_closed,
        fill="toself", name="Dataset Mean",
        line=dict(color="grey", dash="dash"), fillcolor="rgba(128,128,128,0.1)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values_closed, theta=radar_labels_closed,
        fill="toself", name="This Sequence",
        line=dict(color="#3a86ff"), fillcolor="rgba(58,134,255,0.2)",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-3, 3], tickfont=dict(color="white", size=8)),
            angularaxis=dict(tickfont=dict(color="white", size=9)),
            bgcolor="#12122a",
        ),
        paper_bgcolor="#1a1a2e", font_color="white",
        legend=dict(font=dict(color="white")),
        title=dict(text="Top 8 Features (z-scores)", font=dict(color="white")),
        height=420,
        margin=dict(t=60, b=20, l=60, r=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── Section 4: Compare to full dataset ────────────────────────────────────────
st.subheader("Position in Full Dataset")

fig_scatter = px.scatter(
    mf_pred, x="prog_value_20", y="predicted_progression",
    color="possession_team_name",
    opacity=0.4,
    labels={"prog_value_20": "Actual Progression Value", "predicted_progression": "Predicted Progression Value"},
    title="All Predictions — Selected Sequence Highlighted",
    hover_data=["goal_kick_id", "possession_team_name"],
)
# Perfect prediction diagonal
rng = [mf_pred["prog_value_20"].min(), mf_pred["prog_value_20"].max()]
fig_scatter.add_trace(go.Scatter(
    x=rng, y=rng, mode="lines",
    line=dict(color="grey", dash="dash"), name="Perfect", showlegend=False,
))
# Highlighted point
fig_scatter.add_trace(go.Scatter(
    x=[actual], y=[predicted],
    mode="markers",
    marker=dict(symbol="star", size=18, color="#ffbe0b", line=dict(color="white", width=1)),
    name=f"Selected #{selected_id}",
    showlegend=True,
))
fig_scatter.update_layout(
    paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
    font_color="white", title_font_color="white",
    legend=dict(font=dict(size=9)),
)
st.plotly_chart(fig_scatter, use_container_width=True)

with st.expander("All predictions table"):
    display_cols = ["goal_kick_id", "possession_team_name", "period",
                    "prog_value_20", "predicted_progression", "residual"]
    st.dataframe(
        mf_pred[display_cols].sort_values("residual", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )
