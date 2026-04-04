"""Page 1 — Dataset overview and model performance summary."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_goal_kick_features, load_model_features
from utils.model_inference import load_model, predict_batch

st.set_page_config(page_title="Overview", layout="wide")
st.title("Dataset & Model Overview")
st.caption("J1 League 2024 — Goal Kick Pressing Analysis")

# ── Load data ─────────────────────────────────────────────────────────────────
gk      = load_goal_kick_features()
mf      = load_model_features()
bundle  = load_model()
results = bundle["results"]

# Run batch predictions once
mf_pred = predict_batch(id(bundle), mf)

# ── Header metrics ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
col1.metric("Sequences", f"{len(gk):,}")
col2.metric("In Model", f"{len(mf):,}")
col3.metric("Matches", "376")
col4.metric("Teams", str(gk["possession_team_name"].nunique()))
best_name = bundle.get("model_name", "Gradient Boosting")
col5.metric("R² (test)", f"{results[best_name]['R2']:.3f}")
col6.metric("RMSE", f"{results[best_name]['RMSE']:.4f}")
col7.metric("MAE", f"{results[best_name]['MAE']:.4f}")
col8.metric("CV R²", f"{results[best_name]['CV_R2']:.3f}")

st.markdown("---")
st.markdown(
    f"The model uses **{best_name}** to predict the **Sequence Progression Value** "
    "a team generates in the **final 20%** of a goal kick build-up sequence, "
    "based on features from the first 80%: defensive pressure shape, "
    "ball progression, and match context. "
    "Sequence Progression Value is a custom metric inspired by VAEP, measuring "
    "how much each action shifts the probability of scoring vs. conceding."
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Dataset Distributions", "Model Performance", "Model Comparison"])

# ── Tab 1: Dataset ─────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            mf, x="prog_value_20", nbins=60,
            title="Distribution of Sequence Progression Value",
            labels={"prog_value_20": "Sequence Progression Value"},
            color_discrete_sequence=["#3a86ff"],
        )
        mean_val = mf["prog_value_20"].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="white",
                      annotation_text=f"Mean: {mean_val:.3f}", annotation_font_color="white")
        fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
                          font_color="white", title_font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        team_counts = gk["possession_team_name"].value_counts().reset_index()
        team_counts.columns = ["Team", "Goal Kicks"]
        fig2 = px.bar(
            team_counts.sort_values("Goal Kicks"),
            x="Goal Kicks", y="Team", orientation="h",
            title="Goal Kicks by Team",
            color_discrete_sequence=["#3a86ff"],
        )
        fig2.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
                           font_color="white", title_font_color="white",
                           height=500, yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        ft_counts = gk["final_third_entry"].value_counts().reset_index()
        ft_counts.columns = ["Reached Final Third", "Count"]
        ft_counts["Reached Final Third"] = ft_counts["Reached Final Third"].map({True: "Yes", False: "No"})
        fig3 = px.pie(
            ft_counts, names="Reached Final Third", values="Count",
            title="Sequences Reaching the Final Third",
            color_discrete_sequence=["#3a86ff", "#ff006e"],
        )
        fig3.update_layout(paper_bgcolor="#1a1a2e", font_color="white", title_font_color="white")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.histogram(
            gk, x="sequence_start_time", nbins=50,
            title="When in the Match Goal Kicks Occur (seconds)",
            labels={"sequence_start_time": "Match Timestamp (s)"},
            color_discrete_sequence=["#ffbe0b"],
        )
        fig4.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
                           font_color="white", title_font_color="white")
        st.plotly_chart(fig4, use_container_width=True)

# ── Tab 2: Model Performance ───────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            mf_pred, x="prog_value_20", y="predicted_progression",
            color="possession_team_name",
            hover_data=["goal_kick_id", "prog_value_20", "predicted_progression", "residual"],
            title="Actual vs Predicted Progression Value",
            labels={"prog_value_20": "Actual Progression Value", "predicted_progression": "Predicted Progression Value"},
            opacity=0.6,
        )
        # Perfect prediction diagonal
        rng = [mf_pred["prog_value_20"].min(), mf_pred["prog_value_20"].max()]
        fig.add_trace(go.Scatter(x=rng, y=rng, mode="lines",
                                  line=dict(color="grey", dash="dash"), name="Perfect"))
        fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
                          font_color="white", title_font_color="white",
                          legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            mf_pred, x="residual", nbins=60,
            title="Residual Distribution (actual − predicted)",
            labels={"residual": "Residual"},
            color_discrete_sequence=["#8338ec"],
        )
        mean_res = mf_pred["residual"].mean()
        std_res  = mf_pred["residual"].std()
        fig2.add_vline(x=0, line_dash="dash", line_color="white")
        fig2.add_annotation(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text=f"Mean: {mean_res:.4f}<br>Std: {std_res:.4f}",
            showarrow=False, font=dict(color="white", size=11),
            align="left",
        )
        fig2.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#12122a",
                           font_color="white", title_font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    feat_img = Path(__file__).resolve().parents[2] / "model" / "feature_importance_progression.png"
    if feat_img.exists():
        st.subheader("Feature Importance (Ridge Coefficients)")
        st.image(str(feat_img), use_column_width=True)
    else:
        st.info("Feature importance chart not found at model/feature_importance_progression.png")

# ── Tab 3: Model Comparison ────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Comparison — Ridge vs Random Forest vs Gradient Boosting")
    st.caption(f"**{best_name}** was selected as the best model based on RMSE and cross-validation R².")

    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "RMSE":  round(metrics["RMSE"],  4),
            "MAE":   round(metrics["MAE"],   4),
            "R²":    round(metrics["R2"],    4),
            "CV R²": round(metrics["CV_R2"], 4),
        })
    comp_df = pd.DataFrame(rows).set_index("Model")

    def highlight_best(col):
        """Green for the best value per column (lowest RMSE/MAE, highest R²/CV R²)."""
        if col.name in ("RMSE", "MAE"):
            best = col.min()
        else:
            best = col.max()
        return ["background-color: #1a5c2a; color: white" if v == best else "" for v in col]

    st.dataframe(comp_df.style.apply(highlight_best, axis=0), use_container_width=True)
