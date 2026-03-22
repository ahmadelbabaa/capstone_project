"""Pitch visualisation utilities — static snapshots and animations."""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mplsoccer import Pitch

# SkillCorner pitch dimensions (metres)
SKC_X, SKC_Y = 105.0, 68.0
# StatsBomb pitch dimensions
SB_X, SB_Y   = 120.0, 80.0

POSS_COLOR = "#3a86ff"
DEF_COLOR  = "#ff006e"
BALL_COLOR = "#ffbe0b"
BG_COLOR   = "#1a1a2e"
PITCH_COLOR = "#2d6a2d"


def skc_to_sb(x, y):
    """Convert SkillCorner metres (centre-origin) to StatsBomb coords (0–120, 0–80)."""
    sb_x = (x + SKC_X / 2) / SKC_X * SB_X
    sb_y = (y + SKC_Y / 2) / SKC_Y * SB_Y
    return sb_x, sb_y


def get_snapshot_frame(seq_df, frac: float):
    """
    Return all rows for the single frame nearest to frac * sequence duration.
    frac=0.0 → kick-off frame, frac=0.4 → 40% mark, frac=0.8 → 80% mark.
    """
    frames_sorted = sorted(seq_df["frame"].unique())
    frame_times   = seq_df.groupby("frame")["timestamp_seconds"].first().reindex(frames_sorted)
    t0, t1        = frame_times.iloc[0], frame_times.iloc[-1]
    target_t      = t0 + frac * (t1 - t0)
    nearest_frame = frame_times.sub(target_t).abs().idxmin()
    return seq_df[seq_df["frame"] == nearest_frame].copy()


def draw_pitch_snapshot(
    frame_df,
    title: str,
    possession_team: str,
) -> plt.Figure:
    """
    Draw a single-panel static pitch snapshot for one frame.
    Possession team = blue, defending = red, ball = yellow.
    Returns a matplotlib Figure — caller must call plt.close(fig) after st.pyplot(fig).
    """
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=BG_COLOR)

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=PITCH_COLOR,
        line_color="white",
        stripe=True,
        stripe_color="#2a622a",
        linewidth=1.5,
    )
    pitch.draw(ax=ax)

    if frame_df.empty:
        ax.set_title(title, color="white", fontsize=9, pad=6)
        return fig

    # Convert coordinates
    frame_df = frame_df.copy()
    frame_df["px_sb"], frame_df["py_sb"] = skc_to_sb(
        frame_df["player_x"], frame_df["player_y"]
    )
    bx, by = skc_to_sb(
        frame_df["ball_x"].iloc[0], frame_df["ball_y"].iloc[0]
    )

    # Identify teams
    poss_is_home = possession_team == frame_df["home_team"].iloc[0]
    poss_rows = frame_df[frame_df["is_home"] == poss_is_home]
    def_rows  = frame_df[frame_df["is_home"] != poss_is_home]

    # Plot players
    for rows, color in [(poss_rows, POSS_COLOR), (def_rows, DEF_COLOR)]:
        if rows.empty:
            continue
        ax.scatter(
            rows["px_sb"], rows["py_sb"],
            s=300, c=color, edgecolors="white", linewidths=1.5, zorder=6,
        )
        for _, r in rows.iterrows():
            jn = r["jersey_number"]
            label = str(int(jn)) if not (isinstance(jn, float) and np.isnan(jn)) else ""
            ax.text(
                r["px_sb"], r["py_sb"], label,
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=8,
            )

    # Ball
    ax.scatter([bx], [by], s=200, c=BALL_COLOR, edgecolors="black", linewidths=1.5, zorder=7)

    # Timestamp
    rel_t = frame_df["timestamp_seconds"].iloc[0]
    t0    = None  # caller doesn't pass t0; show absolute time in brief
    ax.text(
        1, 79, f"t = {rel_t:.1f} s",
        ha="left", va="top", fontsize=9, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, alpha=0.7), zorder=10,
    )

    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
    fig.patch.set_facecolor(BG_COLOR)
    return fig


def build_animation(seq_df, possession_team: str, cutoff_frac: float = 1.0) -> str:
    """
    Build the two-panel goal kick animation (pitch + cumulative OBV chart)
    and return the jshtml string for embedding in Streamlit.
    Mirrors the logic from testing/test_goal_kick_animation2.ipynb.
    """
    FPS          = 10
    BALL_TRAIL   = 20
    PLAYER_TRAIL = 8

    seq = seq_df.copy()
    seq["px_sb"], seq["py_sb"] = skc_to_sb(seq["player_x"], seq["player_y"])
    seq["bx_sb"], seq["by_sb"] = skc_to_sb(seq["ball_x"],   seq["ball_y"])

    poss_is_home = possession_team == seq["home_team"].iloc[0]
    def_team     = seq["away_team"].iloc[0] if poss_is_home else seq["home_team"].iloc[0]

    all_frames    = sorted(seq["frame"].unique())
    cutoff_idx    = max(1, int(len(all_frames) * cutoff_frac))
    frames_sorted = all_frames[:cutoff_idx]
    frame_times   = seq.groupby("frame")["timestamp_seconds"].first().reindex(frames_sorted)
    t0            = frame_times.iloc[0]
    rel_times     = (frame_times - t0).values

    # Cumulative OBV timeline
    events = (
        seq.drop_duplicates("event_id")
        [["event_id", "timestamp_seconds", "obv_for_net", "obv_against_net"]]
        .sort_values("timestamp_seconds")
        .copy()
    )
    events["obv_for_net"]     = events["obv_for_net"].fillna(0)
    events["obv_against_net"] = events["obv_against_net"].fillna(0)
    events["cum_for"]         = events["obv_for_net"].cumsum()
    events["cum_against"]     = events["obv_against_net"].cumsum()
    events["rel_t"]           = events["timestamp_seconds"] - t0

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams["animation.embed_limit"] = 100  # MB
    fig = plt.figure(figsize=(18, 7.5), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.06)

    pitch    = Pitch(pitch_type="statsbomb", pitch_color=PITCH_COLOR,
                     line_color="white", stripe=True, stripe_color="#2a622a", linewidth=1.5)
    ax_pitch = fig.add_subplot(gs[0])
    pitch.draw(ax=ax_pitch)

    ax_obv = fig.add_subplot(gs[1])
    ax_obv.set_facecolor("#12122a")
    for spine in ax_obv.spines.values():
        spine.set_color("#444466")
    ax_obv.tick_params(colors="white", labelsize=8)
    ax_obv.set_xlabel("Time (s)", color="white", fontsize=9)
    ax_obv.set_ylabel("Cumulative OBV", color="white", fontsize=9)
    ax_obv.set_title("Cumulative OBV", color="white", fontsize=10, fontweight="bold", pad=8)
    ax_obv.set_xlim(0, rel_times[-1])
    obv_margin = max(0.02, events[["cum_for", "cum_against"]].abs().max().max() * 1.2)
    ax_obv.set_ylim(-obv_margin, obv_margin)
    ax_obv.axhline(0, color="white", linewidth=0.8, alpha=0.4, linestyle="--")
    ax_obv.plot(events["rel_t"], events["cum_for"],     color=POSS_COLOR, alpha=0.15, linewidth=1.5)
    ax_obv.plot(events["rel_t"], events["cum_against"], color=DEF_COLOR,  alpha=0.15, linewidth=1.5)

    obv_line_for, = ax_obv.plot([], [], color=POSS_COLOR, linewidth=2.2, label=f"For ({possession_team})")
    obv_line_agt, = ax_obv.plot([], [], color=DEF_COLOR,  linewidth=2.2, label=f"Against ({def_team})")
    cursor_line   = ax_obv.axvline(0, color="white", linewidth=1, linestyle=":", alpha=0.7)
    obv_dot_for   = ax_obv.scatter([], [], s=60, color=POSS_COLOR, zorder=5)
    obv_dot_agt   = ax_obv.scatter([], [], s=60, color=DEF_COLOR,  zorder=5)
    ax_obv.legend(fontsize=8, facecolor=BG_COLOR, labelcolor="white",
                  edgecolor="#444466", loc="upper left")

    scat_poss = ax_pitch.scatter([], [], s=300, c=POSS_COLOR, edgecolors="white", linewidths=1.5, zorder=6)
    scat_def  = ax_pitch.scatter([], [], s=300, c=DEF_COLOR,  edgecolors="white", linewidths=1.5, zorder=6)
    scat_ball = ax_pitch.scatter([], [], s=180, c=BALL_COLOR, edgecolors="black",  linewidths=1.5, zorder=7)

    jersey_texts = [ax_pitch.text(0, 0, "", ha="center", va="center",
                                   fontsize=7, fontweight="bold", color="white", zorder=8)
                    for _ in range(22)]

    ball_trail_line, = ax_pitch.plot([], [], color=BALL_COLOR, linewidth=2,
                                      alpha=0.6, zorder=5, solid_capstyle="round")
    trail_poss_sc = ax_pitch.scatter([], [], s=35, c=POSS_COLOR, alpha=0.2, edgecolors="none", zorder=4)
    trail_def_sc  = ax_pitch.scatter([], [], s=35, c=DEF_COLOR,  alpha=0.2, edgecolors="none", zorder=4)

    time_text = ax_pitch.text(
        1, 79, "", ha="left", va="top", fontsize=11, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, alpha=0.7), zorder=10)

    home = seq["home_team"].iloc[0]
    away = seq["away_team"].iloc[0]
    ax_pitch.set_title(f"{home} vs {away}  |  Possession: {possession_team}",
                       color="white", fontsize=11, fontweight="bold", pad=8)
    ax_pitch.legend(
        handles=[
            mpatches.Patch(color=POSS_COLOR, label=f"Possession: {possession_team}"),
            mpatches.Patch(color=DEF_COLOR,  label=f"Defending: {def_team}"),
            mpatches.Patch(color=BALL_COLOR, label="Ball"),
        ],
        loc="lower right", fontsize=8, facecolor=BG_COLOR, labelcolor="white", edgecolor="white")

    # Precompute per-frame data
    frame_data = {}
    for fr in frames_sorted:
        fdf = seq[seq["frame"] == fr]
        frame_data[fr] = {
            "poss": fdf[fdf["is_home"] == poss_is_home][["px_sb", "py_sb", "jersey_number"]].values,
            "def":  fdf[fdf["is_home"] != poss_is_home][["px_sb", "py_sb", "jersey_number"]].values,
            "ball": fdf[["bx_sb", "by_sb"]].iloc[0].values,
        }

    ball_hist = []

    def update(fi):
        fr    = frames_sorted[fi]
        fd    = frame_data[fr]
        rel_t = rel_times[fi]

        if len(fd["poss"]): scat_poss.set_offsets(fd["poss"][:, :2])
        if len(fd["def"]):  scat_def.set_offsets(fd["def"][:, :2])

        all_players = list(fd["poss"]) + list(fd["def"])
        for k, txt in enumerate(jersey_texts):
            if k < len(all_players):
                txt.set_position((all_players[k][0], all_players[k][1]))
                jn = all_players[k][2]
                txt.set_text(str(int(jn)) if not np.isnan(jn) else "")
            else:
                txt.set_text("")

        scat_ball.set_offsets([fd["ball"]])
        ball_hist.append(fd["ball"].copy())
        trail = ball_hist[max(0, len(ball_hist) - BALL_TRAIL):]
        if len(trail) >= 2:
            tx, ty = zip(*trail)
            ball_trail_line.set_data(tx, ty)

        p_pts, d_pts = [], []
        for pfi in range(max(0, fi - PLAYER_TRAIL), fi):
            pfd = frame_data[frames_sorted[pfi]]
            p_pts.extend(pfd["poss"][:, :2].tolist())
            d_pts.extend(pfd["def"][:, :2].tolist())
        if p_pts: trail_poss_sc.set_offsets(p_pts)
        if d_pts: trail_def_sc.set_offsets(d_pts)

        time_text.set_text(f"t = {rel_t:.1f} s")

        past = events[events["rel_t"] <= rel_t]
        if not past.empty:
            obv_line_for.set_data(past["rel_t"], past["cum_for"])
            obv_line_agt.set_data(past["rel_t"], past["cum_against"])
            last = past.iloc[-1]
            obv_dot_for.set_offsets([[last["rel_t"], last["cum_for"]]])
            obv_dot_agt.set_offsets([[last["rel_t"], last["cum_against"]]])
        cursor_line.set_xdata([rel_t, rel_t])

        return (scat_poss, scat_def, scat_ball, ball_trail_line,
                trail_poss_sc, trail_def_sc, time_text,
                obv_line_for, obv_line_agt, obv_dot_for, obv_dot_agt,
                cursor_line, *jersey_texts)

    anim = FuncAnimation(fig, update, frames=len(frames_sorted),
                         interval=1000 / FPS, blit=True)
    html_str = anim.to_jshtml(fps=FPS, default_mode="loop")
    plt.close(fig)
    return html_str


def draw_three_snapshots(seq_df, possession_team: str):
    """
    Return three figures: at kick-off (0%), 40% mark, and 80% mark.
    """
    specs = [
        (0.0, "At Kick-off"),
        (0.4, "At 40% Mark"),
        (0.8, "At 80% Mark"),
    ]
    figs = []
    for frac, label in specs:
        frame_df = get_snapshot_frame(seq_df, frac)
        fig = draw_pitch_snapshot(frame_df, title=label, possession_team=possession_team)
        figs.append(fig)
    return figs
