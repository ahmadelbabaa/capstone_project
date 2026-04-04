"""
vaep_utils.py
-------------
Shared constants and pure helper functions for the possession value metric.

No I/O — import this module from the other feature engineering scripts.
"""

import math

# ── Action type mapping ────────────────────────────────────────────────────────
# Maps StatsBomb event_type strings to integer codes used as model features.
# Events not in this map are excluded from the action dataset.
ON_BALL_TYPES = {
    "Pass",
    "Carry",
    "Shot",
    "Ball Recovery",
    "Duel",
    "Clearance",
    "Interception",
    "Dribble",
    "Pressure",
    "Goal Keeper",
}

ACTION_TYPE_MAP = {
    "Pass":         0,
    "Carry":        1,
    "Shot":         2,
    "Ball Recovery":3,
    "Duel":         4,
    "Clearance":    5,
    "Interception": 6,
    "Dribble":      7,
    "Pressure":     8,
    "Goal Keeper":  9,
}

# ── Pitch constants (StatsBomb coordinates) ────────────────────────────────────
PITCH_LENGTH = 120.0   # x-axis: 0 (own goal) → 120 (opponent goal)
PITCH_WIDTH  = 80.0    # y-axis: 0 → 80

# Opponent goal centre
GOAL_X = 120.0
GOAL_Y = 40.0

# ── VAEP parameters ────────────────────────────────────────────────────────────
K = 10   # Lookahead window: label = 1 if goal within next K on-ball actions

# ── Per-action feature names (a0_ prefix = current action) ────────────────────
BASE_FEATURES = [
    "action_type",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "dist_to_goal",
    "angle_to_goal",
    "score_diff",
    "minute_norm",
    "period",
]

# Full game-state feature vector (current + 2 lags = 30 features)
GAME_STATE_FEATURES = (
    [f"a0_{f}" for f in BASE_FEATURES]
    + [f"a1_{f}" for f in BASE_FEATURES]
    + [f"a2_{f}" for f in BASE_FEATURES]
)


# ── Helper functions ───────────────────────────────────────────────────────────

def dist_to_goal(x: float, y: float) -> float:
    """
    Euclidean distance from (x, y) to the opponent goal centre (120, 40),
    normalized by pitch length so the result is in [0, ~1].
    """
    return math.sqrt((x - GOAL_X) ** 2 + (y - GOAL_Y) ** 2) / PITCH_LENGTH


def angle_to_goal(x: float, y: float) -> float:
    """
    Angle from (x, y) toward the opponent goal centre, normalized to [-1, 1].
    0 = straight ahead along the x-axis; ±1 = perpendicular.
    """
    return math.atan2(y - GOAL_Y, GOAL_X - x) / math.pi
