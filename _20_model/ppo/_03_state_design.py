# Import require internal Packages
from _00_environment.constants import BALL_TOUCHING_GROUND_Y_COORD
from _00_environment.constants import GROUND_WIDTH


def normalize_minmax(value, minimum_value, maximum_value):
    """====================================================================================================
    ## Min-Max Normalization Wrapper
    ===================================================================================================="""
    if maximum_value <= minimum_value:
        return 0.0

    normalized_value = (float(value) - float(minimum_value)) / \
        (float(maximum_value) - float(minimum_value))

    if normalized_value < 0.0:
        return 0.0
    if normalized_value > 1.0:
        return 1.0
    return float(normalized_value)


def map_action_to_group(action_name):
    action_name = str(action_name or "normal")
    action_group_code = {
        "normal": 0,
        "jump": 1,
        "dive": 2,
        "spike": 3,
    }

    if action_name in ("jump", "jump_forward", "jump_backward"):
        group_name = "jump"
    elif action_name in ("dive_forward", "dive_backward"):
        group_name = "dive"
    elif action_name.startswith("spike_"):
        group_name = "spike"
    else:
        group_name = "normal"

    group_code = int(action_group_code[group_name])
    return normalize_minmax(group_code, 0, len(action_group_code) - 1)


def get_recent_move_direction(opponent_x_history):
    if len(opponent_x_history) < 2:
        return 0.0

    delta_x = float(opponent_x_history[-1]) - float(opponent_x_history[0])
    if delta_x > 1.0:
        direction = 1.0
    elif delta_x < -1.0:
        direction = -1.0
    else:
        direction = 0.0
    return direction


def get_jump_frequency(opponent_jump_history):
    if len(opponent_jump_history) == 0:
        return 0.0
    jump_count = float(sum(int(flag) for flag in opponent_jump_history))
    return jump_count / float(len(opponent_jump_history))


def calculate_state_key(materials, history_context=None):
    """====================================================================================================
    ## Compact State Design with Opponent Recent Pattern
    ===================================================================================================="""
    if history_context is None:
        history_context = {}

    velocity_min = -30
    velocity_max = 30

    raw = materials["raw"]
    self_raw = raw["self"]
    opponent_raw = raw["opponent"]
    ball_raw = raw["ball"]
    score_raw = raw.get("score", {})

    self_x = normalize_minmax(float(self_raw["x"]), 0, GROUND_WIDTH - 1)
    self_y = normalize_minmax(float(self_raw["y"]), 0, BALL_TOUCHING_GROUND_Y_COORD)
    opponent_x = normalize_minmax(float(opponent_raw["x"]), 0, GROUND_WIDTH - 1)
    ball_x = normalize_minmax(float(ball_raw["x"]), 0, GROUND_WIDTH - 1)
    ball_vx_norm = normalize_minmax(
        float(ball_raw["x_velocity"]),
        velocity_min,
        velocity_max,
    )

    ball_to_self_flag = 1.0 if str(ball_raw.get("side", "self")) == "self" else 0.0

    self_score = float(score_raw.get("self", 0.0))
    opponent_score = float(score_raw.get("opponent", 0.0))
    score_diff = self_score - opponent_score
    score_diff_norm = normalize_minmax(score_diff, -15.0, 15.0)

    rally_step = int(raw.get("rally_step", 0))
    serve_phase_flag = 1.0 if rally_step <= 20 else 0.0

    opponent_action_history = list(history_context.get("opponent_action_history", []))
    opponent_prev_action = "normal"
    if len(opponent_action_history) > 0:
        opponent_prev_action = opponent_action_history[-1]
    opponent_prev_action_group = map_action_to_group(opponent_prev_action)

    opponent_x_history = list(history_context.get("opponent_x_history", []))
    opponent_move_dir = get_recent_move_direction(opponent_x_history)
    opponent_move_dir_3step = normalize_minmax(opponent_move_dir, -1.0, 1.0)

    opponent_jump_history = list(history_context.get("opponent_jump_history", []))
    opponent_jump_freq_3step = normalize_minmax(
        get_jump_frequency(opponent_jump_history),
        0.0,
        1.0,
    )

    DESIGNED_STATE_VECTOR = [
        self_x,
        self_y,
        opponent_x,
        ball_x,
        ball_vx_norm,
        ball_to_self_flag,
        score_diff_norm,
        serve_phase_flag,
        opponent_prev_action_group,
        opponent_move_dir_3step,
        opponent_jump_freq_3step,
    ]
    return DESIGNED_STATE_VECTOR


def get_state_dim():
    """====================================================================================================
    ## Get the Dimension of Designed State Vector
    ===================================================================================================="""
    return 11
