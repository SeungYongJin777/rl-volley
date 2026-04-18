from _00_environment.constants import BALL_TOUCHING_GROUND_Y_COORD
from _00_environment.constants import GROUND_WIDTH


VELOCITY_MIN = -30.0
VELOCITY_MAX = 30.0
RELATIVE_X_MIN = -(GROUND_WIDTH - 1)
RELATIVE_X_MAX = GROUND_WIDTH - 1
RELATIVE_Y_MIN = -BALL_TOUCHING_GROUND_Y_COORD
RELATIVE_Y_MAX = BALL_TOUCHING_GROUND_Y_COORD


def clamp(value, minimum_value, maximum_value):
    return max(minimum_value, min(maximum_value, float(value)))


def normalize_minmax(value, minimum_value, maximum_value):
    """====================================================================================================
    ## Min-Max Normalization Wrapper
    ===================================================================================================="""
    if maximum_value <= minimum_value:
        return 0.0

    normalized_value = (float(value) - float(minimum_value)) / (
        float(maximum_value) - float(minimum_value)
    )

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


def map_self_state_to_group(state_name):
    normalized_state = str(state_name or "normal").strip().lower()
    if normalized_state == "jump":
        return 0.5
    if normalized_state == "dive":
        return 1.0
    return 0.0


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


def landing_urgency_phase(ball_to_self_flag, ball_vy, ball_rel_y_to_self):
    if bool(ball_to_self_flag) is not True:
        return 0
    if float(ball_vy) <= 0.0:
        return 0
    if float(ball_rel_y_to_self) > 96.0:
        return 1
    return 2


def hittable_phase(ball_to_self_flag, ball_rel_x_to_self, ball_rel_y_to_self, ball_vy):
    abs_rel_x = abs(float(ball_rel_x_to_self))
    rel_y = float(ball_rel_y_to_self)
    descending_or_neutral = float(ball_vy) >= -1.0

    if bool(ball_to_self_flag) is not True:
        return 0
    if descending_or_neutral and abs_rel_x <= 24.0 and 20.0 <= rel_y <= 120.0:
        return 2
    if abs_rel_x <= 64.0 and 0.0 <= rel_y <= 164.0:
        return 1
    return 0


def normalize_phase_value(phase_value, phase_max_value=2.0):
    return normalize_minmax(float(phase_value), 0.0, float(phase_max_value))


def calculate_state_key(materials, history_context=None):
    """====================================================================================================
    ## Expanded PPO State with Defensive Awareness
    ===================================================================================================="""
    if history_context is None:
        history_context = {}

    raw = materials["raw"]
    self_raw = raw["self"]
    opponent_raw = raw["opponent"]
    ball_raw = raw["ball"]
    score_raw = raw.get("score", {})

    self_x_raw = float(self_raw["x"])
    self_y_raw = float(self_raw["y"])
    opponent_x_raw = float(opponent_raw["x"])
    opponent_y_raw = float(opponent_raw["y"])
    ball_x_raw = float(ball_raw["x"])
    ball_y_raw = float(ball_raw["y"])
    ball_vx_raw = clamp(ball_raw["x_velocity"], VELOCITY_MIN, VELOCITY_MAX)
    ball_vy_raw = clamp(ball_raw["y_velocity"], VELOCITY_MIN, VELOCITY_MAX)
    landing_x_raw = float(ball_raw["expected_landing_x"])

    ball_rel_x_raw = clamp(ball_x_raw - self_x_raw, RELATIVE_X_MIN, RELATIVE_X_MAX)
    ball_rel_y_raw = clamp(self_y_raw - ball_y_raw, RELATIVE_Y_MIN, RELATIVE_Y_MAX)
    landing_rel_x_raw = clamp(
        landing_x_raw - self_x_raw,
        RELATIVE_X_MIN,
        RELATIVE_X_MAX,
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

    landing_phase = landing_urgency_phase(
        ball_to_self_flag=ball_to_self_flag,
        ball_vy=ball_vy_raw,
        ball_rel_y_to_self=ball_rel_y_raw,
    )
    hittable_phase_value = hittable_phase(
        ball_to_self_flag=ball_to_self_flag,
        ball_rel_x_to_self=ball_rel_x_raw,
        ball_rel_y_to_self=ball_rel_y_raw,
        ball_vy=ball_vy_raw,
    )

    designed_state_vector = [
        normalize_minmax(self_x_raw, 0.0, GROUND_WIDTH - 1),
        normalize_minmax(self_y_raw, 0.0, BALL_TOUCHING_GROUND_Y_COORD),
        map_self_state_to_group(self_raw.get("state", "normal")),
        normalize_minmax(opponent_x_raw, 0.0, GROUND_WIDTH - 1),
        normalize_minmax(opponent_y_raw, 0.0, BALL_TOUCHING_GROUND_Y_COORD),
        normalize_minmax(ball_x_raw, 0.0, GROUND_WIDTH - 1),
        normalize_minmax(ball_y_raw, 0.0, BALL_TOUCHING_GROUND_Y_COORD),
        normalize_minmax(ball_vx_raw, VELOCITY_MIN, VELOCITY_MAX),
        normalize_minmax(ball_vy_raw, VELOCITY_MIN, VELOCITY_MAX),
        normalize_minmax(ball_rel_x_raw, RELATIVE_X_MIN, RELATIVE_X_MAX),
        normalize_minmax(ball_rel_y_raw, RELATIVE_Y_MIN, RELATIVE_Y_MAX),
        normalize_minmax(landing_x_raw, 0.0, GROUND_WIDTH - 1),
        normalize_minmax(landing_rel_x_raw, RELATIVE_X_MIN, RELATIVE_X_MAX),
        ball_to_self_flag,
        score_diff_norm,
        serve_phase_flag,
        opponent_prev_action_group,
        opponent_move_dir_3step,
        opponent_jump_freq_3step,
        normalize_phase_value(landing_phase),
        normalize_phase_value(hittable_phase_value),
    ]
    return designed_state_vector


def get_state_dim():
    """====================================================================================================
    ## Get the Dimension of Designed State Vector
    ===================================================================================================="""
    return 21
