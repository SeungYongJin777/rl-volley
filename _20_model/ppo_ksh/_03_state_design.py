from _00_environment.constants import BALL_TOUCHING_GROUND_Y_COORD
from _00_environment.constants import GROUND_WIDTH


def normalize_minmax(value, minimum_value, maximum_value):
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


def maybe_mirror_x(x_value, should_mirror):
    if not should_mirror:
        return float(x_value)
    return float(GROUND_WIDTH - 1) - float(x_value)


def maybe_mirror_vx(vx_value, should_mirror):
    if not should_mirror:
        return float(vx_value)
    return -float(vx_value)


def calculate_state_key(materials):
    action_group_code = {
        "normal": 0,
        "jump": 1,
        "dive": 2,
        "spike": 3,
    }

    velocity_min = -30
    velocity_max = 30

    raw = materials["raw"]

    self_x_raw = float(raw["self"]["x"])
    opponent_x_raw = float(raw["opponent"]["x"])

    # self가 오른쪽 코트에 있으면 좌우 반전
    should_mirror = self_x_raw > opponent_x_raw

    self_x = maybe_mirror_x(raw["self"]["x"], should_mirror)
    self_y = float(raw["self"]["y"])

    opponent_x = maybe_mirror_x(raw["opponent"]["x"], should_mirror)
    opponent_y = float(raw["opponent"]["y"])

    ball_x = maybe_mirror_x(raw["ball"]["x"], should_mirror)
    ball_y = float(raw["ball"]["y"])

    ball_velocity_x = maybe_mirror_vx(raw["ball"]["x_velocity"], should_mirror)
    ball_velocity_y = float(raw["ball"]["y_velocity"])

    landing_x = maybe_mirror_x(raw["ball"]["expected_landing_x"], should_mirror)

    self_action_group = str(raw["self"]["action_name"])
    if self_action_group in ("jump", "jump_forward", "jump_backward"):
        self_action_group = "jump"
    elif self_action_group in ("dive_forward", "dive_backward"):
        self_action_group = "dive"
    elif self_action_group.startswith("spike_"):
        self_action_group = "spike"
    else:
        self_action_group = "normal"

    opponent_action_group = str(raw["opponent"]["action_name"])
    if opponent_action_group in ("jump", "jump_forward", "jump_backward"):
        opponent_action_group = "jump"
    elif opponent_action_group in ("dive_forward", "dive_backward"):
        opponent_action_group = "dive"
    elif opponent_action_group.startswith("spike_"):
        opponent_action_group = "spike"
    else:
        opponent_action_group = "normal"

    designed_state_vector = [
        normalize_minmax(self_x, 0, GROUND_WIDTH - 1),
        normalize_minmax(self_y, 0, BALL_TOUCHING_GROUND_Y_COORD),
        normalize_minmax(action_group_code[self_action_group], 0, len(action_group_code) - 1),

        normalize_minmax(opponent_x, 0, GROUND_WIDTH - 1),
        normalize_minmax(opponent_y, 0, BALL_TOUCHING_GROUND_Y_COORD),
        normalize_minmax(action_group_code[opponent_action_group], 0, len(action_group_code) - 1),

        normalize_minmax(ball_x, 0, GROUND_WIDTH - 1),
        normalize_minmax(ball_y, 0, BALL_TOUCHING_GROUND_Y_COORD),
        normalize_minmax(ball_velocity_x, velocity_min, velocity_max),
        normalize_minmax(ball_velocity_y, velocity_min, velocity_max),
        normalize_minmax(landing_x, 0, GROUND_WIDTH - 1),
    ]

    #print("STATE:", designed_state_vector[:3], designed_state_vector[3:6])

    return designed_state_vector


def get_state_dim():
    return 11
