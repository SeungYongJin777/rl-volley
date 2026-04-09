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


def calculate_state_key(materials):
    """====================================================================================================
    ## Configuration for Action Group Mapping and Normalization
    ===================================================================================================="""
    # Configuration for action group mapping
    action_group_code = {
        "normal": 0,
        "jump": 1,
        "dive": 2,
        "spike": 3,
    }

    # Configuration for normalization
    velocity_min = -30
    velocity_max = 30

    """====================================================================================================
    ## Load Raw Materials for Calculating State Key
    ===================================================================================================="""
    # Load Raw Materials
    raw = materials["raw"]
    materials = {
        "self_position": (raw["self"]["x"], raw["self"]["y"]),
        "self_action_name": raw["self"]["action_name"],
        "opponent_position": (raw["opponent"]["x"], raw["opponent"]["y"]),
        "opponent_action_name": raw["opponent"]["action_name"],
        "ball_position": (raw["ball"]["x"], raw["ball"]["y"]),
        "ball_velocity": (raw["ball"]["x_velocity"], raw["ball"]["y_velocity"]),
        "expected_landing_x": raw["ball"]["expected_landing_x"],
    }

    # Self Position Normalization (Original x 0~431, y 0~252)
    self_x, self_y = materials["self_position"]
    self_x = normalize_minmax(float(self_x), 0, GROUND_WIDTH - 1)
    self_y = normalize_minmax(float(self_y), 0, BALL_TOUCHING_GROUND_Y_COORD)

    # Self Action Group Mapping (Original action names like "normal", "jump_forward", "dive_backward", "spike_high", etc.)
    self_action_group = str(materials["self_action_name"])
    if self_action_group in ("jump", "jump_forward", "jump_backward"):
        self_action_group = "jump"

    elif self_action_group in ("dive_forward", "dive_backward"):
        self_action_group = "dive"

    elif self_action_group.startswith("spike_"):
        self_action_group = "spike"
    else:
        self_action_group = "normal"

    self_action_group = int(action_group_code[self_action_group])
    self_action_group = normalize_minmax(
        self_action_group, 0, len(action_group_code) - 1)

    # Opponent Position Normalization (Original x 0~431, y 0~252)
    opponent_x, opponent_y = materials["opponent_position"]
    opponent_x = normalize_minmax(
        float(opponent_x), 0, GROUND_WIDTH - 1)
    opponent_y = normalize_minmax(
        float(opponent_y), 0, BALL_TOUCHING_GROUND_Y_COORD)

    # Opponent Action Group Mapping (Original action names like "normal", "jump_forward", "dive_backward", "spike_high", etc.)
    opponent_action_group = str(materials["opponent_action_name"])
    if opponent_action_group in ("jump", "jump_forward", "jump_backward"):
        opponent_action_group = "jump"

    elif opponent_action_group in ("dive_forward", "dive_backward"):
        opponent_action_group = "dive"

    elif opponent_action_group.startswith("spike_"):
        opponent_action_group = "spike"

    else:
        opponent_action_group = "normal"

    opponent_action_group = int(action_group_code[opponent_action_group])
    opponent_action_group = normalize_minmax(
        opponent_action_group, 0, len(action_group_code) - 1)

    # Ball Position Normalization (Original x 0~431, y 0~252)
    ball_x, ball_y = materials["ball_position"]
    ball_x = normalize_minmax(float(ball_x), 0, GROUND_WIDTH - 1)
    ball_y = normalize_minmax(float(ball_y), 0, BALL_TOUCHING_GROUND_Y_COORD)

    # Ball Velocity Normalization (Original vx about -30~30, vy about -30~30)
    ball_velocity_x, ball_velocity_y = materials["ball_velocity"]
    ball_velocity_x = float(ball_velocity_x)
    ball_velocity_y = float(ball_velocity_y)

    ball_velocity_x = normalize_minmax(
        ball_velocity_x, velocity_min, velocity_max)
    ball_velocity_y = normalize_minmax(
        ball_velocity_y, velocity_min, velocity_max)

    # Ecpected Landing X Normalization (Original expected landing x about 0~431)
    landing_x = float(materials["expected_landing_x"])
    landing_x = normalize_minmax(landing_x, 0, GROUND_WIDTH - 1)

    """====================================================================================================
    ## State Vector Construction
    ===================================================================================================="""
    # State Vector: You Can Select and Order the Feature as You Like
    DESIGNED_STATE_VECTOR = [
        self_x,
        self_y,
        self_action_group,
        opponent_x,
        opponent_y,
        opponent_action_group,
        ball_x,
        ball_y,
        ball_velocity_x,
        ball_velocity_y,
        landing_x,
    ]

    # Return the Constructed State Vector
    return DESIGNED_STATE_VECTOR


def get_state_dim():
    """====================================================================================================
    ## Get the Dimension of Designed State Vector
    ===================================================================================================="""
    return 11
