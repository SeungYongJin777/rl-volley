# Import require internal Packages
from _00_environment.actions import ACTION_NAMES
from _00_environment.constants import BALL_TOUCHING_GROUND_Y_COORD
from _00_environment.constants import GROUND_WIDTH


def bucket(value, minimum_value, maximum_value, bucket_count):
    """====================================================================================================
    ## Bucket Function: Mapping a continuous value to a discrete bucket index.
    ===================================================================================================="""
    # If the value is less than or equal to the minimum value, return the first bucket.
    if value <= minimum_value:
        return 0

    # If the value is greater than or equal to the maximum value, return the last bucket.
    if value >= maximum_value:
        return bucket_count - 1

    # Calculate the ratio of the value to calculate the bucket index
    ratio = (value - minimum_value) / (maximum_value - minimum_value)

    # Map the ratio to a bucket index
    out = int(ratio * bucket_count)

    # Return the bucket index
    return min(out, bucket_count - 1)


def calculate_state_key(materials):
    """====================================================================================================
    ## Configuration for Action Group Mapping and Bucketing
    ===================================================================================================="""
    # Configuration for action group mapping
    action_group_code = {
        "normal": 0,
        "jump": 1,
        "dive": 2,
        "spike": 3,
    }

    # Configuration for bucketing
    position_bucket_count = 10
    velocity_bucket_count = 10
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

    # Self Position Bucket (Original x 0~431, y 0~252)
    self_x, self_y = materials["self_position"]
    self_x = int(self_x)
    self_y = int(self_y)

    self_x = bucket(self_x, 0, GROUND_WIDTH - 1,
                    position_bucket_count)

    self_y = bucket(self_y, 0, BALL_TOUCHING_GROUND_Y_COORD,
                    position_bucket_count)

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

    # Opponent Position Bucket (Original x 0~431, y 0~252)
    opponent_x, opponent_y = materials["opponent_position"]
    opponent_x = int(opponent_x)
    opponent_y = int(opponent_y)

    opponent_x = bucket(opponent_x, 0, GROUND_WIDTH - 1,
                        position_bucket_count)
    opponent_y = bucket(opponent_y, 0, BALL_TOUCHING_GROUND_Y_COORD,
                        position_bucket_count)

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

    # Ball Position Bucket (Original x 0~431, y 0~252)
    ball_x, ball_y = materials["ball_position"]
    ball_x = int(ball_x)
    ball_y = int(ball_y)

    ball_x = bucket(ball_x, 0, GROUND_WIDTH - 1,
                    position_bucket_count)
    ball_y = bucket(ball_y, 0, BALL_TOUCHING_GROUND_Y_COORD,
                    position_bucket_count)

    # Ball Velocity Bucket (Original vx about -30~30, vy about -30~30)
    ball_velocity_x, ball_velocity_y = materials["ball_velocity"]
    ball_velocity_x = float(ball_velocity_x)
    ball_velocity_y = float(ball_velocity_y)

    ball_velocity_x = bucket(
        ball_velocity_x, velocity_min, velocity_max, velocity_bucket_count)
    ball_velocity_y = bucket(
        ball_velocity_y, velocity_min, velocity_max, velocity_bucket_count)

    # Ecpected Landing X Bucket (Original expected landing x about 0~431)
    landing_x = int(materials["expected_landing_x"])
    landing_x = bucket(landing_x, 0, GROUND_WIDTH - 1, position_bucket_count)

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
