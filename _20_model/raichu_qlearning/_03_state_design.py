from _00_environment.constants import BALL_TOUCHING_GROUND_Y_COORD
from _00_environment.constants import GROUND_HALF_WIDTH
from _00_environment.constants import GROUND_WIDTH


def bucket(value, minimum_value, maximum_value, bucket_count):
    if value <= minimum_value:
        return 0
    if value >= maximum_value:
        return bucket_count - 1

    ratio = (value - minimum_value) / (maximum_value - minimum_value)
    return min(int(ratio * bucket_count), bucket_count - 1)


def relation_bucket(delta, near=24, far=72):
    if delta <= -far:
        return 0
    if delta <= -near:
        return 1
    if delta < near:
        return 2
    if delta < far:
        return 3
    return 4


def opponent_relation_bucket(delta, near=48):
    if delta <= -near:
        return 0
    if delta < near:
        return 1
    return 2


def player_state_bucket(state_name):
    if state_name == "jump":
        return 1
    if state_name in ("dive", "end"):
        return 2
    return 0


def calculate_state_key(materials):
    raw = materials["raw"]

    self_raw = raw["self"]
    opponent_raw = raw["opponent"]
    ball_raw = raw["ball"]

    self_x = int(self_raw["x"])
    opponent_x = int(opponent_raw["x"])
    ball_y = int(ball_raw["y"])
    ball_vx = float(ball_raw["x_velocity"])
    ball_vy = float(ball_raw["y_velocity"])
    landing_x = int(ball_raw["expected_landing_x"])

    ball_side = 0 if str(ball_raw["side"]) == "self" else 1
    self_x_zone = bucket(self_x, 0, GROUND_HALF_WIDTH, 4)
    landing_zone = bucket(landing_x, 0, GROUND_WIDTH - 1, 6)
    self_to_landing = relation_bucket(self_x - landing_x)

    if ball_y <= BALL_TOUCHING_GROUND_Y_COORD // 3:
        ball_height = 0
    elif ball_y <= (BALL_TOUCHING_GROUND_Y_COORD * 2) // 3:
        ball_height = 1
    else:
        ball_height = 2

    ball_vy_phase = 0 if ball_vy <= 0 else 1

    if ball_vx < -3:
        ball_vx_phase = 0
    elif ball_vx <= 3:
        ball_vx_phase = 1
    else:
        ball_vx_phase = 2

    self_state = player_state_bucket(str(self_raw["state"]))
    opponent_to_landing = opponent_relation_bucket(opponent_x - landing_x)

    return [
        ball_side,
        self_x_zone,
        landing_zone,
        self_to_landing,
        ball_height,
        ball_vy_phase,
        ball_vx_phase,
        self_state,
        opponent_to_landing,
    ]
