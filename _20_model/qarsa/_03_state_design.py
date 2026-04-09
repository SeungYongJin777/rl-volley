# Import require internal Packages
from _00_environment.constants import GROUND_HALF_WIDTH
from _00_environment.constants import GROUND_WIDTH


STATE_DESIGN_VERSION = 1


def bucket(value, minimum_value, maximum_value, bucket_count):
    if value <= minimum_value:
        return 0
    if value >= maximum_value:
        return bucket_count - 1
    ratio = (value - minimum_value) / (maximum_value - minimum_value)
    out = int(ratio * bucket_count)
    return min(out, bucket_count - 1)


def clamp(value, minimum_value, maximum_value):
    return max(minimum_value, min(maximum_value, value))


def self_air_state(state_name):
    state_name = str(state_name).strip().lower()
    if state_name == "dive":
        return 2
    if state_name == "jump":
        return 1
    return 0


def ball_approach_state(ball_rel_x, ball_vx, small_threshold=24):
    if abs(ball_vx) <= 1 or abs(ball_rel_x) <= small_threshold:
        return 1
    if ball_rel_x * ball_vx < 0:
        return 2
    return 0


def _extract_raw_values(materials):
    raw = materials["raw"]
    self_x = int(raw["self"]["x"])
    self_y = int(raw["self"]["y"])
    self_state = str(raw["self"]["state"])
    ball_x = int(raw["ball"]["x"])
    ball_y = int(raw["ball"]["y"])
    ball_vx = float(raw["ball"]["x_velocity"])
    ball_vy = float(raw["ball"]["y_velocity"])
    landing_x = int(raw["ball"]["expected_landing_x"])

    return {
        "self_x": self_x,
        "self_y": self_y,
        "self_state": self_state,
        "ball_rel_x": ball_x - self_x,
        "ball_rel_y": self_y - ball_y,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "landing_rel_x": landing_x - self_x,
    }


def _calculate_state_key_v1(values):
    rel_x_min = -GROUND_HALF_WIDTH
    rel_x_max = GROUND_HALF_WIDTH
    rel_y_min = -32
    rel_y_max = 256

    ball_rel_x_bucket = bucket(
        clamp(values["ball_rel_x"], rel_x_min, rel_x_max),
        rel_x_min,
        rel_x_max,
        8,
    )
    ball_rel_y_bucket = bucket(
        clamp(values["ball_rel_y"], rel_y_min, rel_y_max),
        rel_y_min,
        rel_y_max,
        8,
    )
    ball_vx_bucket = bucket(clamp(values["ball_vx"], -20, 20), -20, 20, 6)
    ball_vy_bucket = bucket(clamp(values["ball_vy"], -20, 20), -20, 20, 6)
    landing_rel_x_bucket = bucket(
        clamp(values["landing_rel_x"], rel_x_min, rel_x_max),
        rel_x_min,
        rel_x_max,
        8,
    )

    return [
        ball_rel_x_bucket, 
        ball_rel_y_bucket,
        ball_vx_bucket,
        ball_vy_bucket,
        self_air_state(values["self_state"]),
        landing_rel_x_bucket,
        ball_approach_state(values["ball_rel_x"], values["ball_vx"]),
    ]


def _calculate_state_key_v2(values):
    rel_x_min = -GROUND_HALF_WIDTH
    rel_x_max = GROUND_HALF_WIDTH
    rel_y_min = -32
    rel_y_max = 256

    return [
        bucket(clamp(values["ball_rel_x"], rel_x_min, rel_x_max), rel_x_min, rel_x_max, 8),
        bucket(clamp(values["ball_rel_y"], rel_y_min, rel_y_max), rel_y_min, rel_y_max, 8),
        bucket(clamp(values["ball_vx"], -20, 20), -20, 20, 6),
        bucket(clamp(values["ball_vy"], -20, 20), -20, 20, 6),
        self_air_state(values["self_state"]),
        bucket(clamp(values["landing_rel_x"], rel_x_min, rel_x_max), rel_x_min, rel_x_max, 8),
    ]


def _calculate_state_key_v3(values):
    rel_x_min = -GROUND_HALF_WIDTH
    rel_x_max = GROUND_HALF_WIDTH
    rel_y_min = -32
    rel_y_max = 256

    return [
        bucket(values["self_x"], 0, GROUND_WIDTH - 1, 6),
        self_air_state(values["self_state"]),
        bucket(clamp(values["ball_rel_x"], rel_x_min, rel_x_max), rel_x_min, rel_x_max, 8),
        bucket(clamp(values["ball_rel_y"], rel_y_min, rel_y_max), rel_y_min, rel_y_max, 8),
        bucket(clamp(values["ball_vx"], -20, 20), -20, 20, 6),
        bucket(clamp(values["ball_vy"], -20, 20), -20, 20, 6),
        bucket(clamp(values["landing_rel_x"], rel_x_min, rel_x_max), rel_x_min, rel_x_max, 8),
        ball_approach_state(values["ball_rel_x"], values["ball_vx"]),
    ]


def calculate_state_key(materials):
    values = _extract_raw_values(materials)

    if STATE_DESIGN_VERSION == 1:
        return _calculate_state_key_v1(values)
    if STATE_DESIGN_VERSION == 2:
        return _calculate_state_key_v2(values)
    if STATE_DESIGN_VERSION == 3:
        return _calculate_state_key_v3(values)
    raise ValueError(f"unsupported qarsa state design version: {STATE_DESIGN_VERSION}")
