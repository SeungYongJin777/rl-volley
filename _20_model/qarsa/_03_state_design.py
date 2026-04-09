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


def bucket_from_boundaries(value, boundaries):
    for bucket_index, boundary in enumerate(boundaries):
        if value <= boundary:
            return bucket_index
    return len(boundaries)


def self_air_state(state_name):
    state_name = str(state_name).strip().lower()
    if state_name == "dive":
        return 2
    if state_name == "jump":
        return 1
    return 0


def relative_x_bucket(value):
    boundaries = (-160, -96, -48, -16, 16, 48, 96, 160)
    return bucket_from_boundaries(value, boundaries)


def ball_drop_phase(ball_rel_y, ball_vy, low_height_threshold=56):
    if ball_vy <= 0:
        return 0
    if ball_rel_y > low_height_threshold:
        return 1
    return 2


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
    rel_y_min = -32
    rel_y_max = 256

    ball_rel_x_bucket = relative_x_bucket(
        clamp(values["ball_rel_x"], -GROUND_HALF_WIDTH, GROUND_HALF_WIDTH)
    )
    ball_rel_y_bucket = bucket(
        clamp(values["ball_rel_y"], rel_y_min, rel_y_max),
        rel_y_min,
        rel_y_max,
        8,
    )
    ball_vx_bucket = bucket(clamp(values["ball_vx"], -20, 20), -20, 20, 6)
    ball_vy_bucket = bucket(clamp(values["ball_vy"], -20, 20), -20, 20, 6)
    landing_rel_x_bucket = relative_x_bucket(
        clamp(values["landing_rel_x"], -GROUND_HALF_WIDTH, GROUND_HALF_WIDTH)
    )

    return [
        ball_rel_x_bucket, 
        ball_rel_y_bucket,
        ball_vx_bucket,
        ball_vy_bucket,
        self_air_state(values["self_state"]),
        landing_rel_x_bucket,
        ball_drop_phase(values["ball_rel_y"], values["ball_vy"]),
    ]


def _calculate_state_key_v2(values):
    rel_y_min = -32
    rel_y_max = 256

    return [
        relative_x_bucket(clamp(values["ball_rel_x"], -GROUND_HALF_WIDTH, GROUND_HALF_WIDTH)),
        bucket(clamp(values["ball_rel_y"], rel_y_min, rel_y_max), rel_y_min, rel_y_max, 8),
        bucket(clamp(values["ball_vx"], -20, 20), -20, 20, 6),
        bucket(clamp(values["ball_vy"], -20, 20), -20, 20, 6),
        self_air_state(values["self_state"]),
        relative_x_bucket(clamp(values["landing_rel_x"], -GROUND_HALF_WIDTH, GROUND_HALF_WIDTH)),
    ]


def _calculate_state_key_v3(values):
    rel_y_min = -32
    rel_y_max = 256

    return [
        bucket(values["self_x"], 0, GROUND_WIDTH - 1, 6),
        self_air_state(values["self_state"]),
        relative_x_bucket(clamp(values["ball_rel_x"], -GROUND_HALF_WIDTH, GROUND_HALF_WIDTH)),
        bucket(clamp(values["ball_rel_y"], rel_y_min, rel_y_max), rel_y_min, rel_y_max, 8),
        bucket(clamp(values["ball_vx"], -20, 20), -20, 20, 6),
        bucket(clamp(values["ball_vy"], -20, 20), -20, 20, 6),
        relative_x_bucket(clamp(values["landing_rel_x"], -GROUND_HALF_WIDTH, GROUND_HALF_WIDTH)),
        ball_drop_phase(values["ball_rel_y"], values["ball_vy"]),
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
