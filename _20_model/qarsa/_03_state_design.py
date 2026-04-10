# Import require internal Packages
from _00_environment.constants import GROUND_HALF_WIDTH


STATE_DESIGN_VERSION = 1


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


def relative_y_bucket(value):
    boundaries = (-16, 12, 40, 88, 144)
    return bucket_from_boundaries(value, boundaries)


def velocity_x_bucket(value):
    boundaries = (-12, -4, 4, 12)
    return bucket_from_boundaries(value, boundaries)


def velocity_y_bucket(value):
    boundaries = (-12, -2, 2, 10)
    return bucket_from_boundaries(value, boundaries)


def self_zone_bucket(self_x):
    if self_x < 72:
        return 0
    if self_x < 144:
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
    return [
        self_zone_bucket(values["self_x"]),
        self_air_state(values["self_state"]),
        relative_x_bucket(
            clamp(values["ball_rel_x"], -GROUND_HALF_WIDTH, GROUND_HALF_WIDTH)
        ),
        relative_y_bucket(clamp(values["ball_rel_y"], -32, 256)),
        velocity_x_bucket(clamp(values["ball_vx"], -20, 20)),
        velocity_y_bucket(clamp(values["ball_vy"], -20, 20)),
        relative_x_bucket(
            clamp(values["landing_rel_x"], -GROUND_HALF_WIDTH, GROUND_HALF_WIDTH)
        ),
    ]


def _calculate_state_key_v2(values):
    return _calculate_state_key_v1(values)


def _calculate_state_key_v3(values):
    return _calculate_state_key_v1(values)


def calculate_state_key(materials):
    values = _extract_raw_values(materials)

    if STATE_DESIGN_VERSION == 1:
        return _calculate_state_key_v1(values)
    if STATE_DESIGN_VERSION == 2:
        return _calculate_state_key_v2(values)
    if STATE_DESIGN_VERSION == 3:
        return _calculate_state_key_v3(values)
    raise ValueError(f"unsupported qarsa state design version: {STATE_DESIGN_VERSION}")
