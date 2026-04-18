from _00_environment.constants import GROUND_HALF_WIDTH
from _00_environment.constants import PLAYER_HALF_LENGTH


def _clip(value, minimum_value, maximum_value):
    return max(minimum_value, min(maximum_value, value))


def _raw_from(materials, key):
    raw = materials.get(key)
    if raw is None:
        return None
    if "raw" in raw:
        return raw["raw"]
    return raw


def _landing_distance(raw_state):
    if raw_state is None:
        return None
    self_x = float(raw_state["self"]["x"])
    landing_x = float(raw_state["ball"]["expected_landing_x"])
    return abs(self_x - landing_x)


def _landing_is_self_court(raw_state):
    if raw_state is None:
        return False
    landing_x = float(raw_state["ball"]["expected_landing_x"])
    return landing_x <= GROUND_HALF_WIDTH


def calculate_reward(materials):
    point_scored = float(materials["point_result"]["scored"])
    point_lost = float(materials["point_result"]["lost"])
    match_won = float(materials["match_result"]["won"] > 0.5)
    action_name = str(materials["self_action_name"])

    reward = 30.0 * point_scored
    reward -= 35.0 * point_lost
    reward += 50.0 * match_won

    previous_raw = _raw_from(materials, "previous_raw")
    next_raw = _raw_from(materials, "next_raw")

    if previous_raw is not None and next_raw is not None:
        prev_distance = _landing_distance(previous_raw)
        next_distance = _landing_distance(next_raw)
        ball_side = str(next_raw["ball"]["side"])
        landing_self_court = _landing_is_self_court(next_raw)

        if prev_distance is not None and next_distance is not None:
            if ball_side == "self" or landing_self_court:
                improvement = _clip(
                    (prev_distance - next_distance) / GROUND_HALF_WIDTH,
                    -1.0,
                    1.0,
                )
                reward += 1.2 * improvement

                if next_distance <= PLAYER_HALF_LENGTH:
                    reward += 0.25

                ball_y = float(next_raw["ball"]["y"])
                ball_vy = float(next_raw["ball"]["y_velocity"])
                if ball_y > 170 and ball_vy > 0 and next_distance > PLAYER_HALF_LENGTH * 1.5:
                    reward -= 0.35
            else:
                self_x = float(next_raw["self"]["x"])
                ready_x = GROUND_HALF_WIDTH * 0.62
                ready_score = 1.0 - _clip(abs(self_x - ready_x) / ready_x, 0.0, 1.0)
                reward += 0.12 * ready_score

        if action_name.startswith("spike_"):
            if str(next_raw["ball"]["side"]) == "opponent":
                reward += 0.50
            reward += 0.10

    if action_name.startswith("dive_") and point_lost < 0.5:
        reward -= 0.08
    if action_name.startswith("jump") and point_lost < 0.5:
        reward -= 0.02

    return reward
