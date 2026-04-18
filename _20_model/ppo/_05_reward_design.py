from _00_environment.constants import GROUND_HALF_WIDTH


def clamp(value, minimum_value, maximum_value):
    return max(minimum_value, min(maximum_value, float(value)))


def select_mat_for_reward(materials):
    selected_materials = {
        "self_position": materials["self_position"],
        "ball_position": materials["ball_position"],
        "expected_landing_x": float(materials["expected_landing_x"]),
        "self_action_name": str(materials["self_action_name"]),
        "self_state": str(materials["self_state"]),
        "self_touch": bool(materials["self_touch"]),
        "crossed_to_opponent": bool(materials["crossed_to_opponent"]),
        "point_scored": int(materials["point_result"]["scored"] > 0.5),
        "point_lost": int(materials["point_result"]["lost"] > 0.5),
        "match_won": int(materials["match_result"]["won"] > 0.5),
        "ball_side": str(materials.get("ball_side", "self")),
    }
    return selected_materials


def is_left_side_player(self_position):
    return float(self_position[0]) <= float(GROUND_HALF_WIDTH)


def landing_distance_from_state(state_mat):
    if state_mat is None:
        return 0.0
    raw = state_mat["raw"]
    self_x = float(raw["self"]["x"])
    landing_x = float(raw["ball"]["expected_landing_x"])
    return abs(self_x - landing_x)


def ball_relative_height_from_state(state_mat):
    if state_mat is None:
        return 0.0
    raw = state_mat["raw"]
    self_y = float(raw["self"]["y"])
    ball_y = float(raw["ball"]["y"])
    return self_y - ball_y


def ball_vertical_velocity_from_state(state_mat):
    if state_mat is None:
        return 0.0
    raw = state_mat["raw"]
    return float(raw["ball"]["y_velocity"])


def ball_side_from_state(state_mat):
    if state_mat is None:
        return "self"
    raw = state_mat["raw"]
    return str(raw["ball"].get("side", "self"))


def landing_urgency_phase_from_state(state_mat):
    if state_mat is None:
        return 0

    if ball_side_from_state(state_mat) != "self":
        return 0

    ball_vy = ball_vertical_velocity_from_state(state_mat)
    if ball_vy <= 0.0:
        return 0

    ball_rel_y = ball_relative_height_from_state(state_mat)
    if ball_rel_y > 96.0:
        return 1
    return 2


def hittable_phase_from_state(state_mat):
    if state_mat is None:
        return 0

    raw = state_mat["raw"]
    self_x = float(raw["self"]["x"])
    self_y = float(raw["self"]["y"])
    ball_x = float(raw["ball"]["x"])
    ball_y = float(raw["ball"]["y"])
    ball_vy = float(raw["ball"]["y_velocity"])
    ball_side = str(raw["ball"].get("side", "self"))

    abs_rel_x = abs(ball_x - self_x)
    rel_y = self_y - ball_y
    descending_or_neutral = ball_vy >= -1.0

    if ball_side != "self":
        return 0
    if descending_or_neutral and abs_rel_x <= 24.0 and 20.0 <= rel_y <= 120.0:
        return 2
    if abs_rel_x <= 64.0 and 0.0 <= rel_y <= 164.0:
        return 1
    return 0


def calculate_landing_alignment_reward(
    current_state_mat=None,
    next_state_mat=None,
    point_scored=0,
    point_lost=0,
    alignment_deadband=12.0,
):
    if current_state_mat is None or next_state_mat is None:
        return 0.0
    if point_scored or point_lost:
        return 0.0
    if ball_side_from_state(next_state_mat) != "self":
        return 0.0
    if ball_vertical_velocity_from_state(next_state_mat) <= 0.0:
        return 0.0

    current_distance = float(landing_distance_from_state(current_state_mat))
    next_distance = float(landing_distance_from_state(next_state_mat))
    if current_distance <= alignment_deadband and next_distance <= alignment_deadband:
        return 0.0
    normalized_delta = (current_distance - next_distance) / float(GROUND_HALF_WIDTH)
    return clamp(normalized_delta, -1.0, 1.0)


def calculate_urgent_receive_penalty(next_state_mat=None, point_scored=0, point_lost=0, safe_distance=18.0):
    if next_state_mat is None:
        return 0.0
    if point_scored or point_lost:
        return 0.0
    if landing_urgency_phase_from_state(next_state_mat) != 2:
        return 0.0

    next_distance = float(landing_distance_from_state(next_state_mat))
    if next_distance <= safe_distance:
        return 0.0

    normalized_gap = (next_distance - safe_distance) / float(GROUND_HALF_WIDTH)
    return -clamp(normalized_gap, 0.0, 1.0)


def is_jump_action(action_name):
    normalized_action_name = str(action_name or "")
    return normalized_action_name in ("jump", "jump_forward", "jump_backward")


def action_direction(action_name):
    normalized_action_name = str(action_name or "").strip().lower()
    if normalized_action_name in ("backward", "jump_backward", "dive_backward"):
        return -1
    if normalized_action_name in ("forward", "jump_forward", "dive_forward"):
        return 1
    return 0


def calculate_jump_penalty(materials, next_state_mat=None):
    if next_state_mat is None:
        return 0.0

    action_name = str(materials["self_action_name"])
    if is_jump_action(action_name) is not True:
        return 0.0

    hittable_phase = hittable_phase_from_state(next_state_mat)
    if hittable_phase == 0:
        return -0.6
    if hittable_phase == 1:
        return -0.3
    return 0.0


def calculate_backward_recovery_reward(materials, next_state_mat=None):
    if next_state_mat is None:
        return 0.0
    if landing_urgency_phase_from_state(next_state_mat) != 2:
        return 0.0

    raw = next_state_mat["raw"]
    self_x = float(raw["self"]["x"])
    landing_x = float(raw["ball"]["expected_landing_x"])
    landing_rel_x = landing_x - self_x
    if landing_rel_x >= 0.0:
        return 0.0

    if action_direction(materials["self_action_name"]) < 0:
        return 0.3
    return 0.0


def calculate_wrong_way_penalty(materials, next_state_mat=None):
    if next_state_mat is None:
        return 0.0
    if landing_urgency_phase_from_state(next_state_mat) != 2:
        return 0.0

    raw = next_state_mat["raw"]
    self_x = float(raw["self"]["x"])
    landing_x = float(raw["ball"]["expected_landing_x"])
    landing_rel_x = landing_x - self_x
    if landing_rel_x >= 0.0:
        return 0.0

    if action_direction(materials["self_action_name"]) > 0:
        return -0.3
    return 0.0


def calculate_action_flip_penalty(materials, current_state_mat=None, next_state_mat=None):
    if current_state_mat is None or next_state_mat is None:
        return 0.0
    if landing_urgency_phase_from_state(next_state_mat) == 2:
        return 0.0
    if hittable_phase_from_state(next_state_mat) == 2:
        return 0.0

    previous_action_name = current_state_mat["raw"]["self"].get("action_name", "normal")
    current_action_name = materials["self_action_name"]
    previous_direction = action_direction(previous_action_name)
    current_direction = action_direction(current_action_name)

    if previous_direction == 0 or current_direction == 0:
        return 0.0
    if previous_direction + current_direction != 0:
        return 0.0
    return -0.08


def calculate_reward(materials, current_state_mat=None, next_state_mat=None):
    """====================================================================================================
    ## Dense PPO Reward with Defensive Awareness
    ===================================================================================================="""
    mat = select_mat_for_reward(materials)

    scale_point_score_reward = 25.0
    scale_point_lost_penalty = 25.0
    scale_match_win_bonus = 20.0
    scale_self_touch_reward = 0.5
    scale_cross_net_bonus = 1.0
    scale_deep_attack_bonus = 1.0
    scale_post_hit_recover_bonus = 0.5
    scale_landing_alignment = 0.8
    scale_urgent_receive_penalty = 1.5

    reward = 0.0
    reward += scale_point_score_reward * mat["point_scored"]
    reward -= scale_point_lost_penalty * mat["point_lost"]
    reward += scale_match_win_bonus * mat["match_won"]

    player_is_left = is_left_side_player(mat["self_position"])
    deep_attack_threshold = GROUND_HALF_WIDTH + 72.0
    home_zone_min = 72.0
    home_zone_max = 156.0
    if player_is_left is not True:
        deep_attack_threshold = GROUND_HALF_WIDTH - 72.0
        home_zone_min = GROUND_HALF_WIDTH + 60.0
        home_zone_max = GROUND_HALF_WIDTH + 144.0

    if mat["self_touch"]:
        reward += scale_self_touch_reward

    if mat["self_touch"] and mat["crossed_to_opponent"]:
        reward += scale_cross_net_bonus

    if (
        mat["self_touch"]
        and mat["crossed_to_opponent"]
        and (
            (player_is_left and mat["expected_landing_x"] >= deep_attack_threshold)
            or (player_is_left is not True and mat["expected_landing_x"] <= deep_attack_threshold)
        )
    ):
        reward += scale_deep_attack_bonus

    if (
        mat["self_touch"]
        and mat["crossed_to_opponent"]
        and mat["self_state"] == "normal"
        and home_zone_min <= float(mat["self_position"][0]) <= home_zone_max
    ):
        reward += scale_post_hit_recover_bonus

    reward += scale_landing_alignment * calculate_landing_alignment_reward(
        current_state_mat=current_state_mat,
        next_state_mat=next_state_mat,
        point_scored=mat["point_scored"],
        point_lost=mat["point_lost"],
    )
    reward += scale_urgent_receive_penalty * calculate_urgent_receive_penalty(
        next_state_mat=next_state_mat,
        point_scored=mat["point_scored"],
        point_lost=mat["point_lost"],
    )
    reward += calculate_jump_penalty(
        materials=materials,
        next_state_mat=next_state_mat,
    )
    reward += calculate_backward_recovery_reward(
        materials=materials,
        next_state_mat=next_state_mat,
    )
    reward += calculate_wrong_way_penalty(
        materials=materials,
        next_state_mat=next_state_mat,
    )
    reward += calculate_action_flip_penalty(
        materials=materials,
        current_state_mat=current_state_mat,
        next_state_mat=next_state_mat,
    )

    return reward
