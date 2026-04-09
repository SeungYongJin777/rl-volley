# Import Required Internal Libraries
from _00_environment.constants import GROUND_HALF_WIDTH


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


def select_mat_for_reward(materials):
    """====================================================================================================
    ## Load materials for reward design
    ===================================================================================================="""
    self_position = materials["self_position"]
    opponent_position = materials["opponent_position"]
    ball_position = materials["ball_position"]
    self_action_name = str(materials["self_action_name"])
    opponent_action_name = str(materials["opponent_action_name"])
    rally_total_frames_until_point = float(
        materials["rally_total_frames_until_point"])
    point_scored = int(materials["point_result"]["scored"])
    point_lost = int(materials["point_result"]["lost"])
    self_spike_used = int(self_action_name.startswith("spike_"))
    self_dive_used = int(self_action_name.startswith("dive_"))
    opponent_dive_used = int(opponent_action_name.startswith("dive_"))
    opponent_spike_used = int(opponent_action_name.startswith("spike_"))
    match_won = int(materials["match_result"]["won"] > 0.5)
    self_net_distance = abs(self_position[0] - GROUND_HALF_WIDTH)
    opponent_net_distance = abs(opponent_position[0] - GROUND_HALF_WIDTH)

    SELECTED_MATARIALS = {
        "self_position": self_position,
        "opponent_position": opponent_position,
        "ball_position": ball_position,
        "self_action_name": self_action_name,
        "opponent_action_name": opponent_action_name,
        "self_net_distance": self_net_distance,
        "opponent_net_distance": opponent_net_distance,
        "point_scored": point_scored,
        "point_lost": point_lost,
        "self_spike_used": self_spike_used,
        "self_dive_used": self_dive_used,
        "opponent_dive_used": opponent_dive_used,
        "opponent_spike_used": opponent_spike_used,
        "match_won": match_won,
        "rally_total_frames_until_point": rally_total_frames_until_point,
    }
    return SELECTED_MATARIALS


def calculate_reward(materials):
    """====================================================================================================
    ## Load Materials For Reward Design
    ===================================================================================================="""
    mat = select_mat_for_reward(materials)

    SCALE_POINT_SCORE_REWARD = 25.0
    SCALE_POINT_LOST_PENALTY = 25.0
    SCALE_SELF_SPIKE_BONUS = 0.
    SCALE_SELF_DIVE_BONUS = 0.
    SCALE_OPPONENT_DIVE_BONUS = 0.
    SCALE_OPPONENT_SPIKE_PENALTY = 0.
    SCALE_RALLY_FRAME = 0.
    SCALE_RALLY_FRAME_MAX = 0.
    SCALE_MATCH_WIN_BONUS = 30.0

    reward = 0.0
    reward += SCALE_POINT_SCORE_REWARD * mat["point_scored"]
    reward -= SCALE_POINT_LOST_PENALTY * mat["point_lost"]
    reward += SCALE_SELF_SPIKE_BONUS * mat["self_spike_used"]
    reward += SCALE_SELF_DIVE_BONUS * mat["self_dive_used"]
    reward += SCALE_OPPONENT_DIVE_BONUS * mat["opponent_dive_used"]
    reward -= SCALE_OPPONENT_SPIKE_PENALTY * mat["opponent_spike_used"]

    rally_reward = 0.0
    if mat["point_scored"] > 0.5:
        rally_reward = min(
            mat["rally_total_frames_until_point"] * SCALE_RALLY_FRAME,
            SCALE_RALLY_FRAME_MAX,
        )
    reward += rally_reward

    reward += SCALE_MATCH_WIN_BONUS * mat["match_won"]
    return reward
