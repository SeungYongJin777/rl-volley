# Import Required Internal Libraries
from _00_environment.constants import GROUND_HALF_WIDTH


def select_mat_for_reward(materials):
    """====================================================================================================
    ## Load materials for reward design
    ===================================================================================================="""
    # Self Position (x 0~431, y 0~252)
    self_position = materials["self_position"]

    # Opponent Position (x 0~431, y 0~252)
    opponent_position = materials["opponent_position"]

    # Ball Position (x 0~431, y 0~252)
    ball_position = materials["ball_position"]

    # Self Action Name (String)
    self_action_name = str(materials["self_action_name"])

    # Opponent Action Name (String)
    opponent_action_name = str(materials["opponent_action_name"])

    # Rally Frames (Float)
    rally_total_frames_until_point = float(
        materials["rally_total_frames_until_point"])

    # Whether Self Scored a Point (0 or 1)
    point_scored = int(materials["point_result"]["scored"])

    # Whether Self Lost a Point (0 or 1)
    point_lost = int(materials["point_result"]["lost"])

    # Whether Self Used a Spike (0 or 1)
    self_spike_used = int(self_action_name.startswith("spike_"))

    # Whether Self Used a Dive (0 or 1)
    self_dive_used = int(self_action_name.startswith("dive_"))

    # Whether Opponent Used a Dive (0 or 1)
    opponent_dive_used = int(opponent_action_name.startswith("dive_"))

    # Whether Opponent Used a Spike (0 or 1)
    opponent_spike_used = int(opponent_action_name.startswith("spike_"))

    # Whether Self Won the Match (0 or 1)
    match_won = int(materials["match_result"]["won"] > 0.5)

    # x-distance from Self to Net Center (0.0~ )
    self_net_distance = abs(self_position[0] - GROUND_HALF_WIDTH)

    # x-distance from Opponent to Net Center (0.0~ )
    opponent_net_distance = abs(opponent_position[0] - GROUND_HALF_WIDTH)

    # Slect materials for reward design
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

    # Return selected materials
    return SELECTED_MATARIALS


def landing_distance_from_state(state_mat):
    raw = state_mat["raw"]
    self_x = int(raw["self"]["x"])
    landing_x = int(raw["ball"]["expected_landing_x"])
    return abs(self_x - landing_x)


def ball_relative_height_from_state(state_mat):
    raw = state_mat["raw"]
    self_y = int(raw["self"]["y"])
    ball_y = int(raw["ball"]["y"])
    return self_y - ball_y


def ball_vertical_velocity_from_state(state_mat):
    raw = state_mat["raw"]
    return float(raw["ball"]["y_velocity"])


def drop_phase_from_state(state_mat, low_height_threshold=56):
    ball_rel_y = ball_relative_height_from_state(state_mat)
    ball_vy = ball_vertical_velocity_from_state(state_mat)

    if ball_vy <= 0:
        return 0
    if ball_rel_y > low_height_threshold:
        return 1
    return 2


def landing_alignment_urgency_scale(state_mat):
    phase = drop_phase_from_state(state_mat)
    if phase == 2:
        return 1.8
    if phase == 1:
        return 0.8
    return 0.0


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
    if ball_vertical_velocity_from_state(next_state_mat) <= 0.0:
        return 0.0

    current_distance = float(landing_distance_from_state(current_state_mat))
    next_distance = float(landing_distance_from_state(next_state_mat))

    if current_distance <= alignment_deadband and next_distance <= alignment_deadband:
        return 0.0

    normalized_delta = (current_distance - next_distance) / float(GROUND_HALF_WIDTH)

    if normalized_delta > 1.0:
        normalized_delta = 1.0
    if normalized_delta < -1.0:
        normalized_delta = -1.0

    return normalized_delta * landing_alignment_urgency_scale(next_state_mat)


def calculate_urgent_receive_penalty(
    next_state_mat=None,
    point_scored=0,
    point_lost=0,
    safe_distance=18.0,
):
    if next_state_mat is None:
        return 0.0
    if point_scored or point_lost:
        return 0.0
    if drop_phase_from_state(next_state_mat) != 2:
        return 0.0

    next_distance = float(landing_distance_from_state(next_state_mat))
    if next_distance <= safe_distance:
        return 0.0

    normalized_gap = (next_distance - safe_distance) / float(GROUND_HALF_WIDTH)
    if normalized_gap > 1.0:
        normalized_gap = 1.0
    return -normalized_gap


def calculate_reward(materials, current_state_mat=None, next_state_mat=None):
    """====================================================================================================
    ## Load Materials For Reward Design
    ===================================================================================================="""
    # Load materials for reward design
    mat = select_mat_for_reward(materials)

    """====================================================================================================
    ## Defining Scale Factors and Calculating Reward
    ===================================================================================================="""
    # Define Scale Factor for Point Score Reward
    SCALE_POINT_SCORE_REWARD = 30.0
    SCALE_POINT_LOST_PENALTY = 35.0

    # Define Scale Factor for Self Bonus/Penalty
    SCALE_SELF_SPIKE_BONUS = 0.0
    SCALE_SELF_DIVE_BONUS = -0.5

    # Define Scale Factor for Opponent Bonus/Penalty
    SCALE_OPPONENT_DIVE_BONUS = 0.0
    SCALE_OPPONENT_SPIKE_PENALTY = 0.0

    # Define Scale Factor for Rally Frame Reward
    SCALE_RALLY_FRAME = 0.0
    SCALE_RALLY_FRAME_MAX = 0.0

    # Define Scale Factor for Match Win Bonus
    SCALE_MATCH_WIN_BONUS = 20.0

    # Define Scale Factor for Moving Toward the Predicted Landing Point
    SCALE_LANDING_ALIGNMENT = 1.5

    # Define Scale Factor for Being Too Far from a Low Descending Ball
    SCALE_URGENT_RECEIVE_PENALTY = 1.5

    """====================================================================================================
    ## Calculating Reward by Accumulating Different Components
    ===================================================================================================="""
    # Initialize Reward at the Certain Transition Step
    reward = 0.0

    # Accumulate Point Score Reward
    reward += SCALE_POINT_SCORE_REWARD * mat["point_scored"]
    reward -= SCALE_POINT_LOST_PENALTY * mat["point_lost"]

    # Accumulate Self Bonus/Penalty Reward
    reward += SCALE_SELF_SPIKE_BONUS * mat["self_spike_used"]
    reward += SCALE_SELF_DIVE_BONUS * mat["self_dive_used"]

    # Accumulate Opponent Bonus/Penalty Reward
    reward += SCALE_OPPONENT_DIVE_BONUS * mat["opponent_dive_used"]
    reward -= SCALE_OPPONENT_SPIKE_PENALTY * mat["opponent_spike_used"]

    # Accumulate Rally Frame Reward
    rally_reward = 0.0
    if mat["point_scored"] > 0.5:
        rally_reward = min(mat["rally_total_frames_until_point"] * SCALE_RALLY_FRAME,
                           SCALE_RALLY_FRAME_MAX)
    reward += rally_reward

    # Accumulate Positioning Reward
    reward += SCALE_LANDING_ALIGNMENT * calculate_landing_alignment_reward(
        current_state_mat=current_state_mat,
        next_state_mat=next_state_mat,
        point_scored=mat["point_scored"],
        point_lost=mat["point_lost"],
    )
    reward += SCALE_URGENT_RECEIVE_PENALTY * calculate_urgent_receive_penalty(
        next_state_mat=next_state_mat,
        point_scored=mat["point_scored"],
        point_lost=mat["point_lost"],
    )

    # Accumulate Match Win Reward
    reward += SCALE_MATCH_WIN_BONUS * mat["match_won"]

    # Return Calculated Reward
    return reward
