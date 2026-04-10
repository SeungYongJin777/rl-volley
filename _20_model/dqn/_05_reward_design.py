# Import Required Internal Libraries
from _00_environment.constants import GROUND_HALF_WIDTH


def select_mat_for_reward(materials):
    """====================================================================================================
    ## Load materials for reward design
    ===================================================================================================="""
    selected_materials = {
        "self_position": materials["self_position"],
        "ball_position": materials["ball_position"],
        "expected_landing_x": float(materials["expected_landing_x"]),
        "self_action_name": str(materials["self_action_name"]),
        "self_state": str(materials["self_state"]),
        "self_touch": bool(materials["self_touch"]),
        "crossed_to_opponent": bool(materials["crossed_to_opponent"]),
        "point_scored": int(materials["point_result"]["scored"]),
        "point_lost": int(materials["point_result"]["lost"]),
        "match_won": int(materials["match_result"]["won"] > 0.5),
    }

    # Return selected materials
    return selected_materials


def calculate_reward(materials):
    """====================================================================================================
    ## Load Materials For Reward Design
    ===================================================================================================="""
    # Load materials for reward design
    mat = select_mat_for_reward(materials)

    """====================================================================================================
    ## Defining Scale Factors and Calculating Reward
    ===================================================================================================="""
    # Define Scale Factor for Point Score Reward
    SCALE_POINT_SCORE_REWARD = 25.0
    SCALE_POINT_LOST_PENALTY = 25.0

    # Define Scale Factor for Match Win Bonus
    SCALE_MATCH_WIN_BONUS = 30.0

    # Define Scale Factors for Compact Dense Reward
    SCALE_SELF_TOUCH_REWARD = 1.0
    SCALE_DEEP_ATTACK_REWARD = 2.0
    SCALE_POST_HIT_RECOVER_REWARD = 0.5
    SCALE_BAD_LOSS_PENALTY = 2.0

    """====================================================================================================
    ## Calculating Reward by Accumulating Different Components
    ===================================================================================================="""
    # Initialize Reward at the Certain Transition Step
    reward = 0.0

    # Accumulate Point Score Reward
    reward += SCALE_POINT_SCORE_REWARD * mat["point_scored"]
    reward -= SCALE_POINT_LOST_PENALTY * mat["point_lost"]

    # Reward successful touches that keep the rally actionable.
    if mat["self_touch"]:
        reward += SCALE_SELF_TOUCH_REWARD

    # Reward touches that convert into deep opponent-court attacks.
    if (
        mat["self_touch"]
        and mat["crossed_to_opponent"]
        and mat["expected_landing_x"] >= GROUND_HALF_WIDTH + 72
    ):
        reward += SCALE_DEEP_ATTACK_REWARD

    # Reward quick recovery into a stable middle-home position after an attack.
    if (
        mat["self_touch"]
        and mat["crossed_to_opponent"]
        and mat["self_state"] == "normal"
        and 72 <= float(mat["self_position"][0]) <= 156
    ):
        reward += SCALE_POST_HIT_RECOVER_REWARD

    if mat["point_lost"] and mat["self_action_name"].startswith("dive_"):
        reward -= SCALE_BAD_LOSS_PENALTY

    # Accumulate Match Win Reward
    reward += SCALE_MATCH_WIN_BONUS * mat["match_won"]

    # Return Calculated Reward
    return reward
