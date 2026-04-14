def calculate_reward(materials):
    """====================================================================================================
    ## Tournament-Oriented Compact Reward
    ===================================================================================================="""
    point_scored = int(materials["point_result"]["scored"] > 0.5)
    point_lost = int(materials["point_result"]["lost"] > 0.5)

    self_touch = bool(materials.get("self_touch", False))
    crossed_to_opponent = bool(materials.get("crossed_to_opponent", False))
    ball_side = str(materials.get("ball_side", "self"))

    reward = 0.0

    # Sparse core reward
    reward += 1.0 * point_scored
    reward -= 1.0 * point_lost

    # Very weak shaping
    if self_touch and crossed_to_opponent:
        reward += 0.02

    if self_touch and ball_side == "self":
        reward += 0.01

    return reward
