import numpy as np

# Import Required Internal Libraries
from _20_model import qarsa


def epsilon_greedy_action_selection(policy, state, epsilon):
    """====================================================================================================
    ## Select Action by Epsilon-Greedy Strategy
    ===================================================================================================="""
    # - Load Q-Vector from Policy
    q_vector = np.asarray(
        qarsa._02_qtable.get_qvector(policy, state), dtype=float)

    # - If Random Value is Less than Epsilon, Select a Random Action
    if np.random.rand() < float(epsilon):
        action_idx = int(np.random.choice(range(len(q_vector)), 1)[0])

    # - Otherwise, Select an Action with the Highest Q-Value
    else:
        max_value = float(np.max(q_vector))
        candidate_indexes = np.flatnonzero(q_vector == max_value)
        action_idx = int(np.random.choice(candidate_indexes, 1)[0])

    # - Convert the Action Index to One-Hot Action Vector
    action = np.zeros_like(q_vector)
    action[action_idx] = 1.0

    # - Return the Selected Action Vector
    return action


def decay_epsilon(epsilon_start, epsilon_decay, epsilon_end):
    """====================================================================================================
    ## Decaying Epsilon for Q-Learning Algorithm
    ===================================================================================================="""
    # - Decay Epsilon
    next_epsilon = float(epsilon_start) * float(epsilon_decay)

    # - Ensure Epsilon Does Not Fall Below the Minimum Threshold
    if next_epsilon < float(epsilon_end):
        next_epsilon = float(epsilon_end)

    # - Return Decayed Epsilon
    return next_epsilon


def calculate_qtarget(policy, reward, state_next, action_next, gamma, done, lambda_blend):
    """====================================================================================================
    ## Calculate TD Target for QARSA
    ===================================================================================================="""
    # - Return Immediate Reward if Episode is Done
    if done is True or action_next is None:
        return reward

    # - Load Next-State Q-Vector
    qvector_next = np.asarray(qarsa._02_qtable.get_qvector(
        policy, state_next), dtype=float)
    action_next_idx = int(np.argmax(np.asarray(action_next, dtype=float)))
    q_on_policy = float(qvector_next[action_next_idx])
    q_off_policy = float(np.max(qvector_next))
    blended_next_q = (
        float(lambda_blend) * q_off_policy
        + (1.0 - float(lambda_blend)) * q_on_policy
    )

    # - Return the TD Target
    TD_target = reward + gamma * blended_next_q

    # - Return the TD Target
    return TD_target
