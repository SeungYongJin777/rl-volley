import numpy as np

from _20_model import raichu_qlearning


def _masked_scores(q_vector, state):
    mask = raichu_qlearning._04_action_space_design.action_mask(state)
    scores = np.asarray(q_vector, dtype=np.float32).copy()
    scores[mask <= 0.0] = -np.inf
    return scores


def epsilon_greedy_action_selection(policy, state, epsilon):
    state = tuple(state)
    q_vector = np.asarray(
        raichu_qlearning._02_qtable.get_qvector(policy, state),
        dtype=np.float32,
    )
    legal_indexes = raichu_qlearning._04_action_space_design.legal_action_indexes(state)

    if legal_indexes.size == 0:
        action_idx = raichu_qlearning._04_action_space_design.IDLE_ACTION_INDEX
    elif np.random.rand() < float(epsilon):
        action_idx = int(np.random.choice(legal_indexes))
    else:
        scores = _masked_scores(q_vector, state)
        if np.allclose(q_vector[legal_indexes], 0.0):
            scores = scores + raichu_qlearning._04_action_space_design.heuristic_q_prior(state)

        best_value = float(np.max(scores))
        candidate_indexes = np.flatnonzero(scores == best_value)
        action_idx = int(np.random.choice(candidate_indexes))

    action = np.zeros_like(q_vector)
    action[action_idx] = 1.0
    return action


def decay_epsilon(epsilon_start, epsilon_decay, epsilon_end):
    next_epsilon = float(epsilon_start) * float(epsilon_decay)
    if next_epsilon < float(epsilon_end):
        next_epsilon = float(epsilon_end)
    return next_epsilon


def calculate_qtarget(policy, reward, state_next, gamma, done):
    if done is True:
        return float(reward)

    state_next = tuple(state_next)
    qvector_next = np.asarray(
        raichu_qlearning._02_qtable.get_qvector(policy, state_next),
        dtype=np.float32,
    )
    scores_next = _masked_scores(qvector_next, state_next)
    best_next = float(np.max(scores_next))
    if not np.isfinite(best_next):
        best_next = 0.0

    return float(reward) + float(gamma) * best_next
