import numpy as np

from _00_environment.actions import ACTION_NAMES


INTERNAL_ACTION_NAMES = tuple(ACTION_NAMES) + ("idle",)
IDLE_ACTION_INDEX = len(INTERNAL_ACTION_NAMES) - 1

STATE_INDEX_SELF_TO_LANDING = 3
STATE_INDEX_BALL_HEIGHT = 4
STATE_INDEX_BALL_VY_PHASE = 5
STATE_INDEX_SELF_STATE = 7

SELF_STATE_NORMAL = 0
SELF_STATE_JUMP = 1
SELF_STATE_DIVE_OR_END = 2


def internal_action_names():
    return INTERNAL_ACTION_NAMES


def action_mask(state=None):
    mask = np.ones(len(INTERNAL_ACTION_NAMES), dtype=np.float32)

    if state is None:
        return mask

    try:
        self_state = int(state[STATE_INDEX_SELF_STATE])
    except Exception:
        return mask

    if self_state == SELF_STATE_NORMAL:
        mask[7:13] = 0.0
    elif self_state == SELF_STATE_JUMP:
        mask[2:7] = 0.0
    else:
        mask[:] = 0.0
        mask[IDLE_ACTION_INDEX] = 1.0

    return mask


def legal_action_indexes(state=None):
    return np.flatnonzero(action_mask(state) > 0.0)


def map_internal_to_environment_action(action_source):
    vector = np.asarray(action_source, dtype=np.float32)
    if vector.ndim == 0:
        action_index = int(vector.item())
    else:
        action_index = int(np.argmax(vector))

    env_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    if action_index < len(ACTION_NAMES):
        env_action[action_index] = 1.0
    return env_action


def heuristic_q_prior(state):
    prior = np.zeros(len(INTERNAL_ACTION_NAMES), dtype=np.float32)
    if state is None:
        prior[IDLE_ACTION_INDEX] = 0.01
        return prior

    try:
        ball_side = int(state[0])
        self_x_zone = int(state[1])
        self_to_landing = int(state[STATE_INDEX_SELF_TO_LANDING])
        ball_height = int(state[STATE_INDEX_BALL_HEIGHT])
        ball_vy_phase = int(state[STATE_INDEX_BALL_VY_PHASE])
        self_state = int(state[STATE_INDEX_SELF_STATE])
    except Exception:
        prior[IDLE_ACTION_INDEX] = 0.01
        return prior

    if self_state == SELF_STATE_JUMP:
        prior[10] += 0.22
        prior[11] += 0.30
        prior[12] += 0.18
        prior[0] += 0.05 if self_to_landing < 2 else 0.0
        prior[1] += 0.05 if self_to_landing > 2 else 0.0
    elif self_state == SELF_STATE_NORMAL and ball_side == 0:
        if self_to_landing < 2:
            prior[0] += 0.30
        elif self_to_landing > 2:
            prior[1] += 0.30
        else:
            prior[2] += 0.18
            prior[IDLE_ACTION_INDEX] += 0.05

        if ball_height == 2 and ball_vy_phase == 1:
            if self_to_landing == 0:
                prior[5] += 0.20
            elif self_to_landing == 4:
                prior[6] += 0.20
            elif self_to_landing == 2:
                prior[2] += 0.25
    elif self_state == SELF_STATE_NORMAL:
        if self_x_zone <= 0:
            prior[0] += 0.16
        elif self_x_zone >= 3:
            prior[1] += 0.16
        else:
            prior[IDLE_ACTION_INDEX] += 0.08
    else:
        prior[IDLE_ACTION_INDEX] += 0.20

    return prior * action_mask(state)
