# Import Required External Libraries
import numpy as np


POLICY_ACTION_COUNT = 14
PSEUDO_IDLE_INDEX = 13


def action_mask():
    """====================================================================================================
    ## Defining Action Mask for Action Space Design
    ===================================================================================================="""
    return np.ones(POLICY_ACTION_COUNT, dtype=float)


def _normalized_state_name(state_mat):
    return str(state_mat["raw"]["self"]["state"]).strip().lower()


def _landing_relative_x(state_mat):
    raw = state_mat["raw"]
    return int(raw["ball"]["expected_landing_x"]) - int(raw["self"]["x"])


def _is_descending(state_mat):
    return float(state_mat["raw"]["ball"]["y_velocity"]) > 0.0


def action_mask_for_state(state_mat):
    mask = action_mask()
    if state_mat is None:
        return mask

    state_name = _normalized_state_name(state_mat)
    grounded = state_name == "normal"
    airborne = state_name in ("jump", "dive")
    descending = _is_descending(state_mat)
    landing_gap = abs(_landing_relative_x(state_mat))

    if grounded:
        mask[7:13] = 0.0

    if airborne:
        mask[5] = 0.0
        mask[6] = 0.0

    if not (grounded and descending and landing_gap > 18):
        mask[5] = 0.0
        mask[6] = 0.0

    return mask


def policy_action_to_environment(action_source=None):
    vector = np.zeros(13, dtype=float)
    if action_source is None:
        return vector

    source = np.asarray(action_source, dtype=float).reshape(-1)
    limit = min(13, source.shape[0])
    if limit > 0:
        vector[:limit] = source[:limit]
    return vector
