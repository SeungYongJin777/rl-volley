# Import Required External Libraries
import numpy as np


def action_mask():
    """====================================================================================================
    ## Defining Action Mask for Action Space Design
    ===================================================================================================="""
    ACTION_SPACE_MASK = np.array(
        [
            1,  # forward
            1,  # backward

            1,  # jump
            1,  # jump_forward
            1,  # jump_backward

            1,  # dive_forward
            1,  # dive_backward

            1,  # spike_soft_up
            1,  # spike_soft_flat
            1,  # spike_soft_down
            1,  # spike_fast_up
            1,  # spike_fast_flat
            1,  # spike_fast_down
        ]
    )

    return ACTION_SPACE_MASK
