import numpy as np
import torch


def deterministic_action_selection(policy, state, noise_scale):
    """====================================================================================================
    ## Select Action by Deterministic Policy with Additive Noise
    ===================================================================================================="""
    device = next(policy.parameters()).device

    state = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = policy(state).squeeze(0)

    if float(noise_scale) > 0.0:
        logits = logits + float(noise_scale) * torch.randn_like(logits)

    action_idx = int(torch.argmax(logits).item())
    dim_action = int(logits.shape[0])

    # - Convert the Action Index to One-Hot Action Vector
    action = np.zeros(dim_action, dtype=np.float32)
    action[action_idx] = 1.0

    # - Return the Selected Action Vector
    return action


def decay_noise_scale(noise_scale_start, noise_scale_decay, noise_scale_end):
    """====================================================================================================
    ## Decaying Noise Scale for DDPG Exploration
    ===================================================================================================="""
    # - Decay Noise Scale
    next_noise_scale = float(noise_scale_start) * float(noise_scale_decay)

    # - Ensure Noise Scale Does Not Fall Below the Minimum Threshold
    if next_noise_scale < float(noise_scale_end):
        next_noise_scale = float(noise_scale_end)

    # - Return Decayed Noise Scale
    return next_noise_scale
