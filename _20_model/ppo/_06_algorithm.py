import numpy as np
import torch


def stochastic_action_selection(policy, state):
    """====================================================================================================
    ## Select Action by Stochastic Policy
    ===================================================================================================="""
    device = next(policy.parameters()).device

    state = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = policy(state).squeeze(0)
        action_probs = torch.softmax(logits, dim=0)
        action_idx = int(torch.multinomial(action_probs, num_samples=1).item())
        selected_action_prob = action_probs[action_idx]
        selected_action_prob = torch.clamp(selected_action_prob, min=1e-8)
        selected_log_prob = float(torch.log(selected_action_prob).item())

    dim_action = int(logits.shape[0])
    action = np.zeros(dim_action, dtype=np.float32)
    action[action_idx] = 1.0
    return action, action_idx, selected_log_prob
