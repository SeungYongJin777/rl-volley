import numpy as np
import torch


def stochastic_action_selection(policy, state, epsilon):
    """====================================================================================================
    ## Select Action by Epsilon-Greedy Strategy
    ===================================================================================================="""
    device = next(policy.parameters()).device

    state = torch.as_tensor(state,
                            dtype=torch.float32,
                            device=device).unsqueeze(0)

    if torch.rand(1, device=device).item() > float(epsilon):
        with torch.no_grad():
            logits = policy(state).squeeze(0)
            action_probs = torch.softmax(logits, dim=0).cpu().numpy()
        dim_action = len(action_probs)
        action_idx = np.random.choice(dim_action, p=action_probs)
    else:
        dim_action = int(policy(state).shape[-1])
        action_idx = np.random.choice(dim_action)

    # - Convert the Action Index to One-Hot Action Vector
    action = np.zeros(dim_action, dtype=np.float32)
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
