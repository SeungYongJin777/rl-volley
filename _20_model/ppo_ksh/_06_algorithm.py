import numpy as np
import torch


def stochastic_action_selection(policy, state):
    device = next(policy.parameters()).device

    state = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = policy(state).squeeze(0)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("warning: invalid logits detected, fallback to zeros")
            logits = torch.zeros_like(logits)

        action_probs = torch.softmax(logits, dim=0)

        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            print("warning: invalid action_probs detected, fallback to uniform")
            action_probs = torch.ones_like(action_probs) / action_probs.numel()

        action_probs = torch.clamp(action_probs, min=1e-8)
        action_probs = action_probs / action_probs.sum()

        action_idx = int(torch.multinomial(action_probs, num_samples=1).item())
        selected_action_prob = action_probs[action_idx]
        selected_action_prob = torch.clamp(selected_action_prob, min=1e-8)
        selected_log_prob = float(torch.log(selected_action_prob).item())

    dim_action = int(logits.shape[0])
    action = np.zeros(dim_action, dtype=np.float32)
    action[action_idx] = 1.0
    return action, action_idx, selected_log_prob