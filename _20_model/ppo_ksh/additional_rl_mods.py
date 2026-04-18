"""
additional_rl_mods.py
=====================

This module contains example implementations of several of the advanced
reinforcement learning techniques discussed previously.  These examples are
intended to illustrate how you might extend a basic PPO implementation to
include alternate policy distributions (e.g., Beta distribution for bounded
continuous actions), recurrent policy networks (e.g., GRU-based policies for
partial observability), shaped rewards, and simple off-policy sample reuse.

The code here is **illustrative** and does not constitute a full RL agent on its
own.  You would need to integrate these components into your existing
training loop and environment interfaces.

Note: These examples use PyTorch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical


class BetaActor(nn.Module):
    """
    Actor network using a Beta distribution for bounded continuous actions.

    This actor outputs `alpha` and `beta` parameters (>0) for each action
    dimension.  Actions are sampled in [0, 1] and can be rescaled to the
    appropriate range in your environment wrapper.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output two parameters per action dimension
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use softplus to ensure positivity of alpha/beta parameters
        alpha = F.softplus(self.alpha_head(x)) + 1e-5
        beta = F.softplus(self.beta_head(x)) + 1e-5
        return alpha, beta

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a bounded continuous action and compute its log probability.

        Returns:
            action: Tensor of shape (batch, action_dim) in (0, 1)
            log_prob: log probability of the sampled action
            dist_entropy: entropy of the Beta distribution
        """
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class GRUActor(nn.Module):
    """
    Recurrent actor using a GRU to handle partial observability.

    The GRU processes a sequence of states and outputs a probability
    distribution over discrete actions at each timestep.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_seq: torch.Tensor, hidden: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_seq: Tensor of shape (batch, seq_len, state_dim)
            hidden: initial hidden state of shape (num_layers, batch, hidden_dim)
        Returns:
            logits: Tensor of shape (batch, seq_len, action_dim)
            hidden: final hidden state
        """
        outputs, hidden = self.gru(state_seq, hidden)
        logits = self.fc(outputs)
        return logits, hidden

    def sample(self, state_seq: torch.Tensor, hidden: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the GRU policy at the last timestep.

        Returns:
            action_index: sampled action indices (batch,)
            log_prob: log probability of the sampled action (batch,)
            entropy: entropy of the categorical distribution (batch,)
            hidden: updated hidden state
        """
        logits, hidden = self.forward(state_seq, hidden)
        # Take the last timestep
        last_logits = logits[:, -1, :]
        dist = Categorical(logits=last_logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        return action_idx, log_prob, entropy, hidden


def shaped_reward(base_reward: float, materials: dict) -> float:
    """
    Example reward shaping function that adds a small positive reward when the
    agent hits the ball and a small negative reward when the agent fails to
    react.  You would adapt this to your environment's observation structure.

    Args:
        base_reward: reward from the original reward function
        materials: dictionary containing environment-specific information such as
                   whether the agent hit the ball, made a dive, etc.
    Returns:
        float: shaped reward
    """
    # Copy the base reward
    reward = base_reward
    # Example shaping: encourage hitting the ball
    if materials.get("self_hit_ball", False):
        reward += 0.1  # small bonus for making contact
    # Example shaping: penalize doing nothing when the ball is near
    if materials.get("missed_ball", False):
        reward -= 0.05  # small penalty for missing
    return reward


@dataclass
class ReplayBuffer:
    """
    A simple replay buffer for off-policy sample reuse.  Stores transitions and
    allows sampling mini-batches for training.  This can be integrated into
    PPO variants like GePPO or PTR‑PPO where past experiences are reused.
    """
    capacity: int
    buffer: Deque[Tuple] = field(default_factory=lambda: Deque(maxlen=1_000))

    def __post_init__(self):
        self.buffer = Deque(maxlen=self.capacity)

    def add(self, transition: Tuple):
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a random mini-batch of transitions."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across random, numpy and torch.

    Args:
        seed: The random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def example_training_loop(env, actor, critic, optimizer_actor, optimizer_critic, n_steps: int = 2048, n_epochs: int = 4):
    """
    Skeleton of a PPO-like training loop illustrating the use of the above
    components.  This is simplified and omits details such as GAE, clipping,
    entropy bonus, etc.  You would need to integrate with your environment and
    extend it to match your existing implementation.

    Args:
        env: gym-like environment with reset() and step() methods
        actor: policy network (e.g., BetaActor, GRUActor)
        critic: value network
        optimizer_actor: optimizer for the actor
        optimizer_critic: optimizer for the critic
        n_steps: number of environment steps to collect per update
        n_epochs: number of optimization epochs per update
    """
    state = env.reset()
    rollout = []
    for step in range(n_steps):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Example: sample action from a Beta actor
        if isinstance(actor, BetaActor):
            action, log_prob, entropy = actor.sample(state_t)
            # Rescale action from (0,1) to env action space if necessary
            env_action = action.squeeze(0).cpu().numpy()
        # Example: sample action from a GRU actor (stateless in this demo)
        elif isinstance(actor, GRUActor):
            state_seq = state_t.unsqueeze(1)  # fake sequence length 1
            action_idx, log_prob, entropy, _ = actor.sample(state_seq)
            env_action = action_idx.item()
        else:
            raise TypeError("Unsupported actor type")
        next_state, reward, done, info = env.step(env_action)
        rollout.append((state, env_action, reward, done, log_prob.item()))
        state = next_state
        if done:
            state = env.reset()
    # The collected rollout can now be used for PPO updates
    # ... (compute returns, advantages, and perform gradient updates)


if __name__ == "__main__":
    # This block demonstrates how the module could be tested.  It is not
    # executed automatically when importing the module.
    pass