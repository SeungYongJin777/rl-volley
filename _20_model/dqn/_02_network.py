# Import Required External Libraries
from pathlib import Path
import torch
import torch.nn as nn


class DqnNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_dim, hidden_layer_count):
        # - Herit from nn.Module
        super().__init__()

        # - Build the Network Architecture
        layers = []
        input_dim = dim_state
        for _ in range(int(hidden_layer_count)):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, dim_action))

        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        # - Forward Pass through the Network
        action_mat = self.layers(state)
        return action_mat


def create_nn(dim_state, dim_action, hidden_dim, hidden_layer_count):
    """====================================================================================================
    ## Creation of DQN Checkpoint
    ===================================================================================================="""
    q_network = DqnNetwork(dim_state, dim_action,
                           hidden_dim, hidden_layer_count)
    return q_network


def save_nn(model, path):
    """====================================================================================================
    ## Saving DQN Network Checkpoint to a File
    ===================================================================================================="""
    # - Ensure the Directory for the Save Path Exists
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # - Save the Checkpoint using torch
    torch.save(model.state_dict(), save_path)


def load_nn(model, path):
    """====================================================================================================
    ## Loading DQN Network Checkpoint from a File
    ===================================================================================================="""
    # - Load the Checkpoint using torch
    model_device = next(model.parameters()).device
    state_dict = torch.load(
        path,
        map_location=model_device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    # - Return the Loaded Model
    return model
