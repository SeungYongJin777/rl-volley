from pathlib import Path
import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_dim, hidden_layer_count):
        super().__init__()

        layers = []
        input_dim = dim_state
        for _ in range(int(hidden_layer_count)):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, dim_action))
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)


class CriticNetwork(nn.Module):
    def __init__(self, dim_state, hidden_dim, hidden_layer_count):
        super().__init__()

        layers = []
        input_dim = dim_state
        for _ in range(int(hidden_layer_count)):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)


def create_actor_nn(dim_state, dim_action, hidden_dim, hidden_layer_count):
    model = ActorNetwork(dim_state, dim_action, hidden_dim, hidden_layer_count)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)

    return model


def create_critic_nn(dim_state, hidden_dim, hidden_layer_count):
    model = CriticNetwork(dim_state, hidden_dim, hidden_layer_count)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)

    return model


def save_nn(model, path):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_nn(model, path):
    model_device = next(model.parameters()).device
    state_dict = torch.load(
        path,
        map_location=model_device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model