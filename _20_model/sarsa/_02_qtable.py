# Import Required External Libraries
import pickle
import numpy as np
from pathlib import Path

# Import Required Internal Libraries
from _00_environment.actions import ACTION_NAMES
from _20_model import sarsa


def create_qtable():
    """====================================================================================================
    ## Creation of Q-Table
    ===================================================================================================="""
    # - Return Empty Q-Table as a Dictionary
    return {}


def create_qvector(dim_action):
    """====================================================================================================
    ## Creation of Q-Vector for Each State
    ===================================================================================================="""
    # - Return the Created Q-Vector
    qvector = np.zeros(dim_action, dtype=np.float32)
    return qvector


def get_qvector(qtable, state_key):
    """====================================================================================================
    ## Getting Q-Vector for a Given State from Q-Table
    ===================================================================================================="""
    # - If the State Key is Not in Q-Table, Create a New Q-Vector for that State filled with 0.0 for Each Action
    dim_action = len(sarsa._04_action_space_design.action_mask())
    if state_key not in qtable:
        qtable[state_key] = create_qvector(dim_action)
    elif not isinstance(qtable[state_key], np.ndarray):
        qtable[state_key] = np.asarray(qtable[state_key], dtype=np.float32)

    # - If the State Key is in Q-Table, Return the Corresponding Q-Vector
    return qtable[state_key]


def save_qtable(qtable, path):
    """====================================================================================================
    ## Saving Q-Table to a File
    ===================================================================================================="""
    # - Ensure the Directory for the Save Path Exists
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # - Save the Q-Table using pickle
    payload = {"table": qtable}
    with open(save_path, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_qtable(path):
    """====================================================================================================
    ## Loading Q-Table from a File
    ===================================================================================================="""
    # - Load the Q-Table using pickle
    with open(path, "rb") as file:
        try:
            payload = pickle.load(file)
        except Exception:
            import torch
            file.seek(0)
            payload = torch.load(file, map_location="cpu", weights_only=False)

    # - Pick the Q-Table from the Loaded Payload
    loaded_qtable = payload.get("table", {})

    # - Return the Loaded Q-Table
    return loaded_qtable
