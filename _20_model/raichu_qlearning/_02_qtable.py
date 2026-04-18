import pickle
from pathlib import Path

import numpy as np

from _20_model import raichu_qlearning


def create_qtable():
    return {}


def create_qvector(dim_action=None):
    if dim_action is None:
        dim_action = len(raichu_qlearning._04_action_space_design.internal_action_names())
    return np.zeros(int(dim_action), dtype=np.float32)


def get_qvector(qtable, state_key):
    dim_action = len(raichu_qlearning._04_action_space_design.internal_action_names())
    state_key = tuple(state_key)

    if state_key not in qtable:
        qtable[state_key] = create_qvector(dim_action)
    elif not isinstance(qtable[state_key], np.ndarray):
        qtable[state_key] = np.asarray(qtable[state_key], dtype=np.float32)

    if qtable[state_key].shape[0] != dim_action:
        resized = create_qvector(dim_action)
        copy_size = min(dim_action, qtable[state_key].shape[0])
        resized[:copy_size] = qtable[state_key][:copy_size]
        qtable[state_key] = resized

    return qtable[state_key]


def save_qtable(qtable, path, metadata=None):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "table": qtable,
        "action_names": list(raichu_qlearning._04_action_space_design.internal_action_names()),
    }
    if metadata is not None:
        payload.update(metadata)

    with open(save_path, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_qtable(path):
    with open(path, "rb") as file:
        try:
            payload = pickle.load(file)
        except Exception:
            import torch
            file.seek(0)
            payload = torch.load(file, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        payload = {"table": {}}
    if "table" not in payload:
        payload = {"table": payload}

    table = {}
    for state_key, qvalues in payload.get("table", {}).items():
        table[tuple(state_key)] = np.asarray(qvalues, dtype=np.float32)

    payload["table"] = table
    payload.setdefault(
        "action_names",
        list(raichu_qlearning._04_action_space_design.internal_action_names()),
    )
    payload.setdefault("visit_counts", {})
    payload.setdefault("total_updates", 0)
    return payload
