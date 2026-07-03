import json
import numpy as np


def load_boxes(path):
    """Loads a container JSON file (with a top-level 'boxes' list) into
    plain numpy arrays. Returns a dict so callers can unpack only what
    they need."""
    with open(path) as f:
        raw = json.load(f)
    data = raw["boxes"]
    n = len(data)

    return {
        "n": n,
        "x_pos": np.array([b["x"] for b in data]),
        "y_pos": np.array([b["y"] for b in data]),
        "z_pos": np.array([b["z"] for b in data]),
        "x_dim": np.array([b["x_dim"] for b in data]),
        "y_dim": np.array([b["y_dim"] for b in data]),
        "z_dim": np.array([b["z_dim"] for b in data]),
        "labels": [b["label"] for b in data],
    }