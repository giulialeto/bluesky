"""Demo checkpoint catalog for the ATM Pareto selector.

Scores are illustrative, not evaluation results. All five checkpoints start
as copies of the same policy. Replace the files and scores with trained
policies and measured rewards when available.
"""
from copy import deepcopy
from pathlib import Path
import shutil


DEMO_ENV = "staticobstaclecrenv-v1"
DEMO_ALGORITHM = "sac"
POINTS = [
    {"id": i, "checkpoint": f"policy_{i}.zip",
     "reward": {"drift_reward": drift, "progress_reward": progress},
     "weights": {"drift_reward": weight, "progress_reward": round(1 - weight, 2)}}
    for i, drift, progress, weight in [
        (1, -0.1, 0.2, 1.0),
        (2, -0.3, 0.45, 0.75),
        (3, -0.55, 0.65, 0.5),
        (4, -0.8, 0.8, 0.25),
        (5, -1.1, 0.9, 0.0),
    ]
]


def prepare_demo_checkpoints(model_dir):
    """Create missing demo copies only; never overwrite replacement policies."""
    model_dir = Path(model_dir)
    target = model_dir / "pareto"
    target.mkdir(exist_ok=True)
    for point in POINTS:
        path = target / point["checkpoint"]
        if not path.exists():
            shutil.copyfile(model_dir / "model.zip", path)
    return target


def checkpoint_path(directory, policy_id):
    if type(policy_id) is not int:
        raise ValueError("policy_id must be an integer")
    point = next((p for p in POINTS if p["id"] == policy_id), None)
    if point is None:
        raise ValueError("Unknown policy_id")
    return Path(directory) / point["checkpoint"]


def front_payload(selected_policy_id):
    return {
        "default_policy_id": 1,
        "selected_policy_id": selected_policy_id,
        "demo": True,
        "objectives": [
            {"id": name, "label": name,
             "description": "Illustrative demo reward; higher is better."}
            for name in ("drift_reward", "progress_reward")
        ],
        "points": deepcopy(POINTS),
    }
