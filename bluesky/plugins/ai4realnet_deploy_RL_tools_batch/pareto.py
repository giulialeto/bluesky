"""Checkpoint catalogs for the ATM Pareto selector.

Two kinds of catalog are available, chosen in the scenario with `pareto_front`:

* `pareto_front DEMO` -- five copies of the same SAC policy with illustrative
  scores (also the default for the demo SAC environment).
* `pareto_front <FOLDER>` -- the trained checkpoints of one MORL run, e.g.
  `pareto_front bluesky_sb3_SP` (see `catalog_from_folder`). Rewards are the
  evaluation returns recorded in the folder's `dol_progress.json`.
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil


DEMO_ENV = "staticobstaclecrenv-v1"
DEMO_ALGORITHM = "sac"
FRONT_FILE = "dol_progress.json"  # inside a `pareto_front <FOLDER>` folder
# Objectives, in the order used by `weights` and `returns` in FRONT_FILE
OBJECTIVES = ("reach_reward", "avoid_reward")
DEMO_DESCRIPTION = "Illustrative demo reward; higher is better."
FILE_DESCRIPTION = "Mean evaluation return of the trained checkpoint; higher is better."
POINTS = [
    {"id": i, "checkpoint": f"policy_{i}.zip",
     "reward": {"reach_reward": reach, "avoid_reward": avoid},
     "weights": {"reach_reward": weight, "avoid_reward": round(1 - weight, 2)}}
    for i, reach, avoid, weight in [
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


def _payload(points, selected_policy_id, demo, description, objectives=OBJECTIVES):
    """Return the payload for the front-end, given a list of points and a selected policy."""
    return {
        "default_policy_id": points[0]["id"],
        "selected_policy_id": selected_policy_id,
        "demo": demo,
        "objectives": [
            {"id": name, "label": name, "description": description}
            for name in objectives
        ],
        "points": deepcopy(points),
    }


def front_payload(selected_policy_id):
    return _payload(POINTS, selected_policy_id, True, DEMO_DESCRIPTION)


class Catalog:
    """A set of selectable checkpoints: front points plus their files."""

    def __init__(self, directory, points, paths, demo, description, objectives=OBJECTIVES):
        self.directory = Path(directory)
        self.points = points
        self.demo = demo
        self.description = description
        self.objectives = objectives
        self._paths = paths

    def checkpoint_path(self, policy_id):
        if type(policy_id) is not int:
            raise ValueError("policy_id must be an integer")
        if policy_id not in self._paths:
            raise ValueError("Unknown policy_id")
        return self._paths[policy_id]

    def payload(self, selected_policy_id):
        return _payload(self.points, selected_policy_id, self.demo, self.description, self.objectives)


def demo_catalog(model_dir):
    directory = prepare_demo_checkpoints(model_dir)
    paths = {point["id"]: directory / point["checkpoint"] for point in POINTS}
    return Catalog(directory, POINTS, paths, True, DEMO_DESCRIPTION)


def resolve_front_dir(folder, search_dir):
    """Absolute path, or a folder name relative to `search_dir`."""
    path = Path(folder).expanduser()
    if not path.is_absolute():
        path = Path(search_dir) / path
    if not path.is_dir():
        raise FileNotFoundError(f"Pareto front folder not found: {path}")
    return path


def _checkpoint_file(model_path, front_dir):
    """Return the path of one checkpoint inside a MORL run folder.

    `model_path` is the string of one result in
    `dol_progress.json`. It points into the run folder on the training machine
    (e.g. "runs/<front_dir>/iterations/iter_000/models/model.zip").
    Extracts the path from the ``iterations/`` component onward and resolves it relative to ``front_dir``.

    Args:
        model_path: Checkpoint path stored in ``dol_progress.json``.
        front_dir: Local run directory containing the ``iterations/`` folder.

    Returns:
        Path of the existing checkpoint file.

    Raises:
        ValueError: `model_path` has no "iterations" component.
        FileNotFoundError: the checkpoint is not in `front_dir`.
    """
    parts = Path(model_path).parts
    if "iterations" not in parts:
        raise ValueError(f"model_path '{model_path}' has no 'iterations/' part")
    path = front_dir.joinpath(*parts[parts.index("iterations"):])
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def catalog_from_folder(folder, search_dir):
    """Catalog of the trained checkpoints of one MORL run folder.

    `folder` a folder name inside the `search_dir` (the environment's model folder). 
    Every folder has the same layout:

        <folder>/dol_progress.json
        <folder>/iterations/iter_XXX/models/model.zip

    `dol_progress.json` needs a `results` list; each entry has `weights` and
    `returns` (one value per objective, in OBJECTIVES order) and
    `metadata.model_path`. Policy ids follow the order of `results`, starting at 1.
    """
    front_dir = resolve_front_dir(folder, search_dir)
    front_file = front_dir / FRONT_FILE
    if not front_file.is_file():
        raise FileNotFoundError(f"{FRONT_FILE} not found in {front_dir}")
    try:
        data = json.loads(front_file.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{front_file} is not valid JSON: {exc}") from exc
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        raise ValueError(f"No checkpoints listed under 'results' in {front_file}")

    points, paths = [], {}
    for policy_id, result in enumerate(results, start=1):
        try:
            weights, returns = result["weights"], result["returns"]
            model_path = result["metadata"]["model_path"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Result {policy_id} in {front_file} lacks {exc}") from exc
        if len(weights) != len(OBJECTIVES) or len(returns) != len(OBJECTIVES):
            raise ValueError(
                f"Result {policy_id} in {front_file} must have {len(OBJECTIVES)} "
                f"weights and returns (objectives: {', '.join(OBJECTIVES)})")
        path = _checkpoint_file(model_path, front_dir)
        paths[policy_id] = path
        points.append({
            "id": policy_id,
            "checkpoint": path.parent.parent.name,  # e.g. iter_000
            "reward": {name: round(float(value), 4) for name, value in zip(OBJECTIVES, returns)},
            "weights": {name: round(float(value), 4) for name, value in zip(OBJECTIVES, weights)},
        })
    return Catalog(front_dir, points, paths, False, FILE_DESCRIPTION, OBJECTIVES)
