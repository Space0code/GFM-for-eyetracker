"""Parameter search (grid or random) for legacy emotion training.

This script generates temporary training configs by overriding selected parameters,
runs one training job per config, and aggregates per-strategy summary metrics.

Usage:
  python src/emotions/param_search.py --search_type grid
  python src/emotions/param_search.py --search_type grid --param_grid src/emotions/configs/param_search_grid.yaml
  python src/emotions/param_search.py --search_type random --n_samples 50

Notes:
- The training command contract uses `--config` (one config per run).
- Final resolved arguments are printed at run start for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file into a dictionary payload."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dictionary YAML root in: {path}")
    return payload


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save dictionary payload as YAML."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def generate_grid_combinations(param_grid: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Generate cartesian combinations from a parameter grid."""
    keys = list(param_grid.keys())
    values = [param_grid[key] if isinstance(param_grid[key], list) else [param_grid[key]] for key in keys]
    for combination in product(*values):
        yield dict(zip(keys, combination))


def generate_random_combinations(
    param_grid: Dict[str, Any],
    n_samples: int,
    random_seed: int | None = None,
) -> Iterable[Dict[str, Any]]:
    """Generate random parameter combinations.

    Sampling rule:
    - If a value list has exactly two numeric non-bool values, sample an integer in [min, max].
    - Otherwise, sample one random choice from the list.
    """
    if random_seed is not None:
        random.seed(random_seed)

    keys = list(param_grid.keys())
    for _ in range(n_samples):
        combination: Dict[str, Any] = {}
        for key in keys:
            value_list = param_grid[key] if isinstance(param_grid[key], list) else [param_grid[key]]
            if (
                len(value_list) == 2
                and isinstance(value_list[0], (int, float))
                and isinstance(value_list[1], (int, float))
                and not isinstance(value_list[0], bool)
                and not isinstance(value_list[1], bool)
            ):
                min_val, max_val = value_list[0], value_list[1]
                if min_val == max_val:
                    combination[key] = min_val
                else:
                    combination[key] = random.randint(int(min_val), int(max_val))
            else:
                combination[key] = random.choice(value_list)

        yield combination


def _set_nested_value(config: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Set nested dictionary value by key path, creating missing dict levels."""
    cursor: Dict[str, Any] = config
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def update_config_with_params(base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied config updated with one parameter combination."""
    config = copy.deepcopy(base_config)

    param_mapping: Dict[str, Tuple[str, ...]] = {
        "window_length": ("dataset", "window_length"),
        "kt": ("dataset", "kt"),
        "ks": ("dataset", "ks"),
        "hidden_channels": ("gnn", "model", "hidden_channels"),
        "use_preprocess_mlp": ("gnn", "model", "use_preprocess_mlp"),
        "add_self_loops": ("gnn", "model", "add_self_loops"),
        "strategies": ("cross_validation", "strategies"),
    }

    for param, value in params.items():
        path = param_mapping.get(param)
        if path is None:
            continue
        _set_nested_value(config, path=path, value=value)

    return config


def _discover_new_run_dir(results_dir: str, before: set[str], start_time: datetime) -> str | None:
    """Discover a new trainer run directory after one subprocess call."""
    root = Path(results_dir)
    if not root.exists():
        return None

    after = {path.name for path in root.iterdir() if path.is_dir()}
    new_dirs = sorted(after - before)
    if new_dirs:
        return str(root / new_dirs[-1])

    # Fallback to most recently modified candidate near start_time.
    candidates: List[Path] = []
    threshold = start_time - timedelta(minutes=1)
    for path in root.iterdir():
        if not path.is_dir():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        if modified >= threshold:
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime)
    return str(candidates[-1])


def _extract_run_summary_rows(run_dir: str, preferred_model: str = "GNN") -> pd.DataFrame:
    """Extract one aggregated metric row per strategy from trainer outputs."""
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for strategy_dir in sorted(run_path.iterdir()):
        if not strategy_dir.is_dir():
            continue

        summary_path = strategy_dir / "summary.csv"
        if not summary_path.exists():
            continue

        frame = pd.read_csv(summary_path)
        if frame.empty:
            continue

        if "metric_type" in frame.columns:
            frame = frame[frame["metric_type"] == "aggregated"].copy()
            if frame.empty:
                continue

        if "model" in frame.columns:
            selected = frame[frame["model"] == preferred_model].head(1)
            if selected.empty:
                selected = frame.head(1)
        else:
            selected = frame.head(1)

        source = selected.iloc[0].to_dict()
        row: Dict[str, Any] = {
            "strategy": strategy_dir.name,
            "run_dir": run_dir,
            "model": source.get("model", preferred_model),
        }
        for key, value in source.items():
            if key in {"model", "metric_type"}:
                continue
            row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def _build_ordering_columns(df: pd.DataFrame) -> Tuple[List[str], List[bool]]:
    """Return stable metric sort priority for ordered summaries."""
    preference: List[Tuple[str, bool]] = [
        ("mse", True),
        ("mae", True),
        ("balanced_accuracy", False),
        ("accuracy", False),
        ("f1", False),
        ("auc", False),
        ("macro_f1", False),
        ("macro_auc_ovr", False),
        ("ccc", False),
        ("spearman", False),
    ]

    columns: List[str] = []
    ascending: List[bool] = []
    for metric, is_ascending in preference:
        if metric in df.columns:
            columns.append(metric)
            ascending.append(is_ascending)

    return columns, ascending


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter search for emotion prediction hyperparameters")
    parser.add_argument(
        "--search_type",
        type=str,
        choices=["grid", "random"],
        default="grid",
        help="Search type: 'grid' for exhaustive search or 'random' for random sampling",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="src/emotions/configs/train.yaml",
        help="Path to base training config",
    )
    parser.add_argument(
        "--param_grid",
        type=str,
        default=None,
        help="Path to parameter grid config (defaults based on search_type)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of random samples (only used for random search)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for random search reproducibility",
    )
    parser.add_argument(
        "--preferred_model",
        type=str,
        default="GNN",
        help="Preferred model row to pull from summary.csv when multiple rows exist",
    )
    args = parser.parse_args()

    if args.param_grid is None:
        args.param_grid = (
            "src/emotions/configs/param_search_grid.yaml"
            if args.search_type == "grid"
            else "src/emotions/configs/param_search_random.yaml"
        )

    print(f"Final arguments: {vars(args)}")

    base_config = load_yaml(args.base_config)
    param_grid_config = load_yaml(args.param_grid)

    if "random_samples" in param_grid_config:
        args.n_samples = int(param_grid_config["random_samples"])
        param_grid = {key: value for key, value in param_grid_config.items() if key != "random_samples"}
    else:
        param_grid = param_grid_config

    if args.search_type == "grid":
        combinations = list(generate_grid_combinations(param_grid))
        print(f"Total combinations to evaluate: {len(combinations)}")
    else:
        combinations = list(generate_random_combinations(param_grid, args.n_samples, args.random_seed))
        print(f"Random samples to evaluate: {len(combinations)} (seed: {args.random_seed})")

    results_dir = str(base_config["logging"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="param_search_configs_")
    print(f"Temporary config directory: {temp_dir}")

    result_rows: List[Dict[str, Any]] = []

    try:
        for idx, params in enumerate(combinations):
            config_id = f"config_{idx:04d}"
            config_payload = update_config_with_params(base_config, params)
            config_path = os.path.join(temp_dir, f"{config_id}.yaml")
            save_yaml(config_payload, config_path)

            results_root = Path(results_dir)
            before_dirs: set[str] = set()
            if results_root.exists():
                before_dirs = {path.name for path in results_root.iterdir() if path.is_dir()}

            start_time = datetime.now()
            cmd = [sys.executable, "src/emotions/train.py", "--config", config_path]
            env = os.environ.copy()
            env["MKL_SERVICE_FORCE_INTEL"] = "1"

            print(f"[{idx + 1}/{len(combinations)}] Running {config_id} ...")
            proc = subprocess.run(cmd, cwd=os.getcwd(), env=env)
            if proc.returncode != 0:
                result_rows.append(
                    {
                        "config": config_id,
                        "strategy": "",
                        "run_dir": "",
                        "model": "",
                        "status": "failed",
                    }
                )
                continue

            run_dir = _discover_new_run_dir(results_dir=results_dir, before=before_dirs, start_time=start_time)
            if run_dir is None:
                result_rows.append(
                    {
                        "config": config_id,
                        "strategy": "",
                        "run_dir": "",
                        "model": "",
                        "status": "missing_run_dir",
                    }
                )
                continue

            run_df = _extract_run_summary_rows(run_dir=run_dir, preferred_model=args.preferred_model)
            if run_df.empty:
                result_rows.append(
                    {
                        "config": config_id,
                        "strategy": "",
                        "run_dir": run_dir,
                        "model": "",
                        "status": "missing_summary",
                    }
                )
                continue

            run_df = run_df.copy()
            run_df["config"] = config_id
            run_df["status"] = "success"
            result_rows.extend(run_df.to_dict(orient="records"))

        if not result_rows:
            print("No search results collected.")
            return

        results_df = pd.DataFrame(result_rows)

        param_records = []
        for idx, params in enumerate(combinations):
            record = {"config": f"config_{idx:04d}"}
            record.update(params)
            param_records.append(record)
        params_df = pd.DataFrame(param_records)

        merged_df = results_df.merge(params_df, on="config", how="left")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = f"{args.search_type}_search"
        summary_path = os.path.join(results_dir, f"{prefix}_summary_{timestamp}.csv")
        merged_df.to_csv(summary_path, index=False)
        print(f"Saved parameter search summary to: {summary_path}")

        sort_cols, sort_ascending = _build_ordering_columns(merged_df)
        if sort_cols:
            ordered_df = merged_df.sort_values(by=sort_cols, ascending=sort_ascending)
            ordered_path = os.path.join(results_dir, f"{prefix}_summary_ordered_{timestamp}.csv")
            ordered_df.to_csv(ordered_path, index=False)
            print(f"Saved ordered parameter search summary to: {ordered_path}")

            param_cols = params_df.columns.tolist()[1:]
            display_cols = [
                col
                for col in ["config", "strategy", "model", *param_cols, *sort_cols, "status"]
                if col in ordered_df.columns
            ]
            print("\nTOP 5 CONFIGURATIONS")
            print(ordered_df[display_cols].head(5).to_string(index=False))

    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
