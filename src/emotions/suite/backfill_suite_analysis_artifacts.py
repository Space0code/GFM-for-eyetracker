"""Backfill analysis artifacts for all experiments inside one suite run.

This script reads `suite_experiment_registry.csv`, dispatches per-task backfill
logic to trainer run directories, and can rebuild suite-level comparison CSVs
and plots afterward.

Usage:
  python src/emotions/suite/backfill_suite_analysis_artifacts.py \
      --suite-run-dir results/suite/<suite_run_name> \
      --skip-existing \
      --embedding-method pca \
      --generate-plots

Common options:
  --suite-run-dir PATH      Existing suite run directory (required).
  --task-types ...          Task types to process: binary and/or multiclass.
  --device auto|cpu|cuda    Device for binary GNN artifact backfill (default: auto).
  --skip-existing           Skip binary folds with existing GNN analysis artifacts.
  --embedding-method pca|tsne
                            Embedding projection for binary TP/FP/TN/FN plot generation.
  --generate-plots          Generate task-level plots after backfill.
  --skip-suite-comparison   Skip rebuilding suite-level comparison artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

# Add src directory only for direct script execution.
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from emotions.binary.backfill_gnn_analysis_artifacts import backfill_run as backfill_binary_run
from emotions.binary.results_plotting import generate_and_save_binary_results_plots
from emotions.multiclass.backfill_multiclass_analysis_artifacts import (
    backfill_run as backfill_multiclass_run,
)
from emotions.suite.compare_suite_results import build_suite_comparison_artifacts


SUPPORTED_TASK_TYPES = ("binary", "multiclass")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Backfill artifacts for one suite run directory")
    parser.add_argument(
        "--suite-run-dir",
        type=str,
        required=True,
        help="Path to existing suite run directory (must contain suite_experiment_registry.csv).",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=list(SUPPORTED_TASK_TYPES),
        choices=list(SUPPORTED_TASK_TYPES),
        help="Task types to backfill.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for binary GNN artifact backfill.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip binary fold backfill when gnn_test_analysis_artifacts.npz already exists.",
    )
    parser.add_argument(
        "--embedding-method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Embedding projection method for binary TP/FP/TN/FN embedding plots.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate task-level plots after backfill.",
    )
    parser.add_argument(
        "--skip-suite-comparison",
        action="store_true",
        help="Skip rebuilding suite-level comparison CSVs and plots.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    """Resolve torch device from CLI argument."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML as dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict YAML payload in {path}")
    return payload


def _resolve_existing_path(
    suite_run_dir: Path,
    configured_path: str,
    suite_experiment_id: str,
) -> Path:
    """Resolve trainer run path, including retained suites with moved directories."""
    path = Path(configured_path)
    candidates: List[Path] = [path]
    if not path.is_absolute():
        candidates.extend([suite_run_dir / path, Path.cwd() / path])

    # Retained/copied suite runs can have stale absolute/relative prefixes in registry.
    candidates.append(suite_run_dir / path.name)
    candidates.extend(sorted(suite_run_dir.glob(f"{suite_experiment_id}_*")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def backfill_suite_run(
    suite_run_dir: Path,
    task_types: List[str],
    device: torch.device,
    skip_existing: bool,
    generate_plots: bool,
    embedding_method: str,
    rebuild_suite_comparison: bool,
) -> None:
    """Backfill all eligible trainer runs registered in one suite."""
    registry_path = suite_run_dir / "suite_experiment_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing suite registry: {registry_path}")

    registry_df = pd.read_csv(registry_path)
    if registry_df.empty:
        print("Suite registry is empty. Nothing to backfill.")
        if rebuild_suite_comparison:
            build_suite_comparison_artifacts(
                registry_csv_path=str(registry_path),
                output_root_dir=str(suite_run_dir),
            )
        return

    selected_tasks = {str(task).strip().lower() for task in task_types}
    failures: List[str] = []
    processed_count = 0

    for row in registry_df.to_dict(orient="records"):
        suite_experiment_id = str(row.get("suite_experiment_id", "unknown"))
        status = str(row.get("status", "")).strip().lower()
        task_type = str(row.get("task_type", "")).strip().lower()
        trainer_run_dir_raw = str(row.get("trainer_run_dir", "")).strip()

        if status != "success":
            print(f"[skip] {suite_experiment_id}: status={status}")
            continue
        if task_type not in selected_tasks:
            print(f"[skip] {suite_experiment_id}: task_type={task_type} not in --task-types")
            continue
        if not trainer_run_dir_raw:
            print(f"[skip] {suite_experiment_id}: missing trainer_run_dir")
            continue

        trainer_run_dir = _resolve_existing_path(
            suite_run_dir=suite_run_dir,
            configured_path=trainer_run_dir_raw,
            suite_experiment_id=suite_experiment_id,
        )
        if not trainer_run_dir.exists():
            print(f"[skip] {suite_experiment_id}: trainer run dir not found: {trainer_run_dir}")
            continue

        try:
            print(f"[run] {suite_experiment_id}: task_type={task_type}, run_dir={trainer_run_dir}")
            if task_type == "binary":
                backfill_binary_run(
                    run_dir=trainer_run_dir,
                    device=device,
                    skip_existing=skip_existing,
                )
                if generate_plots:
                    config = _load_yaml(trainer_run_dir / "config.yaml")
                    binary_cfg = config.get("binary_task", {})
                    decision_threshold = float(binary_cfg.get("decision_threshold", 0.5))
                    saved = generate_and_save_binary_results_plots(
                        run_dir=trainer_run_dir,
                        decision_threshold=decision_threshold,
                        models_for_cm=["GNN"],
                        embedding_method=embedding_method,
                    )
                    for path in saved:
                        print(f"  Saved plot: {path}")
            elif task_type == "multiclass":
                saved = backfill_multiclass_run(
                    run_dir=trainer_run_dir,
                    generate_plots=generate_plots,
                )
                for path in saved:
                    print(f"  Saved plot: {path}")
            else:
                print(f"[skip] {suite_experiment_id}: unsupported task_type={task_type}")
                continue

            processed_count += 1
        except Exception as exc:
            failure_text = f"{suite_experiment_id}: {exc}"
            failures.append(failure_text)
            print(f"[failed] {failure_text}")

    if rebuild_suite_comparison:
        print("\nRebuilding suite comparison artifacts...")
        build_suite_comparison_artifacts(
            registry_csv_path=str(registry_path),
            output_root_dir=str(suite_run_dir),
        )
        print("Suite comparison artifacts rebuilt.")

    print(f"\nBackfill completed. Processed runs: {processed_count}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise RuntimeError(f"Suite backfill finished with {len(failures)} failure(s).")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    print(f"Arguments: {vars(args)}")

    suite_run_dir = Path(args.suite_run_dir)
    if not suite_run_dir.exists():
        raise FileNotFoundError(f"Suite run directory not found: {suite_run_dir}")

    device = _resolve_device(args.device)
    print(f"Using device: {device}")
    backfill_suite_run(
        suite_run_dir=suite_run_dir,
        task_types=list(args.task_types),
        device=device,
        skip_existing=bool(args.skip_existing),
        generate_plots=bool(args.generate_plots),
        embedding_method=str(args.embedding_method),
        rebuild_suite_comparison=not bool(args.skip_suite_comparison),
    )


if __name__ == "__main__":
    main()
