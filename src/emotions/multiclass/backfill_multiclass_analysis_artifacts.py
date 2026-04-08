"""Backfill multiclass analysis artifacts for an existing run directory.

This script is intended for already-completed multiclass runs. It validates the
run directory and regenerates multiclass figures (metrics barplots and
confusion matrices) from saved fold outputs.

Usage:
  python src/emotions/multiclass/backfill_multiclass_analysis_artifacts.py \
      --run-dir results/multiclass/<run_name>

Common options:
  --run-dir PATH            Existing multiclass trainer run directory (required).
  --models-for-cm ...       Optional explicit model order for confusion matrices.
  --skip-plot-generation    Validate run and skip figure generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

# Add src directory only for direct script execution.
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from emotions.multiclass.results_plotting_multiclass import (
    generate_and_save_multiclass_results_plots,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill multiclass analysis artifacts for one trainer run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to existing multiclass run directory (must contain config.yaml).",
    )
    parser.add_argument(
        "--models-for-cm",
        nargs="*",
        default=None,
        help=(
            "Optional explicit model order for confusion matrices "
            "(example: --models-for-cm GNN Mean SVM)."
        ),
    )
    parser.add_argument(
        "--skip-plot-generation",
        action="store_true",
        help="Validate run directory and skip figure generation.",
    )
    return parser.parse_args()


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load run config YAML from multiclass run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        registry_path = run_dir / "suite_experiment_registry.csv"
        if registry_path.exists():
            raise ValueError(
                f"Received suite root '{run_dir}', but multiclass backfill expects one trainer run dir "
                "(containing config.yaml). Use "
                "`python src/emotions/suite/backfill_suite_analysis_artifacts.py --suite-run-dir <suite_dir>` "
                "for suite-wide backfill."
            )
        raise FileNotFoundError(f"Missing run config: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid run config content at {config_path}")
    return payload


def backfill_run(
    run_dir: Path,
    generate_plots: bool = True,
    models_for_cm: Sequence[str] | None = None,
) -> List[Path]:
    """Validate one multiclass run and optionally regenerate figure artifacts."""
    config = _load_run_config(run_dir)
    run_experiments = config.get("run_experiments", {})
    if not bool(run_experiments.get("gnn", False)) and not bool(run_experiments.get("baselines", False)):
        raise ValueError(
            "Run config has run_experiments.gnn=false and run_experiments.baselines=false; nothing to backfill."
        )

    if not generate_plots:
        return []

    saved_paths = generate_and_save_multiclass_results_plots(
        run_dir=run_dir,
        models_for_cm=list(models_for_cm) if models_for_cm is not None else None,
    )
    return saved_paths


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    print(f"Arguments: {vars(args)}")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    saved_paths = backfill_run(
        run_dir=run_dir,
        generate_plots=not bool(args.skip_plot_generation),
        models_for_cm=args.models_for_cm,
    )
    if saved_paths:
        for path in saved_paths:
            print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
