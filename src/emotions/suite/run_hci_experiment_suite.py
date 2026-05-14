"""Run full HCI experiment suite (EDA + training + comparison) from one wrapper config.

Usage:
  python src/emotions/suite/run_hci_experiment_suite.py \
      --config src/emotions/suite/configs/run_hci_experiment_suite.yaml
"""

from __future__ import annotations

import argparse
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

import sys

# Add src directory only for direct script execution.
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from emotions.binary.train_binary import run_training_from_config as run_binary_training
from emotions.common.dataset_config import (
    apply_defaults_when_missing,
    resolve_dropna_columns,
    resolve_feature_columns,
    resolve_min_samples_per_window,
)
from emotions.multiclass.train_multiclass import (
    run_training_from_config as run_multiclass_training,
)
from emotions.regression.train_regression import (
    run_training_from_config as run_regression_training,
)
from emotions.suite.compare_suite_results import build_suite_comparison_artifacts
from emotions.suite.config_merge import merge_many
from emotions.suite.data_snapshot import (
    SnapshotBuildResult,
    build_clean_snapshot_dataframe,
    write_snapshot_artifacts,
)
from emotions.suite.eda_summary import build_eda_summary
from emotions.suite.wrapper_config_schema import load_and_normalize_wrapper_config
from emotions.train_baseline import build_tabular_samples
from emotions.label_names import resolve_multiclass_label_name_mapping
from emotions.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HCI experiment suite wrapper")
    parser.add_argument(
        "--config",
        type=str,
        default="src/emotions/suite/configs/run_hci_experiment_suite.yaml",
        help="Path to wrapper YAML config",
    )
    return parser.parse_args()


def _scope_dataset_defaults(scope: str) -> Dict[str, Any]:
    if scope == "emotion-elicitation":
        return {
            "allowed_experiment_types": ["emotion-elicitation"],
            "label_quality_column": "emotion-derivation-status",
            "allowed_label_quality_values": ["ok"],
        }

    if scope in {"image-tagging-1", "image-tagging-2", "video-tagging"}:
        return {
            "allowed_experiment_types": [scope],
            "label_quality_column": "tag-derivation-status",
            "allowed_label_quality_values": ["ok"],
        }

    if scope == "pooled-tagging":
        return {
            "allowed_experiment_types": ["image-tagging-1", "image-tagging-2", "video-tagging"],
            "label_quality_column": "tag-derivation-status",
            "allowed_label_quality_values": ["ok"],
        }

    raise ValueError(
        f"Unsupported scope '{scope}'. Expected emotion-elicitation, image-tagging-1, image-tagging-2, video-tagging, pooled-tagging."
    )


def _required_target_columns(task_type: str, experiment_cfg: Dict[str, Any]) -> List[str]:
    if task_type in {"binary", "regression"}:
        return [str(experiment_cfg["target_column"])]

    task_name = str(experiment_cfg.get("task_name", "emotion-id")).strip().lower().replace("_", "-")
    if task_name in {"emotion-id", "feltemo"}:
        return [str(experiment_cfg.get("target_column", "emotion-id"))]
    if task_name in {"table6-arousal-3class", "table6-valence-3class"}:
        return [str(experiment_cfg.get("target_column", "emotion-id"))]

    # VA quadrant multiclass
    target_columns = experiment_cfg.get("target_columns") or ["emotion-valence", "emotion-arousal"]
    if not isinstance(target_columns, list) or len(target_columns) != 2:
        raise ValueError("va_quadrant requires two target_columns")
    return [str(target_columns[0]), str(target_columns[1])]


def _threshold_description(task_type: str, experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if task_type in {"binary", "regression"}:
        return {
            "task_type": task_type,
            "target_column": experiment_cfg.get("target_column"),
            "threshold": experiment_cfg.get("threshold"),
        }

    task_name = str(experiment_cfg.get("task_name", "emotion-id")).strip().lower().replace("_", "-")
    if task_name in {"emotion-id", "feltemo"}:
        return {
            "task_type": task_type,
            "task_name": "emotion-id",
            "target_column": experiment_cfg.get("target_column", "emotion-id"),
        }

    if task_name in {"table6-arousal-3class", "table6-valence-3class"}:
        return {
            "task_type": task_type,
            "task_name": task_name,
            "target_column": experiment_cfg.get("target_column", "emotion-id"),
            "table6_class_mapping": experiment_cfg.get("table6_class_mapping", {}),
            "drop_unmapped_labels": bool(experiment_cfg.get("drop_unmapped_labels", True)),
            "use_table6_3class_targets": bool(experiment_cfg.get("use_table6_3class_targets", False)),
        }

    return {
        "task_type": task_type,
        "task_name": "va_quadrant",
        "target_columns": experiment_cfg.get("target_columns", ["emotion-valence", "emotion-arousal"]),
        "thresholds": experiment_cfg.get("thresholds", {}),
    }


def _ensure_dropna_targets(dataset_cfg: Dict[str, Any], target_columns: List[str]) -> None:
    resolve_dropna_columns(dataset_cfg, target_columns=target_columns)


def _apply_scope_defaults(dataset_cfg: Dict[str, Any], scope: str) -> None:
    """Apply scope defaults only for keys absent from merged overrides."""
    scope_defaults = _scope_dataset_defaults(scope)
    apply_defaults_when_missing(dataset_cfg, scope_defaults)


def _apply_task_to_trainer_config(
    trainer_config: Dict[str, Any],
    task_type: str,
    experiment_cfg: Dict[str, Any],
) -> None:
    if task_type == "binary":
        trainer_config.setdefault("binary_task", {})
        trainer_config["binary_task"]["target_column"] = experiment_cfg["target_column"]
        if "threshold" in experiment_cfg:
            trainer_config["binary_task"]["threshold"] = experiment_cfg["threshold"]
        if "decision_threshold" in experiment_cfg:
            trainer_config["binary_task"]["decision_threshold"] = experiment_cfg["decision_threshold"]
        return

    if task_type == "multiclass":
        trainer_config.setdefault("multiclass_task", {})
        task_name = str(experiment_cfg.get("task_name", "emotion-id")).strip().lower().replace("_", "-")
        trainer_config["multiclass_task"]["task_name"] = task_name
        if "target_column" in experiment_cfg:
            trainer_config["multiclass_task"]["target_column"] = experiment_cfg["target_column"]
        if "target_columns" in experiment_cfg:
            trainer_config["multiclass_task"]["target_columns"] = experiment_cfg["target_columns"]
        if "thresholds" in experiment_cfg:
            trainer_config["multiclass_task"]["thresholds"] = experiment_cfg["thresholds"]
        if "threshold" in experiment_cfg:
            trainer_config["multiclass_task"]["threshold"] = experiment_cfg["threshold"]
        for key in [
            "class_name_mapping",
            "class_name_spec_path",
            "class_name_spec_key",
            "table6_class_mapping",
            "drop_unmapped_labels",
            "use_table6_3class_targets",
            "class_downsampling",
        ]:
            if key in experiment_cfg:
                trainer_config["multiclass_task"][key] = experiment_cfg[key]
        return

    if task_type == "regression":
        trainer_config.setdefault("regression_task", {})
        trainer_config["regression_task"]["target_column"] = experiment_cfg["target_column"]
        return

    raise ValueError(f"Unsupported task_type '{task_type}'")


def _train_dispatch(task_type: str, config_path: str) -> str:
    if task_type == "binary":
        return run_binary_training(config_path)
    if task_type == "multiclass":
        return run_multiclass_training(config_path)
    if task_type == "regression":
        return run_regression_training(config_path)
    raise ValueError(f"Unsupported task_type '{task_type}'")


def _sanitize_run_name_token(value: str) -> str:
    """Return filesystem-safe token for trainer run folder prefix."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._-") or "experiment"


def _build_suite_experiment_id(experiment_id: str, scope: str) -> str:
    """Return stable experiment token used for folder names and registry IDs."""
    return _sanitize_run_name_token(f"{experiment_id}_{scope}")


def _compute_window_counts(
    dataset_cfg: Dict[str, Any],
    snapshot_csv_path: str,
    target_columns: List[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    min_samples = resolve_min_samples_per_window(dataset_cfg)
    feature_columns = resolve_feature_columns(dataset_cfg)

    samples = build_tabular_samples(
        data_dir=None,
        data_filepath=snapshot_csv_path,
        filter_subjects=None,
        filter_recordings=None,
        file_list=None,
        window_length=int(dataset_cfg.get("window_length", 10)),
        window_overlap=float(dataset_cfg.get("window_overlap", 0.0)),
        min_samples_per_window=min_samples,
        dropping_emotion_threshold=float(dataset_cfg.get("dropping_emotion_threshold", -1)),
        feature_columns=feature_columns,
        target_columns=target_columns,
        target_aggregation=dataset_cfg.get("target_aggregation", "mean"),
        dropna_columns=dataset_cfg.get("dropna_columns"),
        experiment_type_column=dataset_cfg.get("experiment_type_column", "experiment-type"),
        allowed_experiment_types=dataset_cfg.get("allowed_experiment_types"),
        label_quality_column=dataset_cfg.get("label_quality_column"),
        allowed_label_quality_values=dataset_cfg.get("allowed_label_quality_values"),
    )

    subject_counts: Dict[str, int] = {}
    recording_counts: Dict[str, int] = {}
    for sample in samples:
        subject = str(sample.subject)
        recording = str(sample.recording)
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
        recording_counts[recording] = recording_counts.get(recording, 0) + 1

    return subject_counts, recording_counts


def _resolve_eda_label_name_mapping(
    task_type: str,
    experiment_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> Dict[int, str]:
    """Resolve label-name mapping for EDA summary display."""
    if task_type != "multiclass":
        return {}
    return resolve_multiclass_label_name_mapping(
        multiclass_task_cfg=experiment_cfg,
        dataset_cfg=dataset_cfg,
    )


def run_suite(wrapper_config_path: str) -> str:
    wrapper_config = load_and_normalize_wrapper_config(wrapper_config_path)

    suite_cfg = wrapper_config["suite"]
    global_overrides = wrapper_config["global_overrides"]
    experiments = wrapper_config["experiments"]
    snapshot_cache_dir = Path(
        str(suite_cfg.get("snapshot_cache_dir", Path(suite_cfg["results_dir"]) / "_snapshot_cache"))
    )

    np.random.seed(int(suite_cfg["seed"]))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suite_run_dir = Path(suite_cfg["results_dir"]) / timestamp
    suite_run_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = suite_run_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    with (suite_run_dir / "wrapper_config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(wrapper_config, handle, sort_keys=False)

    registry_rows: List[Dict[str, Any]] = []

    for experiment_cfg in experiments:
        if not experiment_cfg.get("enabled", True):
            continue

        task_type = str(experiment_cfg["task_type"])
        experiment_id = str(experiment_cfg["id"])
        experiment_group = str(experiment_cfg.get("experiment_group", "unknown"))
        base_config_path = str(suite_cfg["base_configs"][task_type])

        for scope in experiment_cfg["scopes"]:
            suite_experiment_id = _build_suite_experiment_id(experiment_id=experiment_id, scope=scope)
            experiment_dir = experiments_dir / suite_experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)

            row: Dict[str, Any] = {
                "suite_experiment_id": suite_experiment_id,
                "experiment_id": experiment_id,
                "scope": scope,
                "task_type": task_type,
                "experiment_group": experiment_group,
                "task_suite_root": str(suite_run_dir),
                "status": "pending",
                "error": "",
                "trainer_run_dir": "",
                "snapshot_csv": str(experiment_dir / "snapshot.csv"),
                "snapshot_manifest": str(experiment_dir / "snapshot_manifest.yaml"),
                "eda_summary_path": str(experiment_dir / "eda_summary.txt"),
                "resolved_config_path": str(experiment_dir / "resolved_trainer_config.yaml"),
                "label_entropy": np.nan,
            }

            try:
                base_config = load_config(base_config_path)
                merged_config = merge_many(
                    base_config,
                    global_overrides,
                    experiment_cfg.get("overrides", {}),
                )

                _apply_task_to_trainer_config(
                    trainer_config=merged_config,
                    task_type=task_type,
                    experiment_cfg=experiment_cfg,
                )

                dataset_cfg = merged_config.setdefault("dataset", {})
                if not isinstance(dataset_cfg, dict):
                    raise ValueError("Resolved trainer config dataset section must be a dictionary.")

                _apply_scope_defaults(dataset_cfg=dataset_cfg, scope=scope)

                target_columns = _required_target_columns(task_type, experiment_cfg)
                _ensure_dropna_targets(dataset_cfg, target_columns)

                snapshot_result: SnapshotBuildResult = build_clean_snapshot_dataframe(
                    source_data_root=str(suite_cfg["source_data_root"]),
                    scope=scope,
                    dataset_cfg=dataset_cfg,
                    target_columns=target_columns,
                    experiment_id=suite_experiment_id,
                    threshold_description=_threshold_description(task_type, experiment_cfg),
                    use_cache=bool(dataset_cfg.get("use_cache", False)),
                    cache_dir=str(snapshot_cache_dir),
                )

                write_snapshot_artifacts(
                    snapshot=snapshot_result,
                    snapshot_csv_path=row["snapshot_csv"],
                    manifest_yaml_path=row["snapshot_manifest"],
                )

                if snapshot_result.dataframe.empty:
                    raise ValueError("Snapshot is empty after filtering/cleaning.")

                subject_counts, recording_counts = _compute_window_counts(
                    dataset_cfg=dataset_cfg,
                    snapshot_csv_path=row["snapshot_csv"],
                    target_columns=target_columns,
                )

                eda_result = build_eda_summary(
                    snapshot_df=snapshot_result.dataframe,
                    task_type=task_type,
                    experiment_cfg=experiment_cfg,
                    label_name_mapping=_resolve_eda_label_name_mapping(
                        task_type=task_type,
                        experiment_cfg=experiment_cfg,
                        dataset_cfg=dataset_cfg,
                    ),
                    window_subject_counts=subject_counts,
                    window_recording_counts=recording_counts,
                    snapshot_manifest=snapshot_result.manifest,
                )
                with Path(row["eda_summary_path"]).open("w", encoding="utf-8") as handle:
                    handle.write(eda_result.text)
                row["label_entropy"] = eda_result.stats.get("label_entropy", np.nan)

                # Force trainer to consume the exact snapshot from EDA phase.
                dataset_cfg["data_filepath"] = row["snapshot_csv"]
                dataset_cfg.pop("data_dir", None)
                dataset_cfg.pop("file_list", None)
                dataset_cfg["filter_subjects"] = None
                dataset_cfg["filter_recordings"] = None

                logging_cfg = merged_config.setdefault("logging", {})
                if not isinstance(logging_cfg, dict):
                    raise ValueError("Resolved trainer config logging section must be a dictionary.")

                trainer_results_root = suite_run_dir
                logging_cfg["results_dir"] = str(trainer_results_root)
                logging_cfg["run_name_prefix"] = _sanitize_run_name_token(suite_experiment_id)

                with Path(row["resolved_config_path"]).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(merged_config, handle, sort_keys=False)

                trainer_run_dir = _train_dispatch(task_type=task_type, config_path=row["resolved_config_path"])
                row["trainer_run_dir"] = str(trainer_run_dir)
                row["status"] = "success"
            except Exception as exc:  # Continue suite on per-experiment failures.
                row["status"] = "failed"
                error_text = f"{exc}\n{traceback.format_exc()}"
                row["error"] = error_text
                print("\n" + "!" * 100)
                print(f"[SUITE][FAILED] {suite_experiment_id}")
                print(f"Reason: {exc}")
                print("Traceback:")
                print(traceback.format_exc())
                print("!" * 100 + "\n")

            registry_rows.append(row)
            suite_run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(registry_rows).to_csv(
                suite_run_dir / "suite_experiment_registry.csv",
                index=False,
            )

    registry_csv = suite_run_dir / "suite_experiment_registry.csv"
    if registry_csv.exists():
        build_suite_comparison_artifacts(
            registry_csv_path=str(registry_csv),
            output_root_dir=str(suite_run_dir),
        )

    return str(suite_run_dir)


def main() -> None:
    args = parse_args()
    suite_dir = run_suite(args.config)
    print(f"Suite run directory: {suite_dir}")


if __name__ == "__main__":
    main()
