"""Wrapper config loading, validation, and normalization for HCI suite runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


_VALID_TASK_TYPES = {"binary", "multiclass", "regression"}
_DEPRECATED_RESULTS_DIR = "results/hci-experiment-suite"


class WrapperConfigError(ValueError):
    """Raised when wrapper configuration is invalid."""


def _ensure_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise WrapperConfigError(f"'{name}' must be a dictionary.")
    return value


def _remap_deprecated_results_path(path_value: str) -> str:
    """Remap deprecated suite results root to the canonical results root."""
    normalized = Path(path_value).as_posix().rstrip("/")
    deprecated = Path(_DEPRECATED_RESULTS_DIR).as_posix()
    if normalized == deprecated:
        return "results/suite"
    if normalized.startswith(f"{deprecated}/"):
        suffix = normalized[len(deprecated) + 1 :]
        return str(Path("results/suite") / suffix)
    return path_value


def _default_experiment_group(experiment_id: str, task_type: str) -> str:
    if "tag" in experiment_id:
        return "tagging"
    if task_type == "regression":
        return "regression"
    if task_type == "multiclass":
        return "multiclass"
    return "emotion"


def _normalize_scopes(experiment_id: str, experiment_cfg: Dict[str, Any]) -> List[str]:
    scope = experiment_cfg.get("scope")
    scopes = experiment_cfg.get("scopes")
    if scope is not None and scopes is not None:
        raise WrapperConfigError(
            f"Experiment '{experiment_id}' must define either 'scope' or 'scopes', not both."
        )

    if scopes is None:
        scopes = [scope] if scope is not None else []

    if not isinstance(scopes, list) or not scopes:
        raise WrapperConfigError(f"Experiment '{experiment_id}' must define non-empty scope(s).")

    cleaned = []
    for item in scopes:
        if not isinstance(item, str) or not item.strip():
            raise WrapperConfigError(
                f"Experiment '{experiment_id}' has invalid scope value: {item!r}."
            )
        cleaned.append(item.strip())
    return cleaned


def _normalize_experiment_entry(experiment_id: str, raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(raw_cfg)

    enabled = bool(cfg.get("enabled", True))
    task_type = cfg.get("task_type")
    if task_type not in _VALID_TASK_TYPES:
        raise WrapperConfigError(
            f"Experiment '{experiment_id}' has invalid task_type '{task_type}'. "
            f"Valid: {sorted(_VALID_TASK_TYPES)}"
        )

    scopes = _normalize_scopes(experiment_id, cfg)
    overrides = cfg.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise WrapperConfigError(f"Experiment '{experiment_id}'.overrides must be a dictionary.")

    if task_type in {"binary", "regression"}:
        target_column = cfg.get("target_column")
        if not isinstance(target_column, str) or not target_column.strip():
            raise WrapperConfigError(
                f"Experiment '{experiment_id}' requires string 'target_column'."
            )

    if task_type == "multiclass":
        has_target_column = isinstance(cfg.get("target_column"), str) and bool(cfg.get("target_column").strip())
        has_target_columns = isinstance(cfg.get("target_columns"), list) and len(cfg.get("target_columns")) > 0
        if not has_target_column and not has_target_columns:
            raise WrapperConfigError(
                f"Experiment '{experiment_id}' requires 'target_column' or 'target_columns'."
            )

    normalized = {
        "id": experiment_id,
        "enabled": enabled,
        "task_type": task_type,
        "experiment_group": cfg.get("experiment_group")
        or _default_experiment_group(experiment_id, task_type),
        "scopes": scopes,
        "overrides": overrides,
    }

    # Keep user fields (target definitions, thresholds, flags, etc.)
    for key, value in cfg.items():
        if key in {"scope", "scopes", "overrides", "enabled", "task_type", "experiment_group"}:
            continue
        normalized[key] = value

    return normalized


def load_and_normalize_wrapper_config(config_path: str) -> Dict[str, Any]:
    """Load wrapper YAML config, validate required structure, and normalize it."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Wrapper config not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise WrapperConfigError("Wrapper config root must be a dictionary.")

    suite = _ensure_dict(raw.get("suite"), "suite")
    base_configs = _ensure_dict(suite.get("base_configs"), "suite.base_configs")

    missing_base = [key for key in ["binary", "multiclass", "regression"] if key not in base_configs]
    if missing_base:
        raise WrapperConfigError(
            f"Missing suite.base_configs entries: {missing_base}"
        )

    suite_results_dir = _remap_deprecated_results_path(
        str(suite.get("results_dir", "results/suite"))
    )
    snapshot_cache_default = str(Path(suite_results_dir) / "_snapshot_cache")
    snapshot_cache_dir = _remap_deprecated_results_path(
        str(suite.get("snapshot_cache_dir", snapshot_cache_default))
    )

    normalized_suite = {
        "results_dir": suite_results_dir,
        "seed": int(suite.get("seed", 42)),
        "source_data_root": str(suite.get("source_data_root", "data/processed/hci-tagging")),
        "snapshot_cache_dir": snapshot_cache_dir,
        "base_configs": {
            "binary": str(base_configs["binary"]),
            "multiclass": str(base_configs["multiclass"]),
            "regression": str(base_configs["regression"]),
        },
    }

    global_overrides = deepcopy(raw.get("global_overrides", {}))
    if global_overrides is None:
        global_overrides = {}
    if not isinstance(global_overrides, dict):
        raise WrapperConfigError("'global_overrides' must be a dictionary if provided.")

    cv_cfg = global_overrides.setdefault("cross_validation", {})
    if not isinstance(cv_cfg, dict):
        raise WrapperConfigError("global_overrides.cross_validation must be a dictionary.")
    cv_cfg.setdefault("strategies", ["recording_loo"])

    raw_experiments = raw.get("experiments")
    if raw_experiments is None:
        raise WrapperConfigError("Missing required 'experiments' section.")

    normalized_experiments: List[Dict[str, Any]] = []
    if isinstance(raw_experiments, dict):
        for experiment_id, experiment_cfg in raw_experiments.items():
            if not isinstance(experiment_cfg, dict):
                raise WrapperConfigError(f"Experiment '{experiment_id}' must be a dictionary.")
            normalized_experiments.append(
                _normalize_experiment_entry(str(experiment_id), experiment_cfg)
            )
    elif isinstance(raw_experiments, list):
        for idx, experiment_cfg in enumerate(raw_experiments):
            if not isinstance(experiment_cfg, dict):
                raise WrapperConfigError(f"Experiment at index {idx} must be a dictionary.")
            experiment_id = experiment_cfg.get("id")
            if not isinstance(experiment_id, str) or not experiment_id.strip():
                raise WrapperConfigError(
                    f"Experiment at index {idx} must define non-empty string 'id'."
                )
            normalized_experiments.append(
                _normalize_experiment_entry(experiment_id.strip(), experiment_cfg)
            )
    else:
        raise WrapperConfigError("'experiments' must be a dictionary or list.")

    return {
        "suite": normalized_suite,
        "global_overrides": global_overrides,
        "experiments": normalized_experiments,
    }
