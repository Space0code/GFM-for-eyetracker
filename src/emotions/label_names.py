"""Helpers to resolve human-readable class names for experiment labels.

This module keeps label-name resolution configurable and reusable across EDA
summaries and plotting code. Mappings can be provided directly in task config
or loaded from a YAML spec file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

_DEFAULT_SPEC_CANDIDATES = [
    Path("specifications/hci_to_GFM_spec.yaml"),
]


def _normalize_label_name_mapping(raw_mapping: Mapping[Any, Any] | None) -> Dict[int, str]:
    """Convert arbitrary mapping keys to integer class ids and string names."""
    if not raw_mapping:
        return {}

    normalized: Dict[int, str] = {}
    for raw_key, raw_value in raw_mapping.items():
        try:
            key_int = int(raw_key)
        except (TypeError, ValueError):
            continue

        value_text = str(raw_value).strip()
        if not value_text:
            continue
        normalized[key_int] = value_text
    return normalized


def _load_mapping_from_spec_file(spec_path: str | Path, mapping_key: str) -> Dict[int, str]:
    """Load class-name mapping dictionary from one YAML spec file."""
    path = Path(spec_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}

    raw_mapping = payload.get(mapping_key)
    if not isinstance(raw_mapping, dict):
        return {}
    return _normalize_label_name_mapping(raw_mapping)


def resolve_multiclass_label_name_mapping(
    multiclass_task_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any] | None = None,
) -> Dict[int, str]:
    """Resolve raw-label -> name mapping for multiclass tasks.

    Resolution order:
    1) `multiclass_task.class_name_mapping`
    2) YAML spec mapping loaded from:
       - `multiclass_task.class_name_spec_path`, else
       - `dataset.label_mapping_spec_path`
       using key `multiclass_task.class_name_spec_key` (default:
       `emotion_id_to_name`).
    """
    direct_mapping = multiclass_task_cfg.get("class_name_mapping")
    if isinstance(direct_mapping, dict):
        normalized = _normalize_label_name_mapping(direct_mapping)
        if normalized:
            return normalized

    task_name = str(multiclass_task_cfg.get("task_name", "emotion-id")).strip().lower().replace("_", "-")
    if task_name in {"va-quadrant", "va-quadrants", "va-quadrant-4", "va-quadrant4"}:
        return {0: "LL", 1: "LH", 2: "HL", 3: "HH"}
    if task_name == "table6-arousal-3class":
        return {
            0: "Calm",
            1: "Medium arousal",
            2: "Excited/Activated",
        }
    if task_name == "table6-valence-3class":
        return {
            0: "Unpleasant",
            1: "Neutral valence",
            2: "Pleasant",
        }

    dataset_cfg = dataset_cfg or {}
    spec_path = multiclass_task_cfg.get("class_name_spec_path") or dataset_cfg.get(
        "label_mapping_spec_path"
    )
    mapping_key = str(multiclass_task_cfg.get("class_name_spec_key", "emotion_id_to_name"))

    if spec_path:
        return _load_mapping_from_spec_file(spec_path=spec_path, mapping_key=mapping_key)

    for candidate in _DEFAULT_SPEC_CANDIDATES:
        if candidate.exists():
            loaded = _load_mapping_from_spec_file(spec_path=candidate, mapping_key=mapping_key)
            if loaded:
                return loaded
    return {}


def build_encoded_class_name_mapping(
    unique_raw_labels: Sequence[int],
    raw_label_name_mapping: Mapping[int, str],
) -> Dict[int, str]:
    """Build encoded-index -> display-name mapping from raw labels."""
    encoded_mapping: Dict[int, str] = {}
    for index, raw_label in enumerate(unique_raw_labels):
        raw_value = int(raw_label)
        encoded_mapping[index] = raw_label_name_mapping.get(raw_value, str(raw_value))
    return encoded_mapping
