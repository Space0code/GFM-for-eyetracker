"""Train multiclass emotion models (GNN + baselines) with CV.

Usage:
  python src/emotions/multiclass/train_multiclass.py \
      --config src/emotions/multiclass/configs/train_multiclass_hci_tagging.yaml

The default config now targets the Mahnob-HCI Table-6 three-class arousal
objective. For the companion valence objective, use:
  python src/emotions/multiclass/train_multiclass.py \
      --config src/emotions/multiclass/configs/train_multiclass_hci_tagging_table6_valence_3class.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

# Resolve warnings and matmul precision
torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True

# Add src directory only for direct script execution.
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from data.data import SpacioTemporalDataset
from emotions.common.cv_utils import (
    describe_fold,
    validate_kfold_group_disjointness,
    validate_non_empty_train_splits,
)
from emotions.common.dataset_config import (
    build_graph_dataset_kwargs,
    build_tabular_samples_kwargs,
    resolve_dropna_columns,
    resolve_feature_columns,
    resolve_min_samples_per_window,
    sync_gnn_in_channels,
)
from emotions.multiclass.baseline_model_multiclass import get_multiclass_baseline_by_name
from emotions.multiclass.metrics_multiclass import evaluate_multiclass_classification
from emotions.multiclass.model_multiclass import MulticlassSpatioTemporalGNN, MulticlassSpatioTemporalGNNV1
from emotions.multiclass.results_plotting_multiclass import (
    generate_and_save_multiclass_results_plots,
)
from emotions.label_names import (
    build_encoded_class_name_mapping,
    resolve_multiclass_label_name_mapping,
)
from emotions.train_baseline import build_tabular_samples, samples_to_xy
from emotions.utils import (
    Logger,
    create_splitter,
    load_config,
    print_comparison_table,
    save_comparison_csv,
    save_fold_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for multiclass training."""
    parser = argparse.ArgumentParser(description="Train multiclass classification models")
    parser.add_argument(
        "--config",
        type=str,
        default="src/emotions/multiclass/configs/train_multiclass_hci_tagging.yaml",
        help=(
            "Path to multiclass YAML config file. "
            "Default: Mahnob-HCI Table-6 three-class arousal config."
        ),
    )
    return parser.parse_args()


def _resolve_threshold_value(threshold_spec: Any, train_values: List[float]) -> float:
    series = pd.to_numeric(pd.Series(train_values), errors="coerce").dropna()
    if series.empty:
        raise ValueError("Cannot compute threshold from empty train targets.")

    if isinstance(threshold_spec, str):
        mode = threshold_spec.strip().lower()
        if mode == "mean":
            return float(series.mean())
        if mode == "median":
            return float(series.median())
        try:
            return float(mode)
        except ValueError as exc:
            raise ValueError(
                f"Invalid threshold '{threshold_spec}'. Use float, 'median', or 'mean'."
            ) from exc

    return float(threshold_spec)


def _validate_non_empty_train_splits(
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    strategy: str,
    dataset_label: str,
) -> None:
    validate_non_empty_train_splits(
        splits=splits,
        strategy=strategy,
        dataset_label=dataset_label,
    )


def _validate_kfold_group_disjointness(
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    strategy: str,
    dataset: List[Any],
    dataset_label: str,
) -> None:
    validate_kfold_group_disjointness(
        splits=splits,
        strategy=strategy,
        dataset=dataset,
        dataset_label=dataset_label,
        combined_id_style="underscore",
    )


def _fit_graph_feature_scaler(
    dataset: SpacioTemporalDataset,
    train_idx: np.ndarray,
) -> StandardScaler:
    arrays = []
    for idx in train_idx:
        graph = dataset[int(idx)]
        arrays.append(graph["node"].x.detach().cpu().numpy())
    scaler = StandardScaler()
    scaler.fit(np.vstack(arrays))
    return scaler


def _normalize_int_mapping(raw_mapping: Any, mapping_name: str) -> Dict[int, int]:
    """Normalize arbitrary mapping keys/values to integer->integer mapping."""
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"{mapping_name} must be a dictionary.")

    normalized: Dict[int, int] = {}
    for raw_key, raw_value in raw_mapping.items():
        try:
            key = int(raw_key)
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{mapping_name} keys/values must be integers. Got ({raw_key!r}, {raw_value!r})."
            ) from exc
        normalized[key] = value
    return normalized


def _resolve_table6_enable_flag(
    multiclass_task_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> bool:
    """Resolve explicit Table-6 mode flag from task config or YAML spec default."""
    explicit_flag = multiclass_task_cfg.get("use_table6_3class_targets")
    if explicit_flag is not None:
        return bool(explicit_flag)

    spec_path = dataset_cfg.get("label_mapping_spec_path")
    if not isinstance(spec_path, str) or not spec_path.strip():
        return False

    path = Path(spec_path)
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("use_table6_3class_targets", False))


def _resolve_task_definition(
    multiclass_task_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    task_name = str(multiclass_task_cfg.get("task_name", "emotion-id")).strip().lower().replace("_", "-")

    if task_name in {"emotion-id", "feltemo"}:
        target_column = multiclass_task_cfg.get("target_column", "emotion-id")
        raw_label_names = resolve_multiclass_label_name_mapping(
            multiclass_task_cfg=multiclass_task_cfg,
            dataset_cfg=dataset_cfg,
        )
        return {
            "mode": "emotion-id",
            "task_name": "emotion-id",
            "target_columns": [target_column],
            "target_column": target_column,
            "raw_label_names": raw_label_names,
        }

    if task_name in {"va-quadrant", "va-quadrants", "va-quadrant-4", "va-quadrant4"}:
        target_columns = multiclass_task_cfg.get("target_columns", ["emotion-valence", "emotion-arousal"])
        if not isinstance(target_columns, list) or len(target_columns) != 2:
            raise ValueError("VA quadrant multiclass task requires exactly two target_columns.")

        thresholds = multiclass_task_cfg.get("thresholds", {}) or {}
        if not isinstance(thresholds, dict):
            raise ValueError("multiclass_task.thresholds must be a dictionary.")

        raw_label_names = resolve_multiclass_label_name_mapping(
            multiclass_task_cfg=multiclass_task_cfg,
            dataset_cfg=dataset_cfg,
        )
        return {
            "mode": "va-quadrant",
            "task_name": "va_quadrant",
            "target_columns": target_columns,
            "threshold_specs": {
                "valence": thresholds.get("valence", multiclass_task_cfg.get("threshold", "mean")),
                "arousal": thresholds.get("arousal", multiclass_task_cfg.get("threshold", "mean")),
            },
            "raw_label_names": raw_label_names,
        }

    if task_name in {"table6-arousal-3class", "table6-valence-3class"}:
        enabled = _resolve_table6_enable_flag(
            multiclass_task_cfg=multiclass_task_cfg,
            dataset_cfg=dataset_cfg,
        )
        if not enabled:
            raise ValueError(
                "Table-6 multiclass task requested but disabled. "
                "Set multiclass_task.use_table6_3class_targets=true or set "
                "use_table6_3class_targets: true in the YAML spec."
            )

        target_column = str(multiclass_task_cfg.get("target_column", "emotion-id"))
        table6_mapping = _normalize_int_mapping(
            raw_mapping=multiclass_task_cfg.get("table6_class_mapping"),
            mapping_name="multiclass_task.table6_class_mapping",
        )

        raw_label_names = resolve_multiclass_label_name_mapping(
            multiclass_task_cfg=multiclass_task_cfg,
            dataset_cfg=dataset_cfg,
        )
        if not raw_label_names:
            if task_name == "table6-arousal-3class":
                raw_label_names = {
                    0: "Calm",
                    1: "Medium aroused",
                    2: "Excited/Activated",
                }
            else:
                raw_label_names = {
                    0: "Unpleasant",
                    1: "Neutral valence",
                    2: "Pleasant",
                }

        return {
            "mode": "table6-3class",
            "task_name": task_name,
            "target_columns": [target_column],
            "target_column": target_column,
            "table6_class_mapping": table6_mapping,
            "drop_unmapped_labels": bool(multiclass_task_cfg.get("drop_unmapped_labels", True)),
            "raw_label_names": raw_label_names,
        }

    raise ValueError(
        "Unsupported multiclass task_name. Supported: emotion-id, va-quadrant, "
        "table6-arousal-3class, table6-valence-3class"
    )


def _extract_graph_raw_targets(
    dataset: SpacioTemporalDataset,
    indices: np.ndarray,
    target_columns: List[str],
) -> np.ndarray:
    rows: List[List[float]] = []
    for idx in indices:
        graph = dataset[int(idx)]
        names = getattr(graph, "emotion_names", getattr(dataset, "emotion_names", []))
        row: List[float] = []
        for column in target_columns:
            if column not in names:
                raise ValueError(f"Graph target column '{column}' not found.")
            col_idx = names.index(column)
            row.append(float(graph.y[col_idx].item()))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _extract_tabular_raw_targets(
    samples: List[Any],
    indices: np.ndarray,
    target_columns: List[str],
) -> np.ndarray:
    rows: List[List[float]] = []
    for idx in indices:
        sample = samples[int(idx)]
        row: List[float] = []
        for column in target_columns:
            if column not in sample.targets:
                raise ValueError(f"Tabular target column '{column}' missing in sample.")
            row.append(float(sample.targets[column]))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _resolve_fold_context(task_def: Dict[str, Any], train_raw_targets: np.ndarray) -> Dict[str, Any]:
    if train_raw_targets.size == 0:
        raise ValueError("Cannot resolve fold label context with empty train targets.")

    if task_def["mode"] in {"emotion-id", "table6-3class"}:
        return {}

    valence_train = train_raw_targets[:, 0]
    arousal_train = train_raw_targets[:, 1]
    v_thr = _resolve_threshold_value(task_def["threshold_specs"]["valence"], valence_train.tolist())
    a_thr = _resolve_threshold_value(task_def["threshold_specs"]["arousal"], arousal_train.tolist())
    return {
        "valence_threshold": v_thr,
        "arousal_threshold": a_thr,
    }


def _raw_to_label(task_def: Dict[str, Any], raw_values: np.ndarray, fold_context: Dict[str, Any]) -> np.ndarray:
    if task_def["mode"] == "emotion-id":
        labels = np.asarray(raw_values[:, 0], dtype=float)
        return np.asarray(np.round(labels), dtype=int)

    if task_def["mode"] == "table6-3class":
        emotion_ids = np.asarray(np.round(raw_values[:, 0]), dtype=int)
        mapping = task_def["table6_class_mapping"]
        mapped = np.asarray([int(mapping.get(int(value), -1)) for value in emotion_ids], dtype=int)
        return mapped

    valence = np.asarray(raw_values[:, 0], dtype=float)
    arousal = np.asarray(raw_values[:, 1], dtype=float)
    v_thr = float(fold_context["valence_threshold"])
    a_thr = float(fold_context["arousal_threshold"])

    ll = (valence <= v_thr) & (arousal <= a_thr)
    lh = (valence <= v_thr) & (arousal > a_thr)
    hl = (valence > v_thr) & (arousal <= a_thr)
    labels = np.where(ll, 0, np.where(lh, 1, np.where(hl, 2, 3)))
    return labels.astype(int)


def _encode_labels(raw_labels: np.ndarray, class_to_index: Dict[int, int]) -> np.ndarray:
    encoded: List[int] = []
    for value in raw_labels.tolist():
        if int(value) not in class_to_index:
            raise ValueError(
                f"Observed label {value} not present in class mapping keys {sorted(class_to_index.keys())}."
            )
        encoded.append(class_to_index[int(value)])
    return np.asarray(encoded, dtype=int)


class _IndexSubset(Sequence[Any]):
    """Lightweight index view over a dataset without copying samples."""

    def __init__(self, dataset: Sequence[Any], indices: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=int)
        self.emotion_names = getattr(dataset, "emotion_names", [])

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[int(self.indices[int(idx)])]


def _resolve_class_downsampling_cfg(multiclass_task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized optional class-downsampling settings."""
    cfg = multiclass_task_cfg.get("class_downsampling", {})
    if cfg is None:
        return {"enabled": False}
    if not isinstance(cfg, dict):
        raise ValueError("multiclass_task.class_downsampling must be a dictionary.")
    return cfg


def _select_class_downsample_indices(
    raw_targets: np.ndarray,
    task_def: Dict[str, Any],
    downsampling_cfg: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Select sample indices after optional mapped-class downsampling."""
    all_indices = np.arange(len(raw_targets), dtype=int)
    if not bool(downsampling_cfg.get("enabled", False)):
        return all_indices, {"enabled": False}

    if task_def["mode"] != "table6-3class":
        raise ValueError("class_downsampling is currently supported only for Table-6 multiclass tasks.")

    strategy = str(downsampling_cfg.get("strategy", "match_class_count"))
    if strategy != "match_class_count":
        raise ValueError("Unsupported class_downsampling.strategy. Use 'match_class_count'.")

    source_class = int(downsampling_cfg["source_class"])
    target_class = int(downsampling_cfg["target_class"])
    random_state = int(downsampling_cfg.get("random_state", 42))

    mapped_labels = _raw_to_label(task_def=task_def, raw_values=raw_targets, fold_context={})
    valid_mask = mapped_labels >= 0 if task_def.get("drop_unmapped_labels", False) else np.ones_like(mapped_labels, dtype=bool)
    valid_indices = all_indices[valid_mask]
    valid_labels = mapped_labels[valid_mask]

    source_indices = valid_indices[valid_labels == source_class]
    target_indices = valid_indices[valid_labels == target_class]
    if len(target_indices) == 0:
        raise ValueError(f"Cannot downsample class {source_class}: target class {target_class} has zero samples.")
    if len(source_indices) < len(target_indices):
        raise ValueError(
            f"Cannot downsample class {source_class} to class {target_class}: "
            f"source has {len(source_indices)} samples, target has {len(target_indices)}."
        )

    rng = np.random.default_rng(random_state)
    kept_source = np.sort(rng.choice(source_indices, size=len(target_indices), replace=False))
    other_indices = valid_indices[(valid_labels != source_class)]
    selected = np.sort(np.concatenate([kept_source, other_indices]).astype(int))

    before_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(valid_labels.astype(int), return_counts=True))
    }
    after_labels = mapped_labels[selected]
    after_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(after_labels.astype(int), return_counts=True))
        if int(label) >= 0
    }

    return selected, {
        "enabled": True,
        "strategy": strategy,
        "source_class": source_class,
        "target_class": target_class,
        "random_state": random_state,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "selected_samples": int(len(selected)),
        "dropped_samples": int(len(valid_indices) - len(selected)),
    }


def _build_graph_subset(
    dataset: SpacioTemporalDataset,
    indices: np.ndarray,
    task_def: Dict[str, Any],
    fold_context: Dict[str, Any],
    class_to_index: Dict[int, int],
    scaler: StandardScaler | None,
) -> List[Any]:
    graphs: List[Any] = []
    for idx in indices:
        graph = dataset[int(idx)].clone()

        names = getattr(graph, "emotion_names", getattr(dataset, "emotion_names", []))
        raw = []
        for column in task_def["target_columns"]:
            if column not in names:
                raise ValueError(f"Graph target column '{column}' not found.")
            col_idx = names.index(column)
            raw.append(float(graph.y[col_idx].item()))

        raw_labels = _raw_to_label(
            task_def=task_def,
            raw_values=np.asarray([raw], dtype=float),
            fold_context=fold_context,
        )
        raw_label = int(raw_labels[0])
        if raw_label < 0 and task_def.get("drop_unmapped_labels", False):
            continue
        if raw_label < 0:
            raise ValueError(
                "Encountered unmapped label in table6 mode with drop_unmapped_labels=false."
            )
        encoded = _encode_labels(np.asarray([raw_label], dtype=int), class_to_index=class_to_index)
        graph.y = torch.tensor(encoded[0], dtype=torch.long)

        if scaler is not None:
            x_scaled = scaler.transform(graph["node"].x.detach().cpu().numpy())
            graph["node"].x = torch.tensor(x_scaled, dtype=torch.float32)

        graphs.append(graph)

    if not graphs:
        raise ValueError(
            "Graph split became empty after label mapping/drop filtering. "
            "Check table6_class_mapping coverage and fold composition."
        )
    return graphs


def _train_gnn_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_max_norm: float,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        targets = batch.y.reshape(-1).long()

        loss = F.cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def _evaluate_gnn(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_labels: List[int],
) -> Tuple[Dict[str, Any], float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0

    all_probs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_subjects: List[Any] = []
    all_recordings: List[Any] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.y.reshape(-1).long()
            loss = F.cross_entropy(logits, targets)
            total_loss += float(loss.item())

            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

            batch_subjects = getattr(batch, "subject", None)
            batch_recordings = getattr(batch, "recording", None)
            if isinstance(batch_subjects, (list, tuple)):
                all_subjects.extend(batch_subjects)
            elif batch_subjects is not None:
                all_subjects.append(batch_subjects)
            if isinstance(batch_recordings, (list, tuple)):
                all_recordings.extend(batch_recordings)
            elif batch_recordings is not None:
                all_recordings.append(batch_recordings)

    y_pred_proba = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy().astype(int)

    metadata = None
    if all_subjects and all_recordings and len(all_subjects) == len(y_true):
        metadata = {"subjects": all_subjects, "recordings": all_recordings}

    metrics = evaluate_multiclass_classification(
        y_pred_proba=y_pred_proba,
        y_true=y_true,
        class_labels=class_labels,
        metadata=metadata,
    )

    avg_loss = total_loss / max(len(loader), 1)
    metrics["standard"]["aggregated"]["loss"] = avg_loss
    if metrics.get("per_pair_aggregated") is not None:
        metrics["per_pair_aggregated"]["aggregated"]["loss"] = avg_loss

    return metrics, avg_loss, y_pred_proba, y_true


def _is_loader_thread_error(exc: RuntimeError) -> bool:
    """Return True for known DataLoader pin-memory/IPC worker failures."""
    message = str(exc)
    return (
        "Pin memory thread exited unexpectedly" in message
        or "received 0 items of ancdata" in message
    )


def _train_gnn_fold(
    config: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: SpacioTemporalDataset,
    fold_dir: str,
    test_name: str,
    device: torch.device,
    task_def: Dict[str, Any],
    fold_context: Dict[str, Any],
    class_to_index: Dict[int, int],
    class_labels: List[int],
    standardize_features: bool,
    verbose: bool,
) -> Dict[str, Any]:
    model_cfg = config["gnn"]["model"]
    training_cfg = config["gnn"]["training"]

    model_version = str(model_cfg.get("model_version", "v2")).lower()
    if model_version == "v1":
        model_cls = MulticlassSpatioTemporalGNNV1
    elif model_version == "v2":
        model_cls = MulticlassSpatioTemporalGNN
    else:
        raise ValueError(f"Unsupported gnn.model.model_version='{model_version}'. Choose 'v1' or 'v2'.")

    model_kwargs = {
        "in_channels": model_cfg["in_channels"],
        "hidden_channels": model_cfg["hidden_channels"],
        "num_classes": len(class_labels),
        "use_preprocess_mlp": model_cfg.get("use_preprocess_mlp", True),
        "use_edge_weights": config["dataset"].get("use_edge_weights", True),
        "add_self_loops": model_cfg.get("add_self_loops", False),
        "dropout_mlp": model_cfg.get("dropout_mlp", 0.1),
        "dropout_gnn": model_cfg.get("dropout_gnn", 0.1),
        "dropout_head": model_cfg.get("dropout_head", 0.1),
        "aggr": model_cfg.get("aggr", "mean"),
        "conv_type": model_cfg.get("conv_type", "GCNConv"),
        "num_layers": model_cfg.get("num_layers", 2),
        "pooling": model_cfg.get("pooling", "attention" if model_version == "v2" else "mean_max"),
    }
    if model_version == "v2":
        model_kwargs["head_pooling"] = model_cfg.get("head_pooling")
        model_kwargs["graph_pooling"] = model_cfg.get(
            "graph_pooling",
            model_cfg.get("head_pooling", model_cfg.get("pooling", "attention")),
        )
        model_kwargs["relation_pooling"] = model_cfg.get("relation_pooling", "mlp")
        model_kwargs["edge_weight_mode"] = model_cfg.get(
            "edge_weight_mode",
            config["dataset"].get("edge_weight_mode", "learned_signed"),
        )
        model_kwargs["use_delta_distance_edge_feature"] = model_cfg.get(
            "use_delta_distance_edge_feature",
            bool(config["dataset"].get("use_delta_distance_edge_feature", True)),
        )
        model_kwargs["use_fixation_edges"] = model_cfg.get(
            "use_fixation_edges",
            bool(config["dataset"].get("use_fixation_edges", True)),
        )

    model = model_cls(**model_kwargs).to(device)

    use_compile = training_cfg.get("use_torch_compile", True)
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="default")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))

    num_workers = int(training_cfg.get("num_workers", 4))
    loader_kwargs = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", True)) if device.type == "cuda" else False,
        "persistent_workers": bool(training_cfg.get("persistent_workers", True)) and num_workers > 0,
    }

    scaler: StandardScaler | None = None
    if standardize_features:
        scaler = _fit_graph_feature_scaler(dataset=dataset, train_idx=train_idx)
        joblib.dump(scaler, os.path.join(fold_dir, "gnn_feature_scaler.pkl"))

    train_graphs = _build_graph_subset(
        dataset=dataset,
        indices=train_idx,
        task_def=task_def,
        fold_context=fold_context,
        class_to_index=class_to_index,
        scaler=scaler,
    )
    val_graphs = _build_graph_subset(
        dataset=dataset,
        indices=val_idx,
        task_def=task_def,
        fold_context=fold_context,
        class_to_index=class_to_index,
        scaler=scaler,
    )
    test_graphs = _build_graph_subset(
        dataset=dataset,
        indices=test_idx,
        task_def=task_def,
        fold_context=fold_context,
        class_to_index=class_to_index,
        scaler=scaler,
    )

    train_loader = DataLoader(train_graphs, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_graphs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_graphs, shuffle=False, **loader_kwargs)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    early_stopped = False
    start_time = time.time()
    history_rows: List[Dict[str, Any]] = []
    early_stopping_enabled = bool(training_cfg.get("early_stopping_enabled", False))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 7))
    early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
    early_stopping_restore_best = bool(training_cfg.get("early_stopping_restore_best", True))

    if early_stopping_enabled and early_stopping_patience < 1:
        raise ValueError("gnn.training.early_stopping_patience must be >= 1 when early stopping is enabled.")

    print(f"Training multiclass GNN for {test_name}...")
    for epoch in range(int(training_cfg["num_epochs"])):
        epoch_start_time = time.time()
        train_loss = _train_gnn_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_max_norm=float(training_cfg.get("grad_clip_max_norm", 1.0)),
        )

        val_metrics, val_loss, _, _ = _evaluate_gnn(
            model=model,
            loader=val_loader,
            device=device,
            class_labels=class_labels,
        )

        val_aggregated = val_metrics["standard"]["aggregated"]
        val_balanced_accuracy = val_aggregated.get("balanced_accuracy", np.nan)
        val_macro_f1 = val_aggregated.get("macro_f1", np.nan)
        epoch_runtime_seconds = round(time.time() - epoch_start_time, 3)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == int(training_cfg["num_epochs"])):
            val_acc = val_aggregated.get("accuracy", np.nan)
            print(
                f"  Epoch {epoch + 1}/{training_cfg['num_epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        if val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))
        else:
            no_improve_epochs += 1

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_balanced_accuracy": float(val_balanced_accuracy),
                "val_macro_f1": float(val_macro_f1),
                "epoch_runtime_seconds": epoch_runtime_seconds,
            }
        )

        if early_stopping_enabled and no_improve_epochs >= early_stopping_patience:
            early_stopped = True
            if verbose:
                print(
                    f"  Early stopping at epoch {epoch + 1}: "
                    f"no val_loss improvement for {early_stopping_patience} epoch(s)."
                )
            break

    for row in history_rows:
        row["is_best_epoch"] = int(int(row["epoch"]) == best_epoch + 1)
        row["best_epoch"] = best_epoch + 1
        row["best_val_loss"] = float(best_val_loss)
    pd.DataFrame(history_rows).to_csv(os.path.join(fold_dir, "gnn_training_history.csv"), index=False)

    print(f"  Best model at epoch {best_epoch + 1}")
    if early_stopped:
        print("  Early stopping triggered.")
    if not early_stopping_restore_best and verbose:
        print(
            "  NOTE: early_stopping_restore_best=false requested, "
            "but evaluation still uses the best checkpoint for comparability."
        )
    print(f"  GNN train time: {time.time() - start_time:.2f} seconds")

    model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pt")))
    test_metrics, _, test_pred, test_true = _evaluate_gnn(
        model=model,
        loader=test_loader,
        device=device,
        class_labels=class_labels,
    )

    np.save(os.path.join(fold_dir, "test_predictions.npy"), test_pred)
    np.save(os.path.join(fold_dir, "test_targets.npy"), test_true)

    if verbose:
        test_acc = test_metrics["standard"]["aggregated"].get("accuracy", np.nan)
        print(f"  ❗GNN - Test Accuracy: {test_acc:.4f}")

    return test_metrics


def _save_mlp_training_history(model: Any, output_path: str) -> None:
    """Save sklearn MLP training loss history when the fitted model exposes it."""
    estimator = getattr(model, "model", None)
    loss_curve = getattr(estimator, "loss_curve_", None)
    if loss_curve is None:
        return

    rows = [
        {
            "epoch": epoch_idx + 1,
            "train_loss": float(loss_value),
        }
        for epoch_idx, loss_value in enumerate(loss_curve)
    ]
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)


def _prepare_baseline_split(
    samples: List[Any],
    indices: np.ndarray,
    feature_columns: List[str],
    task_def: Dict[str, Any],
    fold_context: Dict[str, Any],
    class_to_index: Dict[int, int],
) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]:
    X, y, metadata, _, _ = samples_to_xy(samples, indices)

    selected_cols: List[str] = []
    for col in feature_columns:
        if col in X.columns:
            selected_cols.append(col)
        elif f"{col}_mean" in X.columns:
            selected_cols.append(f"{col}_mean")

    if not selected_cols:
        raise ValueError(
            "Configured feature columns were not found in tabular baseline features. "
            f"Configured: {feature_columns}."
        )

    X = X[selected_cols].copy()

    raw_columns = task_def["target_columns"]
    missing_targets = [col for col in raw_columns if col not in y.columns]
    if missing_targets:
        raise ValueError(f"Missing tabular targets for multiclass: {missing_targets}")

    raw_target = y[raw_columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = (~X.isna().any(axis=1)) & (~raw_target.isna().any(axis=1))

    X = X.loc[valid_mask].reset_index(drop=True)
    raw_target_np = raw_target.loc[valid_mask].to_numpy(dtype=float)
    metadata = [meta for meta, keep in zip(metadata, valid_mask.tolist()) if keep]

    if len(X) == 0:
        raise ValueError("Baseline split became empty after NaN filtering.")

    raw_labels = _raw_to_label(task_def=task_def, raw_values=raw_target_np, fold_context=fold_context)
    if task_def.get("drop_unmapped_labels", False):
        keep_mask = raw_labels >= 0
        X = X.loc[keep_mask].reset_index(drop=True)
        raw_labels = raw_labels[keep_mask]
        metadata = [meta for meta, keep in zip(metadata, keep_mask.tolist()) if keep]
    if len(X) == 0:
        raise ValueError(
            "Baseline split became empty after label mapping/drop filtering. "
            "Check table6_class_mapping coverage and fold composition."
        )

    encoded_labels = _encode_labels(raw_labels=raw_labels, class_to_index=class_to_index)
    return X, encoded_labels, metadata


def _train_baselines_fold(
    baseline_cfg: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    samples: List[Any],
    fold_dir: str,
    task_def: Dict[str, Any],
    fold_context: Dict[str, Any],
    class_to_index: Dict[int, int],
    class_labels: List[int],
    standardize_features: bool,
    feature_columns: List[str],
    verbose: bool,
) -> Dict[str, Dict[str, Any]]:
    baselines_dir = os.path.join(fold_dir, "baselines")
    os.makedirs(baselines_dir, exist_ok=True)

    X_train, y_train, _ = _prepare_baseline_split(
        samples=samples,
        indices=train_idx,
        feature_columns=feature_columns,
        task_def=task_def,
        fold_context=fold_context,
        class_to_index=class_to_index,
    )
    X_val, y_val, _ = _prepare_baseline_split(
        samples=samples,
        indices=val_idx,
        feature_columns=feature_columns,
        task_def=task_def,
        fold_context=fold_context,
        class_to_index=class_to_index,
    )
    X_test, y_test, test_meta = _prepare_baseline_split(
        samples=samples,
        indices=test_idx,
        feature_columns=feature_columns,
        task_def=task_def,
        fold_context=fold_context,
        class_to_index=class_to_index,
    )

    scaler: StandardScaler | None = None
    if standardize_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    test_metadata = {
        "subjects": [meta[0] for meta in test_meta if meta],
        "recordings": [meta[1] for meta in test_meta if meta],
    }

    results: Dict[str, Dict[str, Any]] = {}
    class_labels_array = np.asarray(class_labels, dtype=int)

    for model_name in baseline_cfg["models"]:
        if verbose:
            print(f"  Training {model_name}...")

        model_dir = os.path.join(baselines_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model = get_multiclass_baseline_by_name(
            model_name,
            **baseline_cfg.get("hyperparameters", {}).get(model_name, {}),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Stochastic Optimizer.*")
            model.fit(X_train, y_train)
        if model_name == "MLP":
            _save_mlp_training_history(
                model=model,
                output_path=os.path.join(model_dir, "mlp_training_history.csv"),
            )

        test_metrics = model.evaluate(
            X=X_test,
            y=y_test,
            all_classes=class_labels_array,
            metadata=test_metadata,
        )
        test_pred = model.predict_proba(X_test, all_classes=class_labels_array)

        with open(os.path.join(model_dir, "model.pkl"), "wb") as handle:
            pickle.dump(model, handle)
        if scaler is not None:
            joblib.dump(scaler, os.path.join(model_dir, "feature_scaler.pkl"))

        np.save(os.path.join(model_dir, "test_predictions.npy"), test_pred)
        np.save(os.path.join(model_dir, "test_targets.npy"), y_test)

        results[model_name] = test_metrics

        if verbose:
            test_acc = test_metrics["standard"]["aggregated"].get("accuracy", np.nan)
            print(f"    ❗{model_name} - Test Accuracy: {test_acc:.4f}")

    return results


def run_training_from_config(config_path: str) -> str:
    """Run full multiclass training using one YAML config.

    Returns:
        Path to created training run directory.
    """
    config = load_config(config_path)

    run_experiments = config["run_experiments"]
    dataset_cfg = config["dataset"]
    multiclass_task_cfg = config["multiclass_task"]
    cv_cfg = config["cross_validation"]
    logging_cfg = config["logging"]
    metric_names = list(dict.fromkeys([*config["metrics"], "loss"]))
    verbose = bool(logging_cfg.get("verbose", True))

    task_def = _resolve_task_definition(multiclass_task_cfg, dataset_cfg=dataset_cfg)
    standardize_features = bool(dataset_cfg.get("standardize_features", False))
    target_aggregation = dataset_cfg.get("target_aggregation", "mean")
    if target_aggregation not in {"mean", "last"}:
        raise ValueError(
            f"Unsupported dataset.target_aggregation='{target_aggregation}'. Use 'mean' or 'last'."
        )

    print("Multiclass Task:")
    print(f"  task_name: {task_def['task_name']}")
    print(f"  target_columns: {task_def['target_columns']}")
    print(f"  standardize_features: {standardize_features}")
    print(f"  target_aggregation: {target_aggregation}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name_prefix = str(logging_cfg.get("run_name_prefix", "")).strip()
    run_name = f"{run_name_prefix}_{timestamp}" if run_name_prefix else timestamp
    run_dir = os.path.join(logging_cfg["results_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "training_log.txt")
    logger = Logger(log_file)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger

    try:
        print(f"\n{'#'*50} \nTraining started at: {datetime.now()}")
        print(f"Results will be saved to: {run_dir}")
        print(f"Run baselines: {run_experiments['baselines']}")
        print(f"Run GNN: {run_experiments['gnn']}")

        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        if run_experiments["gnn"]:
            training_cfg = config["gnn"]["training"]
            if training_cfg.get("device", "auto") == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(training_cfg["device"])
            print(f"Using device: {device}")

            if training_cfg.get("random_seed") is not None:
                seed = int(training_cfg["random_seed"])
                torch.manual_seed(seed)
                np.random.seed(seed)

        print("\nLoading datasets...")
        target_columns = task_def["target_columns"]

        feature_columns = resolve_feature_columns(dataset_cfg)
        sync_gnn_in_channels(config["gnn"]["model"], feature_columns)
        dropna_columns = resolve_dropna_columns(dataset_cfg, target_columns=target_columns)
        min_samples_per_window = resolve_min_samples_per_window(dataset_cfg)
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        base_gnn_dataset = None
        if run_experiments["gnn"]:
            print("Loading graph dataset for GNN...")
            base_gnn_dataset = SpacioTemporalDataset(
                **build_graph_dataset_kwargs(
                    dataset_cfg=dataset_cfg,
                    target_columns=target_columns,
                    feature_columns=feature_columns,
                    dropna_columns=dropna_columns,
                ),
            )
            print(f"Loaded {len(base_gnn_dataset)} graph samples")

        base_tabular_samples = None
        if run_experiments["baselines"]:
            print("Loading tabular samples for baselines...")
            base_tabular_samples = build_tabular_samples(
                **build_tabular_samples_kwargs(
                    dataset_cfg=dataset_cfg,
                    target_columns=target_columns,
                    feature_columns=feature_columns,
                    dropna_columns=dropna_columns,
                    min_samples_per_window=min_samples_per_window,
                ),
            )
            print(f"Loaded {len(base_tabular_samples)} tabular samples")

        downsampling_cfg = _resolve_class_downsampling_cfg(multiclass_task_cfg)
        downsampling_metadata: Dict[str, Any] = {
            "enabled": bool(downsampling_cfg.get("enabled", False)),
            "datasets": {},
        }
        if bool(downsampling_cfg.get("enabled", False)):
            print("\nApplying class downsampling before CV split construction...")
            if run_experiments["gnn"] and base_gnn_dataset is not None:
                gnn_all_indices = np.arange(len(base_gnn_dataset))
                gnn_raw_all = _extract_graph_raw_targets(base_gnn_dataset, gnn_all_indices, target_columns)
                gnn_keep_idx, gnn_sampling_info = _select_class_downsample_indices(
                    raw_targets=gnn_raw_all,
                    task_def=task_def,
                    downsampling_cfg=downsampling_cfg,
                )
                base_gnn_dataset = _IndexSubset(base_gnn_dataset, gnn_keep_idx)
                downsampling_metadata["datasets"]["gnn"] = gnn_sampling_info
                print(
                    "  GNN class counts before/after: "
                    f"{gnn_sampling_info['before_counts']} -> {gnn_sampling_info['after_counts']}"
                )

            if run_experiments["baselines"] and base_tabular_samples is not None:
                baseline_all_indices = np.arange(len(base_tabular_samples))
                baseline_raw_all = _extract_tabular_raw_targets(
                    base_tabular_samples,
                    baseline_all_indices,
                    target_columns,
                )
                baseline_keep_idx, baseline_sampling_info = _select_class_downsample_indices(
                    raw_targets=baseline_raw_all,
                    task_def=task_def,
                    downsampling_cfg=downsampling_cfg,
                )
                base_tabular_samples = [base_tabular_samples[int(idx)] for idx in baseline_keep_idx]
                downsampling_metadata["datasets"]["baselines"] = baseline_sampling_info
                print(
                    "  Baseline class counts before/after: "
                    f"{baseline_sampling_info['before_counts']} -> {baseline_sampling_info['after_counts']}"
                )

        # Build class mapping.
        if run_experiments["gnn"] and base_gnn_dataset is not None:
            all_indices = np.arange(len(base_gnn_dataset))
            raw_all = _extract_graph_raw_targets(base_gnn_dataset, all_indices, target_columns)
        elif run_experiments["baselines"] and base_tabular_samples is not None:
            all_indices = np.arange(len(base_tabular_samples))
            raw_all = _extract_tabular_raw_targets(base_tabular_samples, all_indices, target_columns)
        else:
            raise ValueError("At least one experiment path (baselines/gnn) must be enabled.")

        if task_def["mode"] == "emotion-id":
            unique_labels = sorted(np.unique(np.round(raw_all[:, 0]).astype(int)).tolist())
        elif task_def["mode"] == "va-quadrant":
            unique_labels = [0, 1, 2, 3]
        elif task_def["mode"] == "table6-3class":
            mapped = _raw_to_label(
                task_def=task_def,
                raw_values=raw_all,
                fold_context={},
            )
            if task_def.get("drop_unmapped_labels", False):
                mapped = mapped[mapped >= 0]
            if mapped.size == 0:
                raise ValueError(
                    "No mapped labels remain for table6 task. "
                    "Check table6_class_mapping and filtered dataset scope."
                )
            unique_labels = sorted(np.unique(mapped.astype(int)).tolist())
        else:
            raise ValueError(f"Unsupported task mode: {task_def['mode']}")

        class_to_index = {int(label): idx for idx, label in enumerate(unique_labels)}
        class_labels = list(range(len(unique_labels)))
        class_display_names = build_encoded_class_name_mapping(
            unique_raw_labels=unique_labels,
            raw_label_name_mapping=task_def.get("raw_label_names", {}),
        )

        print(f"Class mapping raw->index: {class_to_index}")
        if class_display_names:
            print(f"Class mapping index->name: {class_display_names}")

        class_metadata = {
            "task_name": task_def["task_name"],
            "mode": task_def["mode"],
            "target_columns": task_def["target_columns"],
            "drop_unmapped_labels": bool(task_def.get("drop_unmapped_labels", False)),
            "table6_class_mapping": {
                int(raw): int(mapped)
                for raw, mapped in task_def.get("table6_class_mapping", {}).items()
            },
            "class_to_index": {int(raw): int(index) for raw, index in class_to_index.items()},
            "index_to_raw_label": {int(index): int(raw) for raw, index in class_to_index.items()},
            "index_to_name": {int(index): str(name) for index, name in class_display_names.items()},
            "raw_label_names": {
                int(raw): str(name) for raw, name in task_def.get("raw_label_names", {}).items()
            },
            "class_downsampling": downsampling_metadata,
        }
        with open(os.path.join(run_dir, "class_metadata.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(class_metadata, handle, sort_keys=True)

        strategies = cv_cfg["strategies"]
        if isinstance(strategies, str):
            strategies = [strategies]

        print(f"\nWill run {len(strategies)} strategy(ies): {', '.join(strategies)}")

        all_strategies_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for strategy in strategies:
            print("\n" + "=" * 100)
            print(f"Starting CV strategy: {strategy.upper()}")
            print("=" * 100)

            strategy_dir = os.path.join(run_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)

            baseline_splitter = None
            gnn_splitter = None
            if run_experiments["baselines"] and base_tabular_samples is not None:
                baseline_splitter = create_splitter(
                    strategy=strategy,
                    samples=base_tabular_samples,
                    val_size=int(cv_cfg["val_size"]),
                    random_state=cv_cfg.get("random_state"),
                    n_splits=int(cv_cfg.get("n_splits", 3)),
                )
            if run_experiments["gnn"] and base_gnn_dataset is not None:
                gnn_splitter = create_splitter(
                    strategy=strategy,
                    samples=base_gnn_dataset,
                    val_size=int(cv_cfg["val_size"]),
                    random_state=cv_cfg.get("random_state"),
                    n_splits=int(cv_cfg.get("n_splits", 3)),
                )

            reference_splitter = gnn_splitter if gnn_splitter is not None else baseline_splitter
            reference_dataset = base_gnn_dataset if gnn_splitter is not None else base_tabular_samples
            if reference_splitter is None or reference_dataset is None:
                raise ValueError("No splitter/dataset available for CV.")

            baseline_results_all_folds = (
                {name: {} for name in config["baselines"]["models"]}
                if run_experiments["baselines"]
                else {}
            )
            gnn_results_all_folds: Dict[str, Dict[str, Any]] = {}

            if baseline_splitter is not None and gnn_splitter is not None:
                baseline_splits = list(baseline_splitter.split())
                gnn_splits = list(gnn_splitter.split())
                _validate_non_empty_train_splits(baseline_splits, strategy, "Baseline")
                _validate_non_empty_train_splits(gnn_splits, strategy, "GNN")
                _validate_kfold_group_disjointness(
                    baseline_splits,
                    strategy=strategy,
                    dataset=base_tabular_samples,
                    dataset_label="Baseline",
                )
                _validate_kfold_group_disjointness(
                    gnn_splits,
                    strategy=strategy,
                    dataset=base_gnn_dataset,
                    dataset_label="GNN",
                )
                num_folds = len(baseline_splits)
            elif baseline_splitter is not None:
                baseline_splits = list(baseline_splitter.split())
                _validate_non_empty_train_splits(baseline_splits, strategy, "Baseline")
                _validate_kfold_group_disjointness(
                    baseline_splits,
                    strategy=strategy,
                    dataset=base_tabular_samples,
                    dataset_label="Baseline",
                )
                num_folds = len(baseline_splits)
            else:
                gnn_splits = list(gnn_splitter.split())
                _validate_non_empty_train_splits(gnn_splits, strategy, "GNN")
                _validate_kfold_group_disjointness(
                    gnn_splits,
                    strategy=strategy,
                    dataset=base_gnn_dataset,
                    dataset_label="GNN",
                )
                num_folds = len(gnn_splits)

            for fold_num in range(num_folds):
                if baseline_splitter is not None:
                    baseline_train_idx, baseline_val_idx, baseline_test_idx = baseline_splits[fold_num]
                if gnn_splitter is not None:
                    gnn_train_idx, gnn_val_idx, gnn_test_idx = gnn_splits[fold_num]

                ref_test_idx = gnn_test_idx if gnn_splitter is not None else baseline_test_idx
                fold = describe_fold(
                    strategy=strategy,
                    dataset=reference_dataset,
                    test_idx=ref_test_idx,
                    fold_num=fold_num,
                    combined_id_style="underscore",
                )
                test_id = fold.test_id
                test_name = fold.test_name

                fold_dir = os.path.join(strategy_dir, test_id)
                os.makedirs(fold_dir, exist_ok=True)

                print(f"\n{test_name}")

                if gnn_splitter is not None and base_gnn_dataset is not None:
                    train_raw = _extract_graph_raw_targets(base_gnn_dataset, gnn_train_idx, target_columns)
                else:
                    train_raw = _extract_tabular_raw_targets(base_tabular_samples, baseline_train_idx, target_columns)

                fold_context = _resolve_fold_context(task_def=task_def, train_raw_targets=train_raw)
                if task_def["mode"] == "va-quadrant":
                    print(
                        "  Fold VA thresholds: "
                        f"valence={fold_context['valence_threshold']:.6f}, "
                        f"arousal={fold_context['arousal_threshold']:.6f}"
                    )

                if run_experiments["baselines"] and base_tabular_samples is not None:
                    print("Training baselines...")
                    baseline_results = _train_baselines_fold(
                        baseline_cfg=config["baselines"],
                        train_idx=baseline_train_idx,
                        val_idx=baseline_val_idx,
                        test_idx=baseline_test_idx,
                        samples=base_tabular_samples,
                        fold_dir=fold_dir,
                        task_def=task_def,
                        fold_context=fold_context,
                        class_to_index=class_to_index,
                        class_labels=class_labels,
                        standardize_features=standardize_features,
                        feature_columns=feature_columns,
                        verbose=verbose,
                    )
                    for model_name, metrics in baseline_results.items():
                        baseline_results_all_folds[model_name][test_id] = metrics

                if run_experiments["gnn"] and base_gnn_dataset is not None:
                    try:
                        gnn_metrics = _train_gnn_fold(
                            config=config,
                            train_idx=gnn_train_idx,
                            val_idx=gnn_val_idx,
                            test_idx=gnn_test_idx,
                            dataset=base_gnn_dataset,
                            fold_dir=fold_dir,
                            test_name=test_name,
                            device=device,
                            task_def=task_def,
                            fold_context=fold_context,
                            class_to_index=class_to_index,
                            class_labels=class_labels,
                            standardize_features=standardize_features,
                            verbose=verbose,
                        )
                    except RuntimeError as exc:
                        if not _is_loader_thread_error(exc):
                            raise
                        training_cfg = config["gnn"]["training"]
                        training_cfg["num_workers"] = 0
                        training_cfg["pin_memory"] = False
                        training_cfg["persistent_workers"] = False
                        print(
                            "  Warning: DataLoader pin-memory/multiprocessing failed "
                            f"({exc}). Retrying fold with num_workers=0, "
                            "pin_memory=false, persistent_workers=false and applying "
                            "those settings for subsequent folds."
                        )
                        gnn_metrics = _train_gnn_fold(
                            config=config,
                            train_idx=gnn_train_idx,
                            val_idx=gnn_val_idx,
                            test_idx=gnn_test_idx,
                            dataset=base_gnn_dataset,
                            fold_dir=fold_dir,
                            test_name=f"{test_name} [safe-loader mode]",
                            device=device,
                            task_def=task_def,
                            fold_context=fold_context,
                            class_to_index=class_to_index,
                            class_labels=class_labels,
                            standardize_features=standardize_features,
                            verbose=verbose,
                        )
                    gnn_results_all_folds[test_id] = gnn_metrics

            combined_results: Dict[str, Dict[str, Any]] = {}
            if run_experiments["baselines"]:
                combined_results.update(baseline_results_all_folds)
            if run_experiments["gnn"]:
                combined_results["GNN"] = gnn_results_all_folds

            all_strategies_results[strategy] = combined_results

            print_comparison_table(combined_results, metric_names, strategy)
            save_comparison_csv(combined_results, metric_names, os.path.join(strategy_dir, "summary.csv"))
            save_fold_metrics_csv(combined_results, metric_names, os.path.join(strategy_dir, "fold_metrics.csv"))

        print("\nGenerating multiclass result plots...")
        models_for_cm: List[str] = []
        if run_experiments["gnn"]:
            models_for_cm.append("GNN")
        if run_experiments["baselines"]:
            models_for_cm.extend(config["baselines"]["models"])
        models_for_cm = list(dict.fromkeys(models_for_cm))

        try:
            saved = generate_and_save_multiclass_results_plots(
                run_dir=Path(run_dir),
                models_for_cm=models_for_cm if models_for_cm else None,
                class_display_names=class_display_names,
            )
            for path in saved:
                print(f"Saved plot: {path}")
        except Exception as exc:  # pragma: no cover - plotting should not fail full run
            print(f"Warning: failed to generate multiclass plots: {exc}")

        print("\n" + "=" * 100)
        print("Multiclass training complete!")
        print(f"All results saved to: {run_dir}")
        return run_dir

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.close()


def main() -> None:
    args = parse_args()
    run_dir = run_training_from_config(args.config)
    print(f"Multiclass run directory: {run_dir}")


if __name__ == "__main__":
    main()
