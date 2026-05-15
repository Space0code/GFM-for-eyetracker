"""Fold-safe scaling utilities for learned GNN edge attributes."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


EdgeScalerDict = Dict[str, StandardScaler]

TEMPORAL_EDGE_TYPES = [
    ("node", "temporal_forward", "node"),
    ("node", "temporal_backward", "node"),
]
SPATIAL_EDGE_TYPE = ("node", "spatial", "node")
FIXATION_EDGE_TYPE = ("node", "fixation", "node")


def _get_edge_attr(graph: Any, edge_type: tuple[str, str, str]) -> torch.Tensor | None:
    """Return edge attributes for an edge type when present."""
    if edge_type not in getattr(graph, "edge_types", []):
        return None
    store = graph[edge_type]
    if not hasattr(store, "edge_attr"):
        return None
    edge_attr = store.edge_attr
    if edge_attr is None or edge_attr.ndim != 2 or edge_attr.shape[1] <= 1:
        return None
    return edge_attr


def _iter_nonempty_attrs(graphs: Iterable[Any], edge_types: Iterable[tuple[str, str, str]]) -> Iterable[np.ndarray]:
    """Yield non-empty learned edge-attribute matrices for the requested edge types."""
    for graph in graphs:
        for edge_type in edge_types:
            edge_attr = _get_edge_attr(graph, edge_type)
            if edge_attr is None or edge_attr.shape[0] == 0:
                continue
            yield edge_attr.detach().cpu().numpy()


def fit_edge_feature_scalers(dataset: Any, train_idx: np.ndarray) -> EdgeScalerDict:
    """Fit relation-family StandardScalers on train graph edge attributes only."""
    train_graphs = [dataset[int(idx)] for idx in train_idx]
    scalers: EdgeScalerDict = {}

    temporal_arrays = []
    for array in _iter_nonempty_attrs(train_graphs, TEMPORAL_EDGE_TYPES):
        temporal_arrays.append(array[:, :-1])
    if temporal_arrays:
        scaler = StandardScaler()
        scaler.fit(np.vstack(temporal_arrays))
        scalers["temporal"] = scaler

    for relation_name, edge_type in (
        ("spatial", SPATIAL_EDGE_TYPE),
        ("fixation", FIXATION_EDGE_TYPE),
    ):
        arrays = list(_iter_nonempty_attrs(train_graphs, [edge_type]))
        if not arrays:
            continue
        scaler = StandardScaler()
        scaler.fit(np.vstack(arrays))
        scalers[relation_name] = scaler

    return scalers


def apply_edge_feature_scalers(graph: Any, scalers: EdgeScalerDict | None) -> None:
    """Apply fitted relation-family edge scalers to one graph in place."""
    if not scalers:
        return

    temporal_scaler = scalers.get("temporal")
    if temporal_scaler is not None:
        for edge_type in TEMPORAL_EDGE_TYPES:
            edge_attr = _get_edge_attr(graph, edge_type)
            if edge_attr is None or edge_attr.shape[0] == 0:
                continue
            scaled = edge_attr.clone()
            scaled_values = temporal_scaler.transform(edge_attr[:, :-1].detach().cpu().numpy())
            scaled[:, :-1] = torch.tensor(scaled_values, dtype=edge_attr.dtype, device=edge_attr.device)
            graph[edge_type].edge_attr = scaled

    for relation_name, edge_type in (
        ("spatial", SPATIAL_EDGE_TYPE),
        ("fixation", FIXATION_EDGE_TYPE),
    ):
        scaler = scalers.get(relation_name)
        edge_attr = _get_edge_attr(graph, edge_type)
        if scaler is None or edge_attr is None or edge_attr.shape[0] == 0:
            continue
        scaled_values = scaler.transform(edge_attr.detach().cpu().numpy())
        graph[edge_type].edge_attr = torch.tensor(scaled_values, dtype=edge_attr.dtype, device=edge_attr.device)
