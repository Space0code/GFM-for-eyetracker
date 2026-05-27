"""Shared training diagnostics for neural emotion models.

The helpers in this module keep fold-level GNN diagnostics out of the task
trainers. They are intentionally read-only with respect to model selection:
test split diagnostics are collected only by callers after the best checkpoint
has already been restored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader


EDGE_WEIGHT_RELATIONS: tuple[str, ...] = (
    "temporal_forward",
    "temporal_backward",
    "spatial",
    "fixation",
)


@dataclass(frozen=True)
class GradientNormStats:
    """Per-epoch gradient norm summary before optional clipping."""

    mean: float
    max: float


def _nan_stats(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float("nan"),
        f"{prefix}_std": float("nan"),
        f"{prefix}_min": float("nan"),
        f"{prefix}_max": float("nan"),
    }


def collect_gradient_norm_stats(parameters: Any) -> GradientNormStats:
    """Return mean and max parameter-gradient L2 norms without modifying gradients."""
    norms: list[float] = []
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        norm = float(grad.detach().data.norm(2).item())
        if np.isfinite(norm):
            norms.append(norm)

    if not norms:
        return GradientNormStats(mean=float("nan"), max=float("nan"))
    return GradientNormStats(mean=float(np.mean(norms)), max=float(np.max(norms)))


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the original module when torch.compile wrapped the model."""
    return getattr(model, "_orig_mod", model)


def _tensor_summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return _nan_stats(prefix)
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _mean_pairwise_cosine_similarity(
    embeddings: torch.Tensor,
    *,
    max_exact_embeddings: int = 2000,
    max_sampled_pairs: int = 10000,
) -> tuple[float, int]:
    """Compute or deterministically estimate mean off-diagonal cosine similarity."""
    embeddings = embeddings.detach().float().cpu()
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    embeddings = embeddings[torch.isfinite(embeddings).all(dim=1)]
    n_embeddings = int(embeddings.shape[0])
    if n_embeddings < 2:
        return float("nan"), 0

    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    if n_embeddings <= max_exact_embeddings:
        cosine = normalized @ normalized.T
        upper = torch.triu_indices(n_embeddings, n_embeddings, offset=1)
        values = cosine[upper[0], upper[1]]
        return float(values.mean().item()), int(values.numel())

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    first = torch.randint(0, n_embeddings, (max_sampled_pairs,), generator=generator)
    second = torch.randint(0, n_embeddings - 1, (max_sampled_pairs,), generator=generator)
    second = second + (second >= first).long()
    values = (normalized[first] * normalized[second]).sum(dim=1)
    return float(values.mean().item()), int(values.numel())


def _classification_probabilities(logits: torch.Tensor, task_kind: str) -> torch.Tensor:
    """Convert logits to probabilities for entropy diagnostics."""
    if task_kind == "binary":
        positive = torch.sigmoid(logits.reshape(-1))
        return torch.stack([1.0 - positive, positive], dim=1)
    if task_kind == "multiclass":
        return torch.softmax(logits, dim=1)
    raise ValueError(f"Unsupported task_kind='{task_kind}'. Choose 'binary' or 'multiclass'.")


def _prediction_entropy(probabilities: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = probabilities.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=1)


def _forward_with_optional_embedding(
    model: torch.nn.Module,
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        output = model(batch, return_graph_embedding=True)
    except TypeError:
        output = model(batch)

    if isinstance(output, tuple) and len(output) == 2:
        logits, graph_embedding = output
        return logits, graph_embedding
    return output, None


def _collect_relation_edge_weights(
    model: torch.nn.Module,
    batch: Any,
) -> dict[str, list[torch.Tensor]]:
    """Collect learned normalized edge weights for the proposed hetero GNN."""
    base_model = unwrap_compiled_model(model)
    if getattr(base_model, "edge_weight_mode", None) != "learned_signed":
        return {}
    if not bool(getattr(base_model, "use_edge_weights", False)):
        return {}
    if not bool(getattr(base_model, "supports_scalar_edge_weights", False)):
        return {}
    if not hasattr(base_model, "_edge_weight_from_attr"):
        return {}

    edge_index_dict = getattr(batch, "edge_index_dict", {})
    relations = set(getattr(base_model, "relations", ()))
    collected: dict[str, list[torch.Tensor]] = {relation: [] for relation in EDGE_WEIGHT_RELATIONS}
    num_nodes = int(batch["node"].num_nodes)

    for relation in EDGE_WEIGHT_RELATIONS:
        if relation not in relations:
            continue
        edge_type = ("node", relation, "node")
        edge_index = edge_index_dict.get(edge_type)
        if edge_index is None or edge_index.numel() == 0:
            continue
        edge_store = batch[edge_type]
        edge_attr = getattr(edge_store, "edge_attr", None)
        if edge_attr is None:
            continue
        weights = base_model._edge_weight_from_attr(
            relation=relation,
            edge_attr=edge_attr,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        if weights is not None and weights.numel() > 0:
            collected[relation].append(weights.detach().reshape(-1).cpu())

    return collected


def collect_gnn_loader_diagnostics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    split: str,
    task_kind: str,
) -> dict[str, Any]:
    """Summarize GNN embeddings, logits, probabilities, and learned edge weights."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    all_embeddings: list[torch.Tensor] = []
    edge_weights: dict[str, list[torch.Tensor]] = {relation: [] for relation in EDGE_WEIGHT_RELATIONS}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, graph_embedding = _forward_with_optional_embedding(model, batch)
            logits = logits.detach()
            probabilities = _classification_probabilities(logits=logits, task_kind=task_kind)
            all_logits.append(logits.reshape(logits.shape[0], -1).cpu())
            all_probabilities.append(probabilities.cpu())
            if isinstance(graph_embedding, torch.Tensor):
                embedding = graph_embedding.detach().cpu()
                if embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)
                all_embeddings.append(embedding)

            for relation, values in _collect_relation_edge_weights(model=model, batch=batch).items():
                edge_weights.setdefault(relation, []).extend(values)

    row: dict[str, Any] = {"split": split}
    if all_logits:
        logits_tensor = torch.cat(all_logits, dim=0)
        row.update(_tensor_summary(logits_tensor, "logit"))
        row["logit_range"] = float(row["logit_max"] - row["logit_min"])
    else:
        row.update(_nan_stats("logit"))
        row["logit_range"] = float("nan")

    if all_probabilities:
        entropy = _prediction_entropy(torch.cat(all_probabilities, dim=0))
        row.update(_tensor_summary(entropy, "prediction_entropy"))
    else:
        row.update(_nan_stats("prediction_entropy"))

    if all_embeddings:
        embeddings = torch.cat(all_embeddings, dim=0)
        row["graph_embedding_variance"] = (
            float(embeddings.float().var(dim=0, unbiased=False).mean().item())
            if embeddings.shape[0] > 0
            else float("nan")
        )
        mean_cosine, pair_count = _mean_pairwise_cosine_similarity(embeddings)
        row["graph_embedding_mean_pairwise_cosine_similarity"] = mean_cosine
        row["graph_embedding_pair_count"] = int(pair_count)
    else:
        row["graph_embedding_variance"] = float("nan")
        row["graph_embedding_mean_pairwise_cosine_similarity"] = float("nan")
        row["graph_embedding_pair_count"] = 0

    for relation in EDGE_WEIGHT_RELATIONS:
        values = edge_weights.get(relation, [])
        prefix = f"{relation}_edge_weight"
        if values:
            row.update(_tensor_summary(torch.cat(values), prefix))
        else:
            row.update(_nan_stats(prefix))

    return row


def save_gnn_fold_diagnostics(
    model: torch.nn.Module,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    output_path: str | Path,
    *,
    task_kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Save split-level GNN diagnostics for a trained fold."""
    rows: list[dict[str, Any]] = []
    metadata_dict = dict(metadata or {})
    for split, loader in loaders.items():
        row = collect_gnn_loader_diagnostics(
            model=model,
            loader=loader,
            device=device,
            split=str(split),
            task_kind=task_kind,
        )
        row.update(metadata_dict)
        rows.append(row)

    diagnostics = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_path, index=False)
    return diagnostics
