"""Multiclass GNN model wrappers for emotion classification tasks."""

from __future__ import annotations

from emotions.model import BasicGCN, HeteroGCNMean, HeteroGCNMLP, HeteroGCNMLPWeights


class MulticlassBasicGCN(BasicGCN):
    """BasicGCN configured for multiclass logits output."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, **kwargs: object) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            output_scale=1.0,
            **kwargs,
        )


class MulticlassHeteroGCNMean(HeteroGCNMean):
    """HeteroGCNMean configured for multiclass logits output."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, **kwargs: object) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            output_scale=1.0,
            **kwargs,
        )


class MulticlassHeteroGCNMLP(HeteroGCNMLP):
    """HeteroGCNMLP configured for multiclass logits output."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, **kwargs: object) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            output_scale=1.0,
            **kwargs,
        )


class MulticlassHeteroGCNMLPWeights(HeteroGCNMLPWeights):
    """HeteroGCNMLPWeights configured for multiclass logits output."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, **kwargs: object) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            output_scale=1.0,
            **kwargs,
        )
