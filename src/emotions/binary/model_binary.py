"""Binary GNN model wrappers for emotion classification tasks."""

from __future__ import annotations

from emotions.model import BasicGCN, HeteroGCNMean, HeteroGCNMLP, HeteroGCNMLPWeights


class BinaryBasicGCN(BasicGCN):
    """BasicGCN configured for one binary logit."""

    def __init__(self, in_channels: int, hidden_channels: int, **kwargs: object) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            output_scale=1.0,
            **kwargs,
        )


class BinaryHeteroGCNMean(HeteroGCNMean):
    """HeteroGCNMean configured for one binary logit."""

    def __init__(self, in_channels: int, hidden_channels: int, **kwargs: object) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            output_scale=1.0,
            **kwargs,
        )


class BinaryHeteroGCNMLP(HeteroGCNMLP):
    """HeteroGCNMLP configured for one binary logit."""

    def __init__(self, in_channels: int, hidden_channels: int, **kwargs: object) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            output_scale=1.0,
            **kwargs,
        )


class BinaryHeteroGCNMLPWeights(HeteroGCNMLPWeights):
    """HeteroGCNMLPWeights configured for one binary logit."""

    def __init__(self, in_channels: int, hidden_channels: int, **kwargs: object) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            output_scale=1.0,
            **kwargs,
        )
