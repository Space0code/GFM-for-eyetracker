"""Multiclass GNN model for emotion classification tasks."""

from __future__ import annotations

from emotions.model import SpatioTemporalHeteroGNN


class MulticlassSpatioTemporalGNN(SpatioTemporalHeteroGNN):
    """SpatioTemporalHeteroGNN configured for multiclass logits output."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        use_preprocess_mlp: bool = True,
        add_self_loops: bool = False,
        dropout_mlp: float = 0.1,
        dropout_gnn: float = 0.1,
        dropout_head: float = 0.1,
        aggr: str = "mean",
        conv_type: str = "GCNConv",
        num_layers: int = 2,
        pooling: str = "mean_max",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            output_scale=1.0,
            use_preprocess_mlp=use_preprocess_mlp,
            add_self_loops=add_self_loops,
            dropout_mlp=dropout_mlp,
            dropout_gnn=dropout_gnn,
            dropout_head=dropout_head,
            aggr=aggr,
            conv_type=conv_type,
            num_layers=num_layers,
            pooling=pooling,
        )
