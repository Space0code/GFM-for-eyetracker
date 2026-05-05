"""Multiclass GNN model for emotion classification tasks."""

from __future__ import annotations

from emotions.model import SpatioTemporalHeteroGNN, SpatioTemporalHeteroGNNV1


class MulticlassSpatioTemporalGNNV1(SpatioTemporalHeteroGNNV1):
    """Frozen v1 multiclass GNN for comparison runs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        use_preprocess_mlp: bool = True,
        use_edge_weights: bool = True,
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
            use_edge_weights=use_edge_weights,
            add_self_loops=add_self_loops,
            dropout_mlp=dropout_mlp,
            dropout_gnn=dropout_gnn,
            dropout_head=dropout_head,
            aggr=aggr,
            conv_type=conv_type,
            num_layers=num_layers,
            pooling=pooling,
        )


class MulticlassSpatioTemporalGNN(SpatioTemporalHeteroGNN):
    """SpatioTemporalHeteroGNN configured for multiclass logits output."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        use_preprocess_mlp: bool = True,
        use_edge_weights: bool = True,
        add_self_loops: bool = False,
        dropout_mlp: float = 0.1,
        dropout_gnn: float = 0.1,
        dropout_head: float = 0.1,
        aggr: str = "mean",
        conv_type: str = "GCNConv",
        num_layers: int = 2,
        pooling: str = "attention",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            output_scale=1.0,
            use_preprocess_mlp=use_preprocess_mlp,
            use_edge_weights=use_edge_weights,
            add_self_loops=add_self_loops,
            dropout_mlp=dropout_mlp,
            dropout_gnn=dropout_gnn,
            dropout_head=dropout_head,
            aggr=aggr,
            conv_type=conv_type,
            num_layers=num_layers,
            pooling=pooling,
        )
