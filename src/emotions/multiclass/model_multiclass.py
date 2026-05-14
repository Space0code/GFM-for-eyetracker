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
        graph_pooling: str | None = None,
        head_pooling: str | None = None,
        relation_pooling: str = "mlp",
        edge_weight_mode: str = "learned_signed",
        use_delta_distance_edge_feature: bool = False,
        use_fixation_edges: bool = False,
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
            graph_pooling=graph_pooling,
            head_pooling=head_pooling,
            relation_pooling=relation_pooling,
            edge_weight_mode=edge_weight_mode,
            use_delta_distance_edge_feature=use_delta_distance_edge_feature,
            use_fixation_edges=use_fixation_edges,
        )
