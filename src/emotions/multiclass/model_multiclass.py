"""Multiclass GNN model for emotion classification tasks."""

from __future__ import annotations

from emotions.model import BasicGCN, SpatioTemporalHeteroGNN, SpatioTemporalHeteroGNNV1


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
        use_delta_distance_edge_feature: bool = True,
        use_fixation_edges: bool = True,
        spatial_edge_attr_dim: int | None = None,
        temporal_edge_attr_dim: int | None = None,
        fixation_edge_attr_dim: int | None = None,
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
            spatial_edge_attr_dim=spatial_edge_attr_dim,
            temporal_edge_attr_dim=temporal_edge_attr_dim,
            fixation_edge_attr_dim=fixation_edge_attr_dim,
        )


class MulticlassBasicGCN(BasicGCN):
    """BasicGCN configured for multiclass logits output."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        use_preprocess_mlp: bool = True,
        use_edge_weights: bool = False,
        add_self_loops: bool = False,
        dropout_mlp: float = 0.1,
        dropout_gnn: float = 0.1,
        dropout_head: float = 0.1,
        conv_type: str = "GCNConv",
        num_layers: int = 2,
        readout: str = "attention",
        pooling: str | None = None,
        graph_pooling: str | None = None,
        head_pooling: str | None = None,
        **kwargs: object,
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
            conv_type=conv_type,
            num_layers=num_layers,
            readout=readout,
            pooling=pooling,
            graph_pooling=graph_pooling,
            head_pooling=head_pooling,
            **kwargs,
        )
