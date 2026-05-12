"""Binary classification GNN model for emotion recognition."""

from emotions.model import SpatioTemporalHeteroGNN, SpatioTemporalHeteroGNNV1


class BinarySpatioTemporalGNNV1(SpatioTemporalHeteroGNNV1):
    """Frozen v1 binary classification GNN for comparison runs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
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
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
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


class BinarySpatioTemporalGNN(SpatioTemporalHeteroGNN):
    """
    Binary classification variant of SpatioTemporalHeteroGNN.
    
    Overrides forward method to output single probability without scaling.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
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
    ):
        """
        Initialize binary classification GNN.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden layer dimension
            use_preprocess_mlp: Whether to use preprocessing MLP
            add_self_loops: Whether to add self-loops in convolutions
            dropout_mlp: Dropout rate for preprocessing MLP
            dropout_gnn: Dropout rate for GNN layers
            dropout_head: Dropout rate for prediction head
            aggr: Aggregation method for hetero edges
            conv_type: Type of convolutional layer (GCNConv or GATConv)
            num_layers: Number of GNN message-passing layers
            pooling: Graph pooling type ('mean' or 'mean_max')
        """
        # Initialize parent with out_channels=1 and output_scale=1.0 (no scaling)
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,  # Single binary output
            output_scale=1.0,  # No scaling for binary classification
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
        )
    
    def forward(self, data, return_graph_embedding: bool = False):
        """
        Forward pass for binary classification.
        
        Args:
            data: HeteroData graph batch
            
        Returns:
            Tensor of shape [batch_size, 1] with logits.
            When `return_graph_embedding=True`, returns tuple:
            `(logits, graph_embedding)`.
        """
        # Parent returns raw regression-style head outputs; binary trainer
        # applies BCE-with-logits and sigmoid for metrics.
        return super().forward(data, return_graph_embedding=return_graph_embedding)
