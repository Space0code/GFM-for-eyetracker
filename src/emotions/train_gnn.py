"""
GNN-specific training functions for emotion prediction.

This module contains training, evaluation, and fold-level logic for
SpatioTemporalHeteroGNN models. Use train.py as the main script.
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import os

from emotions.model import SpatioTemporalHeteroGNN
from emotions.metrics import compute_metrics


def train_epoch(model, loader, optimizer, device, grad_clip_max_norm=1.0):
    """Train model for one epoch.
    
    Args:
        model: GNN model to train
        loader: DataLoader with training data
        optimizer: Optimizer
        device: Device to train on
        grad_clip_max_norm: Maximum gradient norm for clipping
        
    Returns:
        Average loss over epoch
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        target = data.y.view(-1, 4)
        
        loss = F.mse_loss(out, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device, emotion_names=None, save_outputs=False, 
            save_dir=None, pair_aggregation_fn=np.mean):
    """Evaluate the model and compute comprehensive metrics.

    Args:
        model: GNN model to evaluate
        loader: DataLoader with test data
        device: Device to run evaluation on
        emotion_names: List of emotion names (optional)
        save_outputs: Whether to save outputs
        save_dir: Directory to save outputs
        pair_aggregation_fn: Function to aggregate per-pair metrics (default: np.mean)

    Returns:
        Dictionary with 'standard' and 'per_pair_aggregated' metrics
    """
    model.eval()
    total_loss = 0

    all_outputs = []
    all_targets = []
    all_metadata = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            target = data.y.view(-1, 4)

            loss = F.mse_loss(out, target)
            total_loss += loss.item()

            all_outputs.append(out.cpu())
            all_targets.append(target.cpu())

            # Collect metadata (subject, recording) for each sample in batch
            if hasattr(data, 'idx'):
                batch_indices = data.idx.cpu().numpy().flatten()
                for graph_idx in batch_indices:
                    try:
                        original_graph = loader.dataset[int(graph_idx)]
                        if hasattr(original_graph, 'subject') and hasattr(original_graph, 'recording'):
                            all_metadata.append((original_graph.subject, original_graph.recording))
                        else:
                            all_metadata.append(None)
                    except (AttributeError, IndexError, KeyError):
                        all_metadata.append(None)

    # Concatenate all outputs and targets
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Validate metadata
    if len(all_metadata) == 0:
        all_metadata = None
    elif len(all_metadata) != len(outputs):
        print(f"Warning: metadata length ({len(all_metadata)}) != outputs length ({len(outputs)}). Using None.")
        all_metadata = None
    elif any(m is None for m in all_metadata):
        all_metadata = None

    # Compute comprehensive metrics
    metrics = compute_metrics(
        outputs,
        targets,
        emotion_names=emotion_names,
        metadata=all_metadata,
        pair_aggregation_fn=pair_aggregation_fn
    )

    # Add loss to metrics
    avg_loss = total_loss / len(loader)
    metrics['standard']['aggregated']['loss'] = avg_loss
    if metrics['per_pair_aggregated'] is not None:
        metrics['per_pair_aggregated']['aggregated']['loss'] = avg_loss

    if save_outputs and save_dir:
        torch.save({
            'outputs': outputs,
            'targets': targets,
            'metadata': all_metadata,
            'metrics': metrics
        }, save_dir)

    return metrics


def train_gnn_fold(config: dict, train_idx: np.ndarray, val_idx: np.ndarray, 
                   test_idx: np.ndarray, dataset, fold_dir: str, 
                   test_name: str, device: torch.device) -> dict:
    """Train GNN model for one cross-validation fold.
    
    Args:
        config: Full configuration dictionary
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        dataset: Graph dataset
        fold_dir: Directory to save fold results
        test_name: Name of test fold for logging
        device: Device to train on
    
    Returns:
        Dictionary of test metrics
    """
    gnn_cfg = config['gnn']
    training_cfg = gnn_cfg['training']
    model_cfg = gnn_cfg['model']
    logging_cfg = config['logging']
    
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    emotion_names = dataset.emotion_names if hasattr(dataset, 'emotion_names') else None
    
    train_loader = DataLoader(train_dataset, batch_size=training_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=training_cfg['batch_size'], shuffle=False)
    
    model = SpatioTemporalHeteroGNN(
        in_channels=model_cfg['in_channels'],
        hidden_channels=model_cfg['hidden_channels'],
        out_channels=model_cfg['out_channels'],
        output_scale=model_cfg.get('output_scale', 10.0),
        use_preprocess_mlp=model_cfg.get('use_preprocess_mlp', True),
        add_self_loops=model_cfg.get('add_self_loops', False),
        dropout_mlp=model_cfg.get('dropout_mlp', 0.1),
        dropout_gnn=model_cfg.get('dropout_gnn', 0.1),
        dropout_head=model_cfg.get('dropout_head', 0.1),
        aggr=model_cfg.get('aggr', 'mean'),
        conv_type=model_cfg.get('conv_type', 'GCNConv')
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])
    
    num_epochs = training_cfg['num_epochs']
    best_val_loss = float('inf')
    
    save_interval = logging_cfg.get('save_outputs_interval', 10)
    save_epochs = set(range(save_interval, num_epochs + 1, save_interval))
    
    gnn_fold_dir = os.path.join(fold_dir, 'gnn')
    os.makedirs(gnn_fold_dir, exist_ok=True)
    
    print(f"Training GNN for {test_name}...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, 
                                training_cfg['grad_clip_max_norm'])
        val_metrics = evaluate(model, val_loader, device, emotion_names=emotion_names)
        val_loss = val_metrics['standard']['aggregated']['loss']
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(gnn_fold_dir, 'best_model.pt'))
        
        if epoch in save_epochs or epoch == num_epochs or epoch == 1:
            print(f"  Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(os.path.join(gnn_fold_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device, emotion_names=emotion_names)
    
    # Log test metrics
    test_mae = test_metrics['standard']['aggregated']['mae']
    test_mse = test_metrics['standard']['aggregated']['mse']
    print(f" GNN - Test MAE: {test_mae:.4f} | Test MSE: {test_mse:.4f}")
    
    # Clean up
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return test_metrics
