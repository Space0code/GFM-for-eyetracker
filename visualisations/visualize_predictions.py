#!/usr/bin/env python3
"""
Visualize model predictions vs actual gaze coordinates for a single CSV file.
Shows the actual gaze path and predicted next-point coordinates on a 2D plot.

Example usage:
python visualisations/visualize_predictions.py \
    data/processed/cog-load-mini/s_008.csv --model \
    checkpoints/best200.pt \
    --num_points 100 
"""
import argparse
import os
from random import random, randint
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gnext.data import EyePathDataset
from gnext.model import NextPointGNN
from baseline_models import MLPBaseline, CNNBaseline
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GINConv, TransformerConv

def load_graph_from_csv(csv_path, lookback=1):
    """
    Load a single CSV file and convert it to a graph.
    
    Args:
        csv_path: Path to CSV file
        lookback: Number of previous time steps to connect
        
    Returns:
        torch_geometric.data.Data: Graph object
    """
    temp_dir = os.path.dirname(csv_path)
    temp_name = os.path.basename(csv_path)
    
    # Create temporary dataset with just this file
    dataset = EyePathDataset(temp_dir, lookback=lookback, file_list=[temp_name])
    
    # Find the graph corresponding to our CSV
    graph = None
    for g in dataset.graphs:
        if g.seq_name == temp_name:
            graph = g
            break
    
    if graph is None:
        raise ValueError(f"Could not find {temp_name} in dataset")
    
    return graph

def load_model(model_path):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint
        
    Returns:
        tuple: (model, device, model_name, lookback, is_gnn)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint.get('model_name', 'NextPointGNN')
    lookback = checkpoint.get('lookback', 1)
    is_gnn = model_name == 'NextPointGNN'
    
    if is_gnn:
        # Map layer names to classes
        layer_map = {
            'SAGEConv': SAGEConv,
            'GCNConv': GCNConv,
            'GATConv': GATConv,
            'GINConv': GINConv,
            'TransformerConv': TransformerConv
        }
        layer_name = checkpoint.get('layer_name', 'SAGEConv')
        layer_class = layer_map.get(layer_name, SAGEConv)
        
        model = NextPointGNN(
            in_channels=checkpoint['in_channels'], 
            hidden_dim=checkpoint['hidden'], 
            num_layers=checkpoint['layers'],
            layer=layer_class
        ).to(device)
    elif model_name == 'MLPBaseline':
        model = MLPBaseline(
            input_dim=checkpoint['in_channels'],
            output_dim=2,
            hidden_dims=[checkpoint['hidden']] * checkpoint['layers']
        ).to(device)
    elif model_name == 'CNNBaseline':
        model = CNNBaseline(
            input_channels=checkpoint['in_channels'],
            output_dim=2,
            hidden_dims=[checkpoint['hidden']] * checkpoint['layers']
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, device, model_name, lookback, is_gnn

def get_predictions(model, graph, device, is_gnn=True, num_points=100):
    """
    Get model predictions for a graph.
    
    Args:
        model: Trained model (GNN or baseline)
        graph: Graph data object
        device: torch device
        is_gnn: Whether model is GNN-based
        num_points: Number of points to use from sequence
        
    Returns:
        tuple: (actual_coords, pred_coords, targets, mask) as numpy arrays
    """
    # Move graph to device and subsample if needed
    g = graph.clone()
    g = g.to(device)

    if g.x.size(0) > num_points:
        start_i = randint(0, max(0, g.x.size(0) - num_points))
        end_i = start_i + num_points
        print(f"Subsampling from {start_i} to {end_i} (total nodes: {g.x.size(0)})")
        
        # Subsample nodes on the same device
        g.x = g.x[start_i:end_i]
        g.y = g.y[start_i:end_i]
        g.mask = g.mask[start_i:end_i]

        # Adjust edges to new indexing, keeping device consistent
        edge_index = g.edge_index
        edge_mask = (edge_index[0] >= start_i) & (edge_index[0] < end_i) & \
                    (edge_index[1] >= start_i) & (edge_index[1] < end_i)
        g.edge_index = (edge_index[:, edge_mask] - start_i)

    with torch.no_grad():
        predictions = model(g.x, g.edge_index) if is_gnn else model(g.x)
    
    # Convert to numpy for plotting
    actual_coords = g.x.cpu().numpy()[:, :2]  # Only use x, y coordinates
    pred_coords = predictions.cpu().numpy()[:, :2]  # Only use x, y coordinates
    targets = g.y.cpu().numpy()[:, :2]  # Only use x, y coordinates
    mask = g.mask.cpu().numpy()
    
    return actual_coords, pred_coords, targets, mask

def plot_predictions(actual_coords, pred_coords, targets, mask, csv_path):
    """
    Create the visualization plot.
    
    Args:
        actual_coords: Actual coordinates [n_nodes, 2]
        pred_coords: Predicted coordinates [n_nodes, 2]  
        targets: Target coordinates [n_nodes, 2]
        mask: Valid prediction mask [n_nodes]
        csv_path: Path to original CSV for title
        
    Returns:
        matplotlib figure and axes
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    num_nodes = len(actual_coords)
    # Compute alpha values that fade in from 0.1 to 1.0
    alphas = np.linspace(0.1, 1.0, num_nodes)
    
    # Plot actual gaze path with fading line segments
    for i in range(num_nodes - 1):
        ax.plot(actual_coords[i:i+2, 0], actual_coords[i:i+2, 1], 
                'b-', alpha=alphas[i], linewidth=2)
    
    # Mark start and end of gaze path
    ax.scatter(actual_coords[0, 0], actual_coords[0, 1], 
               c='lightgreen', s=150, marker='o', edgecolors='green', linewidths=2, 
               label='Start', zorder=5)
    ax.scatter(actual_coords[-1, 0], actual_coords[-1, 1], 
               c='lightcoral', s=150, marker='o', edgecolors='red', linewidths=2, 
               label='End', zorder=5)
    
    # Plot target and prediction coordinates with fading
    valid_targets = targets[mask]
    valid_predictions = pred_coords[mask]
    valid_alphas = alphas[mask]
    
    ax.scatter(valid_targets[:, 0], valid_targets[:, 1], 
               c='green', s=40, alpha=valid_alphas, marker='s', label='Target next points')
    ax.scatter(valid_predictions[:, 0], valid_predictions[:, 1], 
               c='red', s=40, alpha=valid_alphas, marker='^', label='Predicted next points')
    
    # Draw arrows from current point to prediction and targets
    current_points = actual_coords[mask]
    for i in range(len(current_points)):
        ax.annotate('', xy=valid_predictions[i], xytext=current_points[i],
                   arrowprops=dict(arrowstyle='->', color='red', alpha=valid_alphas[i], lw=1))
        ax.annotate('', xy=valid_targets[i], xytext=current_points[i],
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=valid_alphas[i], lw=1))

    # Calculate metrics
    errors = np.linalg.norm(valid_predictions - valid_targets, axis=1)
    mae = np.round(np.mean(np.abs(valid_predictions - valid_targets)), 2)
    mean_euclidean = np.round(np.mean(errors), 2)
    pearson_x = np.round(100*np.corrcoef(valid_targets[:, 0], valid_predictions[:, 0])[0, 1], 1)
    pearson_y = np.round(100*np.corrcoef(valid_targets[:, 1], valid_predictions[:, 1])[0, 1], 1)
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate') 
    ax.set_title(f'Gaze Path Prediction: {os.path.basename(csv_path)}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add metrics text box outside plot area
    metrics_text = f'MAE: {mae:.2f}\nMean Euclidean: {mean_euclidean:.2f}\nPearson r (x): {pearson_x:.1f}\nPearson r (y): {pearson_y:.1f}'
    ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, 
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')
    
    return fig, ax

def print_statistics(pred_coords, targets, mask):
    """
    Print prediction error statistics.
    
    Args:
        pred_coords: Predicted coordinates [n_nodes, 2]
        targets: Target coordinates [n_nodes, 2]
        mask: Valid prediction mask [n_nodes]
    """
    valid_targets = targets[mask]
    valid_predictions = pred_coords[mask]
    
    if len(valid_targets) > 0:
        errors = np.linalg.norm(valid_predictions - valid_targets, axis=1)
        print(f"\nPrediction Statistics:")
        print(f"Number of predictions: {len(valid_predictions)}")
        print(f"Mean error (Euclidean distance): {np.mean(errors):.4f}")
        print(f"Std error: {np.std(errors):.4f}")
        print(f"Max error: {np.max(errors):.4f}")
        print(f"Min error: {np.min(errors):.4f}")

def visualize_predictions(csv_path, model_path, num_points=100, save_path=None):
    """
    Visualize model predictions vs actual coordinates for a single CSV file.
    
    Args:
        csv_path: Path to CSV file to visualize
        model_path: Path to saved model checkpoint
        num_points: Number of points from sequence to visualize
        save_path: Optional path to save the plot
    """
    # Load components
    model, device, model_name, lookback, is_gnn = load_model(model_path)
    graph = load_graph_from_csv(csv_path, lookback)
    
    # Get predictions
    actual_coords, pred_coords, targets, mask = get_predictions(
        model, graph, device, is_gnn, num_points)
    
    # Create plot
    fig, ax = plot_predictions(actual_coords, pred_coords, targets, mask, csv_path)

    # Add model name and lookback to title
    title = ax.get_title()
    ax.set_title(f"{title} | Model: {model_name} | Lookback: {lookback}")
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Print statistics
    print_statistics(pred_coords, targets, mask)

def main():
    parser = argparse.ArgumentParser(description="Visualize gaze path predictions")
    parser.add_argument("csv_path", type=str, help="Path to CSV file to visualize")
    parser.add_argument("--model", type=str, default="checkpoints/best.pt", 
                       help="Path to saved model checkpoint")
    parser.add_argument("--save", type=str, help="Path to save the plot (optional)")
    parser.add_argument("--num_points", type=int, default=100, 
                        help="Number of points from the sequence to visualize")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        return

    visualize_predictions(args.csv_path, args.model, args.num_points, args.save)

if __name__ == "__main__":
    # run until manually stopped (plots one plot and waits for it to be closed)
    while True:
        main()
