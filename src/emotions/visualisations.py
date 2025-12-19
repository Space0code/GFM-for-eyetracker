"""
Visualize graph predictions from saved model outputs.

Usage: python src/emotions/visualisations.py --run_dir results/SpatioTemporalHeteroGNN/eSEEd_v2/2025-12-19_XX-XX-XX
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.data import SpacioTemporalDataset


def load_saved_outputs(data_file):
    """Load model outputs and targets."""
    data = torch.load(data_file)
    return data['outputs'], data['targets']


def compute_edge_alpha(values, src, dst, max_diff):
    """Compute edge alpha based on distance (closer = higher alpha)."""
    diff = abs(values[src] - values[dst])
    alpha = 1.0 - (diff / max_diff) if max_diff > 0 else 0.5
    return np.clip(alpha, 0.1, 1.0)


def visualize_graph(graph, prediction, target, sample_info="Unknown", save_path=None):
    """Visualize graph structure with predictions."""
    # Extract features and edges
    x = graph['node'].x.numpy()
    temporal_edges = graph['node', 'temporal', 'node'].edge_index.numpy()
    spatial_edges = graph['node', 'spatial', 'node'].edge_index.numpy()
    
    pos_x, pos_y, times = x[:, 1], x[:, 2], x[:, 0]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw temporal edges (blue, alpha by time distance)
    max_time = times.max() - times.min() if len(times) > 1 else 1.0
    for i in range(temporal_edges.shape[1]):
        src, dst = temporal_edges[:, i]
        alpha = compute_edge_alpha(times, src, dst, max_time)
        ax.plot([pos_x[src], pos_x[dst]], [pos_y[src], pos_y[dst]], 
                'b-', alpha=alpha, linewidth=0.5, zorder=1)
    
    # Draw spatial edges (red, alpha by spatial distance)
    spatial_dists = [np.sqrt((pos_x[s] - pos_x[d])**2 + (pos_y[s] - pos_y[d])**2) 
                     for s, d in spatial_edges.T]
    max_dist = max(spatial_dists) if spatial_dists else 1.0
    
    for i, (src, dst) in enumerate(spatial_edges.T):
        alpha = 1.0 - (spatial_dists[i] / max_dist) if max_dist > 0 else 0.5
        alpha = np.clip(alpha, 0.1, 1.0)
        ax.plot([pos_x[src], pos_x[dst]], [pos_y[src], pos_y[dst]], 
                'r-', alpha=alpha, linewidth=0.5, zorder=2)
    
    # Draw nodes
    ax.scatter(pos_x[1:-1], pos_y[1:-1], c='green', s=30, alpha=0.6, zorder=3)
    ax.scatter(pos_x[0], pos_y[0], c='lightgreen', s=150, edgecolors='darkgreen', 
               linewidths=2, zorder=4, label='Start')
    ax.scatter(pos_x[-1], pos_y[-1], c='red', s=150, edgecolors='darkred', 
               linewidths=2, zorder=4, label='End')
    
    # Format title with predictions
    emotions = ['Anger', 'Tenderness', 'Sadness', 'Disgust']
    pred_str = ', '.join([f'{e}={p:.2f}' for e, p in zip(emotions, prediction)])
    tgt_str = ', '.join([f'{e}={t:.2f}' for e, t in zip(emotions, target)])
    
    ax.set_title(f'{sample_info}\nTarget: {tgt_str}\nPredicted: {pred_str}', fontsize=10, pad=20)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color='blue', linewidth=2, label='Temporal'),
        Line2D([0], [0], color='red', linewidth=2, label='Spatial'),
        ax.scatter([], [], c='lightgreen', s=100, label='Start'),
        ax.scatter([], [], c='red', s=100, label='End')
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize graph predictions')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory with saved data')
    parser.add_argument('--data_dir', type=str, default='./data/processed/eSEEd_v2/')
    parser.add_argument('--random_seed', type=int, default=None)
    args = parser.parse_args()
    
    if args.random_seed:
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    
    # Find saved epoch files
    data_dir = os.path.join(args.run_dir, 'data')
    epoch_files = sorted(Path(data_dir).glob('epoch_*.pt'))
    
    if not epoch_files:
        print(f"No data files found in {data_dir}")
        return
    
    # Use last epoch (highest epoch number)
    selected_file = epoch_files[-1]
    epoch_num = selected_file.stem.split('_')[1]
    print(f"Using epoch {epoch_num} (last available)")
    
    outputs, targets = load_saved_outputs(selected_file)
    sample_idx = random.randint(0, len(outputs) - 1)
    prediction = outputs[sample_idx].numpy()
    target = targets[sample_idx].numpy()
    
    print(f"Epoch: {epoch_num}, Sample: {sample_idx}/{len(outputs)}")
    print(f"Target:     {target}")
    print(f"Prediction: {prediction}")
    
    # Load dataset and get corresponding graph
    from sklearn.model_selection import train_test_split
    
    dataset = SpacioTemporalDataset(
        root_dir=args.data_dir,
        file_list=["sample_01_recording_01_merged.csv",
                   "sample_02_recording_01_merged.csv",
                   "sample_03_recording_01_merged.csv",
                   "sample_04_recording_01_merged.csv"],
        recursive=True,
        window_length=10,
        kt=2,
        ks=2,
    )
    
    _, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    graph = dataset[test_idx[sample_idx]]
    
    # Extract source file info
    source_file = getattr(graph, 'source_file', 'Unknown')
    sample_info = f"Epoch {epoch_num} | {source_file}"
    
    # Save visualization
    vis_dir = os.path.join(args.run_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f'epoch_{epoch_num}_sample_{sample_idx}.png')
    
    visualize_graph(graph, prediction, target, sample_info, save_path)


if __name__ == "__main__":
    main()
