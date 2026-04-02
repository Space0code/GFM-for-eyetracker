from collections import Counter
import sys
from torch_geometric.data import DataLoader
import os
import yaml
from pathlib import Path
# Adjust imports based on your dataset class
project_root = Path(".").resolve()
print(f"Project root: {project_root}")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.data.data import SpacioTemporalDataset

# Load your dataset
dataset_cfg = yaml.safe_load(open(os.path.join("src/emotions/binary/configs/train_binary.yaml"), 'r'))['dataset']
dataset = SpacioTemporalDataset(
            root_dir=dataset_cfg.get('data_dir'),
            data_filepath=dataset_cfg.get('data_filepath'),
            filter_subjects=dataset_cfg.get('filter_subjects'),
            filter_recordings=dataset_cfg.get('filter_recordings'),
            file_list=dataset_cfg.get('file_list'),
            recursive=dataset_cfg['recursive'],
            ignore_dirs=dataset_cfg.get('ignore_dirs', []),
            window_length=dataset_cfg['window_length'],
            window_overlap=dataset_cfg['window_overlap'],
            kt=dataset_cfg['kt'],
            ks=dataset_cfg['ks'],
            use_edge_weights=dataset_cfg['use_edge_weights'],
            tau=dataset_cfg['tau'],
            cache_dir=dataset_cfg.get('cache_dir'),
            use_cache=dataset_cfg.get('use_cache', True),
            dropping_emotion_threshold=dataset_cfg.get('dropping_emotion_threshold', -1),
        )
loader = DataLoader(dataset, batch_size=1, shuffle=False)

temporal_counts = []
spatial_counts = []
node_counts = []

for batch in loader:
    edge_dict = batch.edge_index_dict
    node_counts.append(batch['node'].num_nodes)

    # Get edge counts by type
    if ("node", "temporal", "node") in edge_dict:
        temporal_counts.append(edge_dict[("node", "temporal", "node")].size(1))
    if ("node", "spatial", "node") in edge_dict:
        spatial_counts.append(edge_dict[("node", "spatial", "node")].size(1))

print(f"Temporal edges: min={min(temporal_counts)}, max={max(temporal_counts)}, mean={sum(temporal_counts)/len(temporal_counts):.1f}")
print(f"Spatial edges:  min={min(spatial_counts)}, max={max(spatial_counts)}, mean={sum(spatial_counts)/len(spatial_counts):.1f}")
print(f"Total graphs: {len(temporal_counts)}")
print(f"Node counts: min={min(node_counts)}, max={max(node_counts)}, mean={sum(node_counts)/len(node_counts):.1f}")

# Ratio
avg_ratio = (sum(temporal_counts) / sum(spatial_counts)) if sum(spatial_counts) > 0 else 0
print(f"Temporal/Spatial ratio: {avg_ratio:.2f}")


import matplotlib.pyplot as plt

# temporal_thresh = 9900
# spatial_thresh = 6200
# # temporal_counts = [max(count, temporal_thresh) for count in temporal_counts]
# # spatial_counts = [max(count, spatial_thresh) for count in spatial_counts]

# temporal_counts = [count if count > temporal_thresh else float("nan") for count in temporal_counts]
# spatial_counts = [count if count > spatial_thresh else float("nan") for count in spatial_counts]

# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].hist(temporal_counts, bins=20, alpha=0.7, label='Temporal')
# axes[1].hist(spatial_counts, bins=20, alpha=0.7, label='Spatial', color='orange')
# for ax in axes:
#     ax.set_xlabel('Number of Edges')
#     ax.set_ylabel('Frequency')
# plt.tight_layout()
# plt.show()

# draw a random graph
import random
import networkx as nx
import torch_geometric.utils as pyg_utils
import torch
while True:
    random_idx = random.randint(0, len(dataset)-1)
    data = dataset[random_idx]
    print(f"Random graph index: {random_idx}")
    print(f"Node features shape: {data['node'].x.shape}")
    print(f"Temporal edges shape: {data['node', 'temporal', 'node'].edge_index.shape if ('node', 'temporal', 'node') in data.edge_index_dict else 'None'}")
    print(f"Spatial edges shape: {data['node', 'spatial', 'node'].edge_index.shape if ('node', 'spatial', 'node') in data.edge_index_dict else 'None'}")

    draw_temporal = True
    draw_spatial = True


    # Create subgraph for nodes 100-300
    node_subset = list(range(100, 160))
    node_subset_set = set(node_subset)

    # Filter edges to include only those between nodes in the subset
    temporal_edge_index = data['node', 'temporal', 'node'].edge_index
    spatial_edge_index = data['node', 'spatial', 'node'].edge_index if ('node', 'spatial', 'node') in data.edge_index_dict else None

    node_subset_tensor = torch.tensor(list(node_subset_set))
    temporal_mask = torch.isin(temporal_edge_index[0], node_subset_tensor) & torch.isin(temporal_edge_index[1], node_subset_tensor)
    temporal_edges_filtered = temporal_edge_index[:, temporal_mask]

    spatial_mask = torch.isin(spatial_edge_index[0], node_subset_tensor) & torch.isin(spatial_edge_index[1], node_subset_tensor)
    spatial_edges_filtered = spatial_edge_index[:, spatial_mask]

    # Remap node indices to 0-200 for clearer visualization
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_subset)}
    temporal_edges_remapped = torch.stack([
        torch.tensor([node_mapping[int(u.item())] for u in temporal_edges_filtered[0]]),
        torch.tensor([node_mapping[int(v.item())] for v in temporal_edges_filtered[1]])
    ])
    spatial_edges_remapped = torch.stack([
        torch.tensor([node_mapping[int(u.item())] for u in spatial_edges_filtered[0]]),
        torch.tensor([node_mapping[int(v.item())] for v in spatial_edges_filtered[1]])
    ])

    # Debug: Print temporal edge structure
    print(f"\nTemporal edges (before filtering): {temporal_edge_index.shape}")
    print(f"Temporal edges (after filtering): {temporal_edges_filtered.shape}")
    print(f"First 10 temporal edges (remapped):")
    for i in range(min(10, temporal_edges_remapped.shape[1])):
        u, v = int(temporal_edges_remapped[0, i].item()), int(temporal_edges_remapped[1, i].item())
        print(f"  {u} -> {v}")

    print(f"\nSpatial edges (before filtering): {spatial_edge_index.shape}")
    print(f"Spatial edges (after filtering): {spatial_edges_filtered.shape}")
    print(f"First 10 spatial edges (remapped):")
    for i in range(min(10, spatial_edges_remapped.shape[1])):
        u, v = int(spatial_edges_remapped[0, i].item()), int(spatial_edges_remapped[1, i].item())
        print(f"  {u} -> {v}")

    # Create networkx graph (use MultiDiGraph to support multiple edge types)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(node_subset)))

    # Add edges with type information
    temporal_edges_list = [(int(u), int(v), {'edge_type': 'temporal'}) 
                        for u, v in zip(temporal_edges_remapped[0], temporal_edges_remapped[1])]
    spatial_edges_list = [(int(u), int(v), {'edge_type': 'spatial'}) 
                        for u, v in zip(spatial_edges_remapped[0], spatial_edges_remapped[1])]
    if draw_temporal:
        G.add_edges_from(temporal_edges_list)
    if draw_spatial:
        G.add_edges_from(spatial_edges_list)

    # Compute alpha values based on node indices: linear interpolation from 0.3 to 0.8
    num_nodes = len(node_subset)
    alpha_min, alpha_max = 0.3, 0.8
    node_alphas = {i: alpha_min + (i / max(num_nodes - 1, 1)) * (alpha_max - alpha_min) for i in range(num_nodes)}

    # Extract x, y coordinates from node features for the subset
    node_features = data['node'].x
    pos = {}
    for remapped_idx, original_idx in enumerate(node_subset):
        # Feature order: time, x, y, pupil_1, pupil_2
        x = float(node_features[original_idx, 1].item())
        y = float(node_features[original_idx, 2].item())
        pos[remapped_idx] = (x, y)

    # Create two separate plots: one for temporal, one for spatial
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot 1: Temporal edges only
    ax = axes[0]
    plt.sca(ax)

    # Draw temporal edges in blue
    temporal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'temporal']
    for u, v in temporal_edges:
        alpha = (node_alphas[u] + node_alphas[v]) / 2
        nx.draw_networkx_edges(G, pos, [(u, v)], edge_color='blue', alpha=alpha, width=1.5, arrowsize=10)

    # Draw nodes
    node_colors = [node_alphas[i] for i in range(num_nodes)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap='Blues', vmin=alpha_min, vmax=alpha_max)
    nx.draw_networkx_labels(G, pos, font_size=6)

    ax.set_title(f"Temporal Edges Only ({len(temporal_edges)} edges)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Feature Y")

    # Plot 2: Spatial edges only
    ax = axes[1]
    plt.sca(ax)

    # Draw spatial edges in red
    spatial_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'spatial']
    for u, v in spatial_edges:
        alpha = (node_alphas[u] + node_alphas[v]) / 2
        nx.draw_networkx_edges(G, pos, [(u, v)], edge_color='red', alpha=alpha, width=1.5, arrowsize=10)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap='Blues', vmin=alpha_min, vmax=alpha_max)
    nx.draw_networkx_labels(G, pos, font_size=6)

    ax.set_title(f"Spatial Edges Only ({len(spatial_edges)} edges)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Feature Y")

    fig.suptitle(f"Spatio-Temporal Graph: Nodes 100-160 | Dataset Index {random_idx}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()