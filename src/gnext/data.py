# data.py
"""
Dataset class for loading eye tracking sequences as temporal graphs.
Each CSV becomes a directed path graph for next-point prediction.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def load_csv_files(self, root_dir, recursive, ignore_dirs, file_list):
    if file_list is not None:
        self.files = [os.path.join(root_dir, f) for f in file_list]
    elif recursive:
        # search recursively in subdirectories
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))
    else:
        # search only in the root directory
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
    
    # filter out files in ignored directories
    if ignore_dirs:
        self.files = [f for f in self.files if not any(
            ig == os.path.basename(os.path.dirname(f)) for ig in ignore_dirs
        )]
    
    if not self.files:
        search_type = "recursively" if recursive else "in root directory"
        raise FileNotFoundError(f"No CSVs found {search_type} in {root_dir}")

def clean_dataset(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Clean the dataset by removing rows with NaN values in 'x-avg' or 'y-avg' columns.
    """

    if not {"time-rel-seconds", "x-avg", "y-avg"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: time-rel-seconds,x-avg,y-avg")

    df = df.sort_values("time-rel-seconds").reset_index(drop=True)

    df_cleaned = df.dropna().reset_index(drop=True)
    return df_cleaned

class EyePathDataset(Dataset):
    """
    Loads each CSV (time-rel-seconds,x-avg,y-avg) as a directed path graph.
    Nodes store only (x, y).
    Edge rule: i -> i+1 (previous datapoint to current).
    Targets: for node i, predict coords of node i+1 (last node masked).
    """
    def __init__(self, root_dir: str, recursive: bool = False, lookback: int = 1, ignore_dirs: list = [], file_list: list = None):
        """
        Load all CSV files from directory and convert to graphs.
        If file_list is provided, search for the files in the list in root_dir.
        """
        load_csv_files(self, root_dir, recursive, ignore_dirs, file_list)
        
        # pre-load all graphs into memory for simplicity
        self.graphs = [self._load_one(p, lookback) for p in self.files]
        print(f"Loaded {len(self.graphs)} graphs from {root_dir}")


    def _load_one(self, path: str, lookback: int = 1) -> Data:
        """Convert a single CSV file to a PyTorch Geometric Data object."""
        df = pd.read_csv(path)
        df = clean_dataset(self, df, path)        

        # node features: [x, y] coordinates
        x = torch.tensor(df[["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]].values, dtype=torch.float32)

        # temporal edges: connect each node to previous 1-10 steps
        src_list, dst_list = [], []
        n = len(df)
        for i in range(n):
            # Look back up to 10 steps (or fewer if near start)
            lb = min(lookback, i)
            for j in range(1, lb + 1):
                src_list.append(i - j)  # previous node
                dst_list.append(i)      # current node
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # targets: predict next coordinates (last node has no target)
        y = torch.full((n, 2), float("nan"), dtype=torch.float32)
        y[:-1] = x[1:, :2]  # target for node i is next node's (x, y) coordinates
        mask = torch.zeros(n, dtype=torch.bool)
        mask[:-1] = True  # mask out last node from loss calculation

        data = Data(x=x, edge_index=edge_index, y=y, mask=mask)
        data.seq_name = os.path.basename(path)
        data.dataset_name = os.path.basename(os.path.dirname(path))

        # print("Processed file:", path)
        # print(f"Loaded {data.seq_name}: {n} nodes, {edge_index.size(1)} edges, lookback={lookback}")

        return data



    def __len__(self):
        """Return number of sequences (graphs) in dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get a specific graph by index."""
        return self.graphs[idx]
    
class SpacioTemporalDataset(Dataset):
    """
    Dataset for spatio-temporal graphs combining spatial and temporal edges.
    Each CSV becomes a graph with both spatial and temporal connections.
    """
    def __init__(self, root_dir: str, recursive: bool = False, ignore_dirs: list = [], file_list: list = None, lookback: int = 1, k: int = 10):
        """
        Load all CSV files from directory and convert to graphs.
        If file_list is provided, search for the files in the list in root_dir.
        """
        self.k = k  # number of spatial neighbors
        self.lookback = lookback

        load_csv_files(self, root_dir, recursive, ignore_dirs, file_list)
        
        # pre-load all graphs into memory for simplicity
        self.graphs = [self._load_one(p, lookback) for p in self.files]
        print(f"Loaded {len(self.graphs)} graphs from {root_dir}")

    def _load_one(self, path: str, lookback: int = 1) -> Data:
        """Convert a single CSV file to a PyTorch Geometric Data object with spatio-temporal edges."""
        df = pd.read_csv(path)

        df = clean_dataset(self, df, path)
        df = df.loc[:, ["time-rel-seconds", "x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]]
        X = torch.tensor(df[["time-rel-seconds", "x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]].values, dtype=torch.float32)

        def _delta(feature: str):
            return df.loc[1:, feature] - df.loc[:-1, feature]
        def _l2_norm(a, b):
            return np.sqrt(np.square(a) + np.square(b))

        dt = _delta("time-rel-seconds")
        ddt = _delta(dt)

        dx = _delta("x-avg")
        dy = _delta("y-avg")
        d_gaze = np.sqrt(np.square(dx) + np.square(dy))
        v_gaze = d_gaze / dt
        d_v_gaze = _delta(v_gaze)
        a_gaze = d_v_gaze / ddt
        
        d_pupil_left = _delta("pupil-size-left-avg")
        d_pupil_right = _delta("pupil-size-right-avg")
        d_pupil = _l2_norm(d_pupil_left, d_pupil_right)
        v_pupil = d_pupil / dt
        d_v_pupil = _delta(v_pupil)
        a_pupil = d_v_pupil / ddt

        edges_temporal_weights = torch.tensor([dt, v_gaze, v_pupil, a_gaze, a_pupil], dtype=torch.float32).t()  # [num_edges, 5]
        edges_temporal = []    # connect each node to previous lookback steps


        n = len(df)
        k = getattr(self, "k", 10)
        # reserve integer index space [n, k], use -1 as sentinel for "no neighbor"
        edges_spatial = torch.full((n, k), -1, dtype=torch.long)
        for i in range(n):
            # Look back up to lookback steps (or fewer if near start)
            lb = min(lookback, i)
            for j in range(1, lb + 1):
                edges_temporal.append((i - j, i))  # previous node -> current node
            
            # spatial neighbors based on KDTree
            tree = KDTree(X[max(0, i):i+1, 1:3].numpy()) 
            dist, ind = tree.query(X[i, 1:3].numpy().reshape(1, -1), k=k+1)  # +1 to exclude self
            neighbors = ind[0][1:] + max(0, i)  # adjust indices
            edges_spatial[i] = torch.tensor(neighbors, dtype=torch.long)

            # d_gaze = 
        
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass