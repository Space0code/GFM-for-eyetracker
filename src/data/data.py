# data.py
import os
import glob
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData


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

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing rows with NaN values in "time-rel-seconds", 'x-avg' or 'y-avg' columns.
    """

    if not {"time-rel-seconds", "x-avg", "y-avg"}.issubset(df.columns):
        raise ValueError(f"The DataFrame must have columns: time-rel-seconds,x-avg,y-avg")

    df = df.dropna(subset=["time-rel-seconds", "x-avg", "y-avg"])
    df = df.sort_values("time-rel-seconds").reset_index(drop=True)

    return df

  
class SpacioTemporalDataset(Dataset):
    """
    Dataset for spatio-temporal graphs combining spatial and temporal edges.
    Each CSV becomes a graph (or multiple graphs) with both spatial and temporal connections.
    """
    def __init__(self, root_dir: str, recursive: bool = False, ignore_dirs: list = None, file_list: list = None, kt: int = 5, ks: int = 10, window_length: int = 60, window_overlap: float = 0):
        """
        Load all CSV files from directory and convert to graphs.
        If file_list is provided, search for the files in the list in root_dir.

        Parameters:
        - root_dir: root dir of data files
        - recursive: search recursively (bool)
        - ignore_dirs: list of directories with data to ignore
        - file_list: list of csv files relative root_dir to be loaded (exclusively these)
        - kt: k temporal neigbors
        - ks: k spatial neigbors
        - window_length: in seconds
        - window_overlap: fraction in range [0, 1)
        """
        self.kt = kt  # temporal horizon
        self.ks = ks  # k for spatial kNN
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.files = []
        self.graphs = []

        load_csv_files(self, root_dir, recursive, ignore_dirs, file_list)
        
        # pre-load all graphs into memory for simplicity
        for path in self.files:
            df = self._load_df(path)
            # generate window slices based on time
            for window_slice in self._generate_window_slices(df):
                if (window_slice.stop - window_slice.start) < max(self.kt, self.ks):
                    print(f"Window {window_slice} too small for kt={kt} and ks={ks}. Skipping... ")
                    continue
                graph = self._load_one(df, window_slice)
                self.graphs.append(graph)

        print(f"Loaded {len(self.graphs)} graphs from {root_dir}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def _load_df(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = clean_dataset(df)
        return df

    def _generate_window_slices(self, df: pd.DataFrame):
        """
        Generate window slices based on time with specified overlap.
        
        Yields:
            slice objects for each window based on time-rel-seconds
        """
        times = df["time-rel-seconds"].values
        if len(times) == 0:
            return
        
        start_time = times[0]
        end_time = times[-1]
        
        # calculate step size based on overlap
        assert 0 <= self.window_overlap < 1
        assert self.window_length > 0
        step_size = self.window_length * (1 - self.window_overlap)
        
        # generate windows
        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + self.window_length, end_time)
            
            # find indices for this time window
            start_idx = np.searchsorted(times, current_start, side='left')
            end_idx = np.searchsorted(times, current_end, side='right')
            
            # only yield if window has at least some data points
            if end_idx > start_idx:
                yield slice(start_idx, end_idx)
            
            # move to next window
            current_start += step_size
            
    def _load_one(self, df: pd.DataFrame, window_slice: slice) -> HeteroData:
        """
        Basic graph.
        V: time, x, y, pupil-left, pupil-right
        E-temporal: v, u not more than kt time-steps distant
        E-spatial: kNN (k:=ks)
        No edge weights yet.
        """
        ks = self.ks
        kt = self.kt

        df = df.loc[window_slice, ["time-rel-seconds", "x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]]
        n = len(df)

        #### node features matrix X
        X = torch.tensor(df[["time-rel-seconds", "x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]].values, dtype=torch.float32)
        
        #### creating TEMPORAL edge_index matrix
        idx = torch.arange(n)          # [0, 1, ..., n-1]

        # relative offsets: -k..-1 and 1..k  (both directions)
        rel = torch.arange(-kt, kt + 1)
        rel = rel[rel != 0]            # drop 0

        # all candidate (src, dst) pairs before boundary check
        src = idx.repeat_interleave(len(rel))         # shape [n * (2k)]
        dst = (idx.view(-1, 1) + rel.view(1, -1)).reshape(-1)

        # mask out invalid indices (outside [0, n-1])
        valid = (dst >= 0) & (dst < n)
        src = src[valid]
        dst = dst[valid]

        edge_index_temporal = torch.stack([src, dst], dim=0)  # [2, E]

        #### creating SPATIAL edge_index matrix
        tree = KDTree(X[:, 1:3].numpy())  # use (x, y) for spatial neighbors
        dist, idx = tree.query(X[:, 1:3].numpy(), k=ks + 1)  # +1 to exclude self, idx shape: [N, k+1], idx[i, 0] == i (self)
        neighbors = idx[:, 1:].reshape(-1)          # drop self, flatten
        src = np.repeat(np.arange(n), ks)            # each node repeated k times

        edge_index_spatial = torch.tensor(
            np.vstack([np.concatenate([src, neighbors]),
                    np.concatenate([neighbors, src])]),
            dtype=torch.long,
        )

        data = HeteroData()
        data["node"].x = X
        data["node"].num_nodes = n
        data["node", "temporal", "node"].edge_index = edge_index_temporal
        data["node", "spatial", "node"].edge_index = edge_index_spatial

        return data

# class EyePathDataset(Dataset):
#     """
#     Loads each CSV (time-rel-seconds,x-avg,y-avg) as a directed path graph.
#     Nodes store only (x, y).
#     Edge rule: i -> i+1 (previous datapoint to current).
#     Targets: for node i, predict coords of node i+1 (last node masked).
#     """
#     def __init__(self, root_dir: str, recursive: bool = False, lookback: int = 1, ignore_dirs: list = [], file_list: list = None):
#         """
#         Load all CSV files from directory and convert to graphs.
#         If file_list is provided, search for the files in the list in root_dir.
#         """
#         load_csv_files(self, root_dir, recursive, ignore_dirs, file_list)
        
#         # pre-load all graphs into memory for simplicity
#         self.graphs = [self._load_one(p, lookback) for p in self.files]
#         print(f"Loaded {len(self.graphs)} graphs from {root_dir}")


#     def _load_one(self, path: str, lookback: int = 1) -> Data:
#         """Convert a single CSV file to a PyTorch Geometric Data object."""
#         df = pd.read_csv(path)
#         df = clean_dataset(self, df, path)        

#         # node features: [x, y] coordinates
#         x = torch.tensor(df[["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]].values, dtype=torch.float32)

#         # temporal edges: connect each node to previous 1-10 steps
#         src_list, dst_list = [], []
#         n = len(df)
#         for i in range(n):
#             # Look back up to 10 steps (or fewer if near start)
#             lb = min(lookback, i)
#             for j in range(1, lb + 1):
#                 src_list.append(i - j)  # previous node
#                 dst_list.append(i)      # current node
        
#         edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

#         # targets: predict next coordinates (last node has no target)
#         y = torch.full((n, 2), float("nan"), dtype=torch.float32)
#         y[:-1] = x[1:, :2]  # target for node i is next node's (x, y) coordinates
#         mask = torch.zeros(n, dtype=torch.bool)
#         mask[:-1] = True  # mask out last node from loss calculation

#         data = Data(x=x, edge_index=edge_index, y=y, mask=mask)
#         data.seq_name = os.path.basename(path)
#         data.dataset_name = os.path.basename(os.path.dirname(path))

#         # print("Processed file:", path)
#         # print(f"Loaded {data.seq_name}: {n} nodes, {edge_index.size(1)} edges, lookback={lookback}")

#         return data



#     def __len__(self):
#         """Return number of sequences (graphs) in dataset."""
#         return len(self.graphs)

#     def __getitem__(self, idx):
#         """Get a specific graph by index."""
#         return self.graphs[idx]
  