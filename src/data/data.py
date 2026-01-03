# data.py
import os
import glob
import math
import hashlib
import pickle
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
    Clean the dataset by removing rows with NaN values in required feature columns.
    Drops rows with NaN in: time-rel-seconds, x-avg, y-avg, pupil-size-left-avg, pupil-size-right-avg
    """

    required_cols = ["time-rel-seconds", "x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"The DataFrame must have columns: {', '.join(required_cols)}")

    # Drop NaN in required columns
    cols_to_check = required_cols.copy()
    
    # Also check pupil columns if they exist
    if "pupil-size-left-avg" in df.columns:
        cols_to_check.append("pupil-size-left-avg")
    if "pupil-size-right-avg" in df.columns:
        cols_to_check.append("pupil-size-right-avg")
    
    df = df.dropna(subset=cols_to_check)
    df = df.sort_values("time-rel-seconds").reset_index(drop=True)

    return df

  
class SpacioTemporalDataset(Dataset):
    """
    Dataset for spatio-temporal graphs combining spatial and temporal edges.
    Each CSV becomes a graph (or multiple graphs) with both spatial and temporal connections.
    """
    def __init__(self, root_dir: str, recursive: bool = False, ignore_dirs: list = None, file_list: list = None, kt: int = 5, ks: int = 10, window_length: int = 60, window_overlap: float = 0, cache_dir: str = None, use_cache: bool = True):
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
        - cache_dir: directory to store cached processed graphs (default: root_dir/.cache)
        - use_cache: whether to use caching (default: True)
        """
        self.kt = kt  # temporal horizon
        self.ks = ks  # k for spatial kNN
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.files = []
        self.graphs = []
        self.emotion_names = []  # Store emotion column names
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.join(root_dir, ".cache")
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Try to load from cache first
        if use_cache:
            cache_path = self._get_cache_path(root_dir, recursive, ignore_dirs, file_list)
            if os.path.exists(cache_path):
                print(f"Loading dataset from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.graphs = cached_data['graphs']
                    self.files = cached_data['files']
                    self.emotion_names = cached_data.get('emotion_names', [])
                print(f"Loaded {len(self.graphs)} graphs from cache")
                return
        else:
            print("Caching disabled, processing dataset from scratch.")

        # Process dataset from scratch
        load_csv_files(self, root_dir, recursive, ignore_dirs, file_list)
        
        # pre-load all graphs into memory for simplicity
        for path in self.files:
            df = self._load_df(path)
            # generate window slices based on time
            for window_slice in self._generate_window_slices(df):
                if (window_slice.stop - window_slice.start) < max(self.kt, self.ks) + 1:
                    print(f"Window {window_slice} too small for kt={kt} and ks={ks}. [path={path}]. Skipping... ")
                    continue
                graph = self._load_one(df, window_slice)
                # Store source file information
                graph.source_file = os.path.basename(path)
                # Store emotion names from first graph
                if not self.emotion_names and hasattr(graph, 'emotion_names'):
                    self.emotion_names = graph.emotion_names
                self.graphs.append(graph)

        print(f"Loaded {len(self.graphs)} graphs from {root_dir}")
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_path)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def _get_cache_path(self, root_dir: str, recursive: bool, ignore_dirs: list, file_list: list) -> str:
        """
        Generate a unique cache filename based on dataset parameters.
        Uses hash of configuration to ensure different parameters create different caches.
        """
        # Create a unique identifier based on parameters
        config_str = f"kt={self.kt}_ks={self.ks}_wl={self.window_length}_wo={self.window_overlap}"
        config_str += f"_rec={recursive}_ignore={ignore_dirs}_files={file_list}"
        
        # Hash the configuration
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache filename includes parameters for readability
        cache_filename = f"dataset_kt{self.kt}_ks{self.ks}_wl{self.window_length}_wo{self.window_overlap}_{config_hash}.pkl"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _save_to_cache(self, cache_path: str):
        """Save processed graphs to cache file."""
        try:
            print(f"Saving dataset to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'graphs': self.graphs,
                    'files': self.files,
                    'emotion_names': self.emotion_names,
                    'kt': self.kt,
                    'ks': self.ks,
                    'window_length': self.window_length,
                    'window_overlap': self.window_overlap
                }, f)
            print(f"Successfully cached {len(self.graphs)} graphs")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            # Continue without caching - don't fail the entire dataset loading

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

        # Select feature columns
        feature_cols = ["time-rel-seconds", "x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]
        df_window = df.loc[window_slice, :]
        n = len(df_window)

        #### node features matrix X
        X = torch.tensor(df_window[feature_cols].values, dtype=torch.float32)
        
        #### Extract emotion labels (graph-level targets)
        emotion_cols = [col for col in df_window.columns if "emotion" in col.lower()]
        if emotion_cols:
            # Average emotion values across the window for graph-level prediction
            y = torch.tensor(df_window[emotion_cols].mean(axis=0).values, dtype=torch.float32)
        else:
            y = None
        
        #### Extract subject and recording if available
        subject = df_window["subject"].iloc[0] if "subject" in df_window.columns else None
        recording = df_window["recording"].iloc[0] if "recording" in df_window.columns else None
        
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

        # remove duplicate spatial edges
        edge_index_spatial = torch.unique(edge_index_spatial, dim=1)

        data = HeteroData()
        data["node"].x = X
        data["node"].num_nodes = n
        data["node", "temporal", "node"].edge_index = edge_index_temporal
        data["node", "spatial", "node"].edge_index = edge_index_spatial
        
        # Add emotion targets if available
        if y is not None:
            data.y = y
        
        # Add subject and recording if available
        if subject is not None:
            data.subject = subject
        if recording is not None:
            data.recording = recording
        
        # Store emotion column names in data object for later use
        if emotion_cols:
            data.emotion_names = emotion_cols

        return data
