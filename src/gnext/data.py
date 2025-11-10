# data.py
"""
Dataset class for loading eye tracking sequences as temporal graphs.
Each CSV becomes a directed path graph for next-point prediction.
"""
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class EyePathDataset(Dataset):
    """
    Loads each CSV (time-rel-seconds,x-avg,y-avg) as a directed path graph.
    Nodes store only (x, y).
    Edge rule: i -> i+1 (previous datapoint to current).
    Targets: for node i, predict coords of node i+1 (last node masked).
    """
    def __init__(self, root_dir: str, recursive: bool = False, lookback: int = 1):
        """Load all CSV files from directory and convert to graphs."""
        if recursive:
            # search recursively in subdirectories
            self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))
        else:
            # search only in the root directory
            self.files = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
        
        if not self.files:
            search_type = "recursively" if recursive else "in root directory"
            raise FileNotFoundError(f"No CSVs found {search_type} in {root_dir}")
        # pre-load all graphs into memory for simplicity
        self.graphs = [self._load_one(p, lookback) for p in self.files]
        print(f"Loaded {len(self.graphs)} graphs from {root_dir}")

    def _load_one(self, path: str, lookback: int = 1) -> Data:
        """Convert a single CSV file to a PyTorch Geometric Data object."""
        df = pd.read_csv(path)

        df = self._clean_dataset(df, path)
        
        n = len(df)
        if n < 2:
            raise ValueError(f"{path} must have >=2 rows")

        # node features: [x, y] coordinates
        x = torch.tensor(df[["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]].values, dtype=torch.float32)

        # temporal edges: connect each node to previous 1-10 steps
        src_list, dst_list = [], []
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
        return data

    def _clean_dataset(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        """
        Clean the dataset by removing rows with NaN values in 'x-avg' or 'y-avg' columns.
        """

        if not {"time-rel-seconds", "x-avg", "y-avg"}.issubset(df.columns):
            raise ValueError(f"{path} must have columns: time-rel-seconds,x-avg,y-avg")

        df = df.sort_values("time-rel-seconds").reset_index(drop=True)

        df_cleaned = df.dropna().reset_index(drop=True)
        return df_cleaned

    def __len__(self):
        """Return number of sequences (graphs) in dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get a specific graph by index."""
        return self.graphs[idx]