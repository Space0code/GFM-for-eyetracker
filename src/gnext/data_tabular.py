# data_tabular.py
"""
Tabular dataset for baseline models - no graph structure.
"""
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset

class TabularEyePathDataset(Dataset):
    """
    Loads CSV files as sliding windows for next-point prediction.
    Each sample: [lookback timesteps of (x,y,pupil_l,pupil_r)] -> (x_next, y_next)
    """
    def __init__(self, root_dir: str, recursive: bool = False, lookback: int = 1, ignore_dirs: list = [], file_list: list = None):
        if file_list is not None:
            files = [os.path.join(root_dir, f) for f in file_list]
        elif recursive:
            files = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))
        else:
            files = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
        
        if ignore_dirs:
            files = [f for f in files if not any(ig == os.path.basename(os.path.dirname(f)) for ig in ignore_dirs)]
        
        if not files:
            raise FileNotFoundError(f"No CSVs found in {root_dir}")
        
        # Load all data as windows
        self.samples = []
        self.seq_ids = []  # Track which sequence each sample belongs to
        
        for seq_id, path in enumerate(files):
            df = pd.read_csv(path)
            df.dropna(inplace=True)
            coords = df[['x-avg', 'y-avg']].values
            pupil_left = df['pupil-size-left-avg'].values
            pupil_right = df['pupil-size-right-avg'].values
            features = torch.tensor(
                [[coords[i,0], coords[i,1], pupil_left[i], pupil_right[i]] for i in range(len(df))],
                dtype=torch.float32
            )
            
            # Create sliding windows
            for i in range(len(features) - lookback):
                window = features[i:i+lookback].flatten()  # [lookback*4]
                target = features[i+lookback, :2]  # [2]
                self.samples.append((window, target))
                self.seq_ids.append(seq_id)
        
        print(f"Loaded {len(self.samples)} samples from {len(files)} sequences")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_seq_id(self, idx):
        """Return sequence ID for splitting by sequence."""
        return self.seq_ids[idx]
