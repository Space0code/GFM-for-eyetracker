# data.py
import os
import glob
import math
import hashlib
import pickle
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData

from data.hci_signals import (
    DISTANCE_AVG_COLUMN,
    FIXATION_INDEX_COLUMN,
    feature_interpolation_columns,
    prepare_hci_eye_tracking_signals,
    resolve_optional_hci_feature_columns,
)


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

def load_single_csv_file(
    self,
    data_filepath,
    filter_subjects=None,
    filter_recordings=None,
    exclude_subjects=None,
):
    """Load data from a single CSV file containing all subjects and recordings.
    
    Args:
        data_filepath: Path to the single CSV file
        filter_subjects: Optional list of subject IDs to include
        filter_recordings: Optional list of recording IDs to include
        exclude_subjects: Optional list of subject IDs to exclude
    """
    if not os.path.exists(data_filepath):
        raise FileNotFoundError(f"Data file not found: {data_filepath}")
    
    df = pd.read_csv(data_filepath)
    
    # Check required columns
    if 'subject' not in df.columns or 'recording' not in df.columns:
        raise ValueError(f"CSV must contain 'subject' and 'recording' columns")
    
    # Apply filters if provided
    if filter_subjects is not None:
        df = df[df['subject'].isin(filter_subjects)]
    if filter_recordings is not None:
        df = df[df['recording'].isin(filter_recordings)]
    if exclude_subjects is not None:
        df = df[~df['subject'].isin(exclude_subjects)]
    
    if len(df) == 0:
        raise ValueError("No data remaining after applying filters")
    
    # Group by (subject, recording) and create virtual file entries
    self.files = []
    self.dataframes = {}  # Store dataframes by virtual filename
    
    grouped = df.groupby(['subject', 'recording'])
    for (subject, recording), group_df in grouped:
        virtual_filename = f"subject_{subject}_recording_{recording}.csv"
        self.files.append(virtual_filename)
        self.dataframes[virtual_filename] = group_df.reset_index(drop=True)
    
    print(f"Loaded data from {data_filepath}: {len(self.files)} subject-recording pairs")

def infer_default_target_columns(df: pd.DataFrame) -> List[str]:
    """Infer numeric emotion target columns from a DataFrame."""
    targets: List[str] = []
    for col in df.columns:
        if not col.startswith("emotion-"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            targets.append(col)
    return sorted(targets)


def clean_dataset(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    interpolation_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Clean the dataset by removing rows with NaN values in required feature columns.
    Drops rows with NaN in: time-rel-seconds, x-avg, y-avg, pupil-size-left-avg, pupil-size-right-avg
    """

    if required_cols is None:
        required_cols = [
            "time-rel-seconds",
            "x-avg",
            "y-avg",
            "pupil-size-left-avg",
            "pupil-size-right-avg",
        ]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"The DataFrame must have columns: {', '.join(required_cols)}")

    df = df.sort_values("time-rel-seconds").reset_index(drop=True)
    df = interpolate_missing_data(df, interpolation_columns=interpolation_cols)
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df

def interpolate_missing_data(
    df: pd.DataFrame,
    window_size_ms: float = 100.0,
    interpolation_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Interpolate missing data in the DataFrame using a rolling window approach.
    Missing values in numeric columns are filled using linear interpolation within a time window.

    Parameters:
    - df: Input DataFrame with potential missing values. Must contain 'time-rel-seconds' column.
    - window_size_ms: Maximum time gap (in milliseconds) to interpolate across.

    Returns:
    - DataFrame with interpolated values.
    """
    assert 'time-rel-seconds' in df.columns, "DataFrame must contain 'time-rel-seconds' column."
    
    # Calculate average sampling interval in seconds.
    # Guard tiny/degenerate groups (for example one-row snapshots after filtering).
    time_diffs = pd.to_numeric(df["time-rel-seconds"], errors="coerce").diff().dropna()
    if time_diffs.empty:
        return df

    avg_sampling_interval_s = float(time_diffs.mean())
    if not np.isfinite(avg_sampling_interval_s) or avg_sampling_interval_s <= 0:
        return df
    window_size_s = window_size_ms / 1000.0
    
    # Convert time window to number of samples
    limit_samples = max(1, int(np.ceil(window_size_s / avg_sampling_interval_s)))
    
    # Interpolate only selected numeric columns (excluding time column)
    if interpolation_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "time-rel-seconds"]
    else:
        numeric_cols = [
            col
            for col in interpolation_columns
            if col in df.columns and col != "time-rel-seconds" and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='forward', limit=limit_samples)
    
    return df
  
class SpacioTemporalDataset(Dataset):
    """
    Dataset for spatio-temporal graphs combining spatial and temporal edges.
    Each CSV becomes a graph (or multiple graphs) with both spatial and temporal connections.
    """
    def __init__(
            self, root_dir: str = None, data_filepath: str = None, 
            filter_subjects: list = None, filter_recordings: list = None,
            exclude_subjects: list = None,
            recursive: bool = False, ignore_dirs: list = None, file_list: list = None, 
            kt: int = 5, ks: int = 10, use_edge_weights: bool = True, tau: float = 0.05,
            graph_version: str = "v2", edge_weight_mode: str = "learned_signed",
            window_length: int = 60, window_overlap: float = 0, 
            cache_dir: str = None, use_cache: bool = True, dropping_emotion_threshold: float = -1,
            min_samples_per_window: Optional[int] = None,
            feature_columns: Optional[List[str]] = None, target_columns: Optional[List[str]] = None,
            dropna_columns: Optional[List[str]] = None, experiment_type_column: str = "experiment-type",
            allowed_experiment_types: Optional[List[str]] = None, label_quality_column: Optional[str] = None,
            allowed_label_quality_values: Optional[List[str]] = None,
            target_aggregation: str = "mean",
            use_distance_avg: bool = True,
            use_fixation_duration: bool = True,
            use_delta_distance_edge_feature: bool = True,
            use_fixation_edges: bool = True):
        """
        Load CSV files and convert to graphs.
        
        Use either root_dir (old behavior) or data_filepath (new behavior).

        Parameters:
        - root_dir: root dir of data files (mutually exclusive with data_filepath)
        - data_filepath: path to single CSV file with all data (mutually exclusive with root_dir)
        - filter_subjects: list of subject IDs to include (only with data_filepath)
        - filter_recordings: list of recording IDs to include (only with data_filepath)
        - exclude_subjects: list of subject IDs to exclude
        - recursive: search recursively (bool)
        - ignore_dirs: list of directories with data to ignore
        - file_list: list of csv files relative root_dir to be loaded (exclusively these)
        - kt: k temporal neigbors
        - ks: k spatial neigbors
        - graph_version: "v1" creates temporal/spatial edges; "v2" splits temporal forward/backward edges
        - edge_weight_mode: "handcrafted" stores scalar exp(-dt/tau) weights; "learned_signed" stores relation features for v2
        - window_length: in seconds
        - window_overlap: fraction in range [0, 1)
        - cache_dir: directory to store cached processed graphs (default: root_dir/.cache for old, data/.cache for new)
        - use_cache: whether to use caching (default: True)
        - dropping_emotion_threshold: threshold to drop (subject, recording) pairs where all emotion values are <= this threshold
        """
        
        # Validate input: exactly one of root_dir or data_filepath must be provided
        if (root_dir is None) == (data_filepath is None):
            raise ValueError("Must provide exactly one of: root_dir or data_filepath")

        self.kt = kt  # temporal horizon
        self.ks = ks  # k for spatial kNN
        self.use_edge_weights = use_edge_weights
        self.tau = tau # edge weight time decay constant (in seconds)
        if graph_version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported graph_version='{graph_version}'. Choose 'v1' or 'v2'.")
        self.graph_version = graph_version
        if edge_weight_mode not in {"handcrafted", "learned_signed"}:
            raise ValueError(
                f"Unsupported edge_weight_mode='{edge_weight_mode}'. "
                "Choose 'handcrafted' or 'learned_signed'."
            )
        self.edge_weight_mode = edge_weight_mode
        self.window_length = window_length
        self.window_overlap = window_overlap
        resolved_min_samples = max(self.kt, self.ks) + 1 if min_samples_per_window is None else int(min_samples_per_window)
        if resolved_min_samples <= 0:
            raise ValueError("min_samples_per_window must be > 0.")
        self.min_samples_per_window = resolved_min_samples
        self.files = []
        self.graphs = []
        self.emotion_names = []  # Store emotion column names
        self.dropping_emotion_threshold = dropping_emotion_threshold
        self.use_distance_avg = bool(use_distance_avg)
        self.use_fixation_duration = bool(use_fixation_duration)
        self.use_delta_distance_edge_feature = bool(use_delta_distance_edge_feature)
        self.use_fixation_edges = bool(use_fixation_edges)
        self.feature_columns = resolve_optional_hci_feature_columns(
            feature_columns,
            use_distance_avg=self.use_distance_avg,
            use_fixation_duration=self.use_fixation_duration,
        )
        self.target_columns = target_columns
        self.target_aggregation = target_aggregation
        self.dropna_columns = dropna_columns
        self.experiment_type_column = experiment_type_column
        self.allowed_experiment_types = allowed_experiment_types
        self.label_quality_column = label_quality_column
        self.allowed_label_quality_values = allowed_label_quality_values
        self.dataframes = {}  # For single CSV mode
        self.use_single_file = data_filepath is not None
        self.filter_subjects = filter_subjects
        self.filter_recordings = filter_recordings
        self.exclude_subjects = exclude_subjects

        # Setup cache directory
        if cache_dir is None:
            if root_dir is not None:
                cache_dir = os.path.join(root_dir, ".cache")
            else:
                cache_dir = os.path.join(os.path.dirname(data_filepath), ".cache")
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Try to load from cache first
        if use_cache:
            cache_path = self._get_cache_path(
                root_dir, data_filepath, filter_subjects, filter_recordings,
                exclude_subjects, recursive, ignore_dirs, file_list
            )
            if os.path.exists(cache_path):
                print(f"Loading dataset from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.graphs = cached_data['graphs']
                    self.files = cached_data['files']
                    self.emotion_names = cached_data.get('emotion_names', [])
                    self.dataframes = cached_data.get('dataframes', {})
                print(f"Loaded {len(self.graphs)} graphs from cache")
                return
        else:
            print("Caching disabled, processing dataset from scratch.")

        # Process dataset from scratch
        if data_filepath is not None:
            load_single_csv_file(
                self,
                data_filepath,
                filter_subjects,
                filter_recordings,
                exclude_subjects,
            )
        else:
            load_csv_files(self, root_dir, recursive, ignore_dirs, file_list)
        
        # pre-load all graphs into memory for simplicity
        for path in self.files:
            df = self._load_df(path)

            if len(df) == 0:
                continue

            target_cols = self._resolve_target_columns(df)
            if target_cols and self.dropping_emotion_threshold > -np.inf:
                all_zero = (df[target_cols] <= self.dropping_emotion_threshold).all(axis=1)
                if all_zero.all():
                    print(
                        f"All target values below or equal to threshold "
                        f"{self.dropping_emotion_threshold} in file {path}. Skipping file."
                    )
                    continue

            # generate window slices based on time
            for window_slice in self._generate_window_slices(df):
                if (window_slice.stop - window_slice.start) < self.min_samples_per_window:
                    # remove spammy warning about small windows, just skip them silently
                    # print(
                    #     f"Window {window_slice} too small for min_samples_per_window="
                    #     f"{self.min_samples_per_window}. "
                    #     f"[path={path}]. Skipping... "
                    # )
                    continue
                graph = self._load_one(df, window_slice)
                # Store source file information
                graph.source_file = os.path.basename(path)
                # Store emotion names from first graph
                if not self.emotion_names and hasattr(graph, 'emotion_names'):
                    self.emotion_names = graph.emotion_names
                self.graphs.append(graph)
        
        source_desc = data_filepath if data_filepath else root_dir
        print(f"Loaded {len(self.graphs)} graphs from {source_desc}")
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_path)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        # Add index as an attribute so we can retrieve metadata after batching
        graph.idx = torch.tensor([idx], dtype=torch.long)
        return graph
    
    def _get_cache_path(self, root_dir: str, data_filepath: str, 
                       filter_subjects: list, filter_recordings: list,
                       exclude_subjects: list,
                       recursive: bool, ignore_dirs: list, file_list: list) -> str:
        """
        Generate a unique cache filename based on dataset parameters.
        Uses hash of configuration to ensure different parameters create different caches.
        """
        # Create a unique identifier based on parameters.
        # Include all graph-construction options and data source identity to avoid stale collisions.
        config_str = (
            f"kt={self.kt}_ks={self.ks}_tau={self.tau}_wl={self.window_length}_wo={self.window_overlap}"
            f"_edgew={self.use_edge_weights}_graphv={self.graph_version}_ewmode={self.edge_weight_mode}"
            f"_dropthr={self.dropping_emotion_threshold}"
        )
        
        if data_filepath is not None:
            abs_data_filepath = os.path.abspath(data_filepath)
            config_str += f"_file={abs_data_filepath}"
            try:
                stat = os.stat(abs_data_filepath)
                config_str += f"_fsize={stat.st_size}_fmtime_ns={stat.st_mtime_ns}"
            except OSError:
                # If stat fails, path string still participates in keying.
                pass
            config_str += (
                f"_subj={filter_subjects}_rec={filter_recordings}_exsubj={exclude_subjects}"
            )
        else:
            abs_root_dir = os.path.abspath(root_dir) if root_dir is not None else None
            config_str += (
                f"_root={abs_root_dir}_rec={recursive}_ignore={ignore_dirs}_files={file_list}"
                f"_exsubj={exclude_subjects}"
            )
        config_str += f"_feat={self.feature_columns}_targets={self.target_columns}_dropna={self.dropna_columns}"
        config_str += f"_expcol={self.experiment_type_column}_expvals={self.allowed_experiment_types}"
        config_str += f"_lqcol={self.label_quality_column}_lqvals={self.allowed_label_quality_values}"
        config_str += f"_tagg={self.target_aggregation}"
        config_str += f"_minsamples={self.min_samples_per_window}"
        config_str += (
            f"_usedistavg={self.use_distance_avg}"
            f"_usefixdur={self.use_fixation_duration}"
            f"_usedeltadistedge={self.use_delta_distance_edge_feature}"
            f"_usefixedges={self.use_fixation_edges}"
        )
        
        # Hash the configuration
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache filename includes parameters for readability
        cache_filename = f"dataset_kt{self.kt}_ks{self.ks}_tau{self.tau}_wl{self.window_length}_wo{self.window_overlap}_{config_hash}.pkl"
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
                    'dataframes': self.dataframes if self.use_single_file else {},
                    'kt': self.kt,
                    'ks': self.ks,
                    'window_length': self.window_length,
                    'window_overlap': self.window_overlap,
                    'min_samples_per_window': self.min_samples_per_window,
                    'feature_columns': self.feature_columns,
                    'use_distance_avg': self.use_distance_avg,
                    'use_fixation_duration': self.use_fixation_duration,
                    'use_delta_distance_edge_feature': self.use_delta_distance_edge_feature,
                    'use_fixation_edges': self.use_fixation_edges,
                }, f)
            print(f"Successfully cached {len(self.graphs)} graphs")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            # Continue without caching - don't fail the entire dataset loading

    def _load_df(self, path: str) -> pd.DataFrame:
        # Check if using single file mode
        if self.use_single_file and path in self.dataframes:
            df = self.dataframes[path].copy()
        else:
            df = pd.read_csv(path)

        if self.allowed_experiment_types and self.experiment_type_column in df.columns:
            df = df[df[self.experiment_type_column].isin(self.allowed_experiment_types)].reset_index(drop=True)

        if (
            self.label_quality_column
            and self.allowed_label_quality_values
            and self.label_quality_column in df.columns
        ):
            df = df[df[self.label_quality_column].isin(self.allowed_label_quality_values)].reset_index(drop=True)

        if self.exclude_subjects and "subject" in df.columns:
            df = df[~df["subject"].isin(self.exclude_subjects)].reset_index(drop=True)

        if len(df) == 0:
            return df

        df = prepare_hci_eye_tracking_signals(df)
        required_clean_cols = ["time-rel-seconds"] + self.feature_columns
        df = clean_dataset(
            df,
            required_cols=required_clean_cols,
            interpolation_cols=feature_interpolation_columns(self.feature_columns),
        )
        df = prepare_hci_eye_tracking_signals(df)

        if self.dropna_columns:
            missing = [col for col in self.dropna_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing configured dropna columns in {path}: {missing}")
            df = df.dropna(subset=self.dropna_columns).reset_index(drop=True)

        return df

    def _resolve_target_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve target columns either from config or inferred numeric emotion columns."""
        if self.target_columns is not None:
            missing = [col for col in self.target_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing configured target columns: {missing}")
            return self.target_columns
        return infer_default_target_columns(df)

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

        feature_cols = self.feature_columns
        df_window = df.loc[window_slice, :]
        n = len(df_window)
        t = torch.tensor(df_window["time-rel-seconds"].values, dtype=torch.float32)

        #### node features matrix X
        X = torch.tensor(df_window[feature_cols].values, dtype=torch.float32)

        # Resolve spatial coordinates by feature name to avoid silent column-order bugs.
        if "x-avg" not in feature_cols or "y-avg" not in feature_cols:
            raise ValueError(
                "Spatial kNN graph construction requires 'x-avg' and 'y-avg' in feature_columns. "
                f"Got feature_columns={feature_cols}"
            )
        x_idx = feature_cols.index("x-avg")
        y_idx = feature_cols.index("y-avg")
        x_values = X[:, x_idx]
        y_values = X[:, y_idx]

        distance_avg_values = None
        if self.use_delta_distance_edge_feature:
            if DISTANCE_AVG_COLUMN not in df_window.columns:
                raise ValueError(
                    f"{DISTANCE_AVG_COLUMN!r} is required when "
                    "use_delta_distance_edge_feature=True."
                )
            distance_avg_values = torch.tensor(
                pd.to_numeric(df_window[DISTANCE_AVG_COLUMN], errors="coerce").values,
                dtype=torch.float32,
            )

        def build_edge_features(edge_index: torch.Tensor, direction: float | None = None) -> torch.Tensor:
            """Build relation features for learned edge-weight MLPs."""
            src_idx, dst_idx = edge_index[0], edge_index[1]
            delta_t = t[dst_idx] - t[src_idx]
            delta_x = x_values[dst_idx] - x_values[src_idx]
            delta_y = y_values[dst_idx] - y_values[src_idx]
            distance = torch.sqrt(delta_x.square() + delta_y.square())
            features = [
                t[src_idx],
                t[dst_idx],
                delta_t,
                delta_x,
                delta_y,
                distance,
            ]
            if self.use_delta_distance_edge_feature:
                if distance_avg_values is None:
                    raise ValueError(
                        f"{DISTANCE_AVG_COLUMN!r} is required when "
                        "use_delta_distance_edge_feature=True."
                    )
                features.append(distance_avg_values[dst_idx] - distance_avg_values[src_idx])
            if direction is not None:
                features.append(torch.full_like(delta_t, float(direction)))
            return torch.stack(features, dim=1)

        def build_fixation_edge_index(window_df: pd.DataFrame) -> torch.Tensor:
            """Connect consecutive samples that share a fixation id, bidirectionally."""
            empty = torch.empty((2, 0), dtype=torch.long)
            if FIXATION_INDEX_COLUMN not in window_df.columns:
                return empty
            fixation_ids = pd.to_numeric(window_df[FIXATION_INDEX_COLUMN], errors="coerce").to_numpy()
            src_edges: list[int] = []
            dst_edges: list[int] = []
            for local_idx in range(len(fixation_ids) - 1):
                current_id = fixation_ids[local_idx]
                next_id = fixation_ids[local_idx + 1]
                if not np.isfinite(current_id) or not np.isfinite(next_id):
                    continue
                if current_id != next_id:
                    continue
                src_edges.extend([local_idx, local_idx + 1])
                dst_edges.extend([local_idx + 1, local_idx])
            if not src_edges:
                return empty
            return torch.tensor([src_edges, dst_edges], dtype=torch.long)
        
        #### Extract graph-level targets
        target_cols = self._resolve_target_columns(df_window)
        if target_cols:
            if self.target_aggregation == "mean":
                target_values = df_window[target_cols].mean(axis=0).values
            elif self.target_aggregation == "last":
                target_values = df_window[target_cols].iloc[-1].values
            else:
                raise ValueError(
                    f"Unsupported target_aggregation='{self.target_aggregation}'. "
                    "Use 'mean' or 'last'."
                )
            y = torch.tensor(target_values, dtype=torch.float32)
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
        if self.use_edge_weights:
            src, dst = edge_index_temporal[0], edge_index_temporal[1]
            df_temporal = (t[dst] - t[src]).abs()
            w_temporal = torch.exp(-df_temporal / self.tau)  # shape [E]

        if self.graph_version == "v2":
            temporal_forward_mask = edge_index_temporal[1] > edge_index_temporal[0]
            temporal_backward_mask = edge_index_temporal[1] < edge_index_temporal[0]
            edge_index_temporal_forward = edge_index_temporal[:, temporal_forward_mask]
            edge_index_temporal_backward = edge_index_temporal[:, temporal_backward_mask]
            if self.use_edge_weights:
                w_temporal_forward = w_temporal[temporal_forward_mask]
                w_temporal_backward = w_temporal[temporal_backward_mask]

        #### creating SPATIAL edge_index matrix
        xy = X[:, [x_idx, y_idx]].cpu().numpy()

        tree = KDTree(xy)  # use (x, y) for spatial neighbors
        _, idx = tree.query(xy, k=ks + 1)  # +1 to include self at idx[:, 0], then drop it
        neighbors = idx[:, 1:].reshape(-1)          # drop self, flatten
        src = np.repeat(np.arange(n), ks)            # each node repeated k times

        edge_index_spatial = torch.tensor(
            np.vstack([np.concatenate([src, neighbors]),
                    np.concatenate([neighbors, src])]),
            dtype=torch.long,
        )

        # remove duplicate spatial edges
        edge_index_spatial = torch.unique(edge_index_spatial, dim=1)
        if self.use_edge_weights:
            src, dst = edge_index_spatial[0], edge_index_spatial[1]
            df_spatial = (t[dst] - t[src]).abs()
            w_spatial = torch.exp(-df_spatial / self.tau)  # shape [E]

        data = HeteroData()
        data["node"].x = X
        data["node"].num_nodes = n
        if self.graph_version == "v1":
            data["node", "temporal", "node"].edge_index = edge_index_temporal
        else:
            data["node", "temporal_forward", "node"].edge_index = edge_index_temporal_forward
            data["node", "temporal_backward", "node"].edge_index = edge_index_temporal_backward
            if self.use_fixation_edges:
                edge_index_fixation = build_fixation_edge_index(df_window)
                data["node", "fixation", "node"].edge_index = edge_index_fixation
        data["node", "spatial", "node"].edge_index = edge_index_spatial
        if self.use_edge_weights:
            if self.graph_version == "v1":
                data["node", "temporal", "node"].edge_attr = w_temporal
            elif self.edge_weight_mode == "handcrafted":
                data["node", "temporal_forward", "node"].edge_attr = w_temporal_forward
                data["node", "temporal_backward", "node"].edge_attr = w_temporal_backward
            else:
                data["node", "temporal_forward", "node"].edge_attr = build_edge_features(
                    edge_index_temporal_forward,
                    direction=1.0,
                )
                data["node", "temporal_backward", "node"].edge_attr = build_edge_features(
                    edge_index_temporal_backward,
                    direction=-1.0,
                )
            if self.edge_weight_mode == "learned_signed" and self.graph_version == "v2":
                data["node", "spatial", "node"].edge_attr = build_edge_features(edge_index_spatial)
                if self.use_fixation_edges:
                    data["node", "fixation", "node"].edge_attr = build_edge_features(edge_index_fixation)
            else:
                data["node", "spatial", "node"].edge_attr = w_spatial
        
        # Add emotion targets if available
        if y is not None:
            data.y = y
        
        # Add subject and recording if available
        if subject is not None:
            data.subject = subject
        if recording is not None:
            data.recording = recording
        
        # Store emotion column names in data object for later use
        if target_cols:
            data.emotion_names = target_cols

        return data
