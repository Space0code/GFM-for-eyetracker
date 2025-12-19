"""
Tabular dataset preparation for classical ML models.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class TabularDataset:
    """Prepare eyetracking data as tabular format for classical ML."""
    
    def __init__(self, root_dir, file_list=None, window_length=10, test_size=0.2, random_state=42):
        """
        Args:
            root_dir: Directory with processed CSV files
            file_list: List of CSV files to load
            window_length: Window size in seconds for aggregation
            test_size: Fraction for test split
            random_state: Random seed
        """
        self.root_dir = root_dir
        self.window_length = window_length
        
        # Load files
        if file_list:
            files = [Path(root_dir) / f for f in file_list]
        else:
            files = list(Path(root_dir).glob('*.csv'))
        
        # Process all files into windows
        all_windows = []
        for file in files:
            windows = self._process_file(file)
            all_windows.extend(windows)
        
        # Convert to DataFrame and split
        df = pd.DataFrame(all_windows)
        
        # Separate features and targets
        emotion_cols = [c for c in df.columns if 'emotion' in c.lower()]
        feature_cols = [c for c in df.columns if c not in emotion_cols]
        
        X = df[feature_cols]
        y = df[emotion_cols]
        
        # Train/test split (keep as DataFrames)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.feature_names = feature_cols
        self.target_names = emotion_cols
        
        print(f"Train: {len(self.X_train)} samples | Test: {len(self.X_test)} samples")
        print(f"Features: {len(feature_cols)} | Targets: {len(emotion_cols)}")
    
    def _process_file(self, filepath):
        """Process single CSV into time windows with aggregated features."""
        df = pd.read_csv(filepath)
        df = df.dropna()
        
        if len(df) == 0:
            return []
        
        windows = []
        time_col = 'time-rel-seconds'
        
        # Create time windows
        max_time = df[time_col].max()
        start_time = 0
        
        while start_time < max_time:
            end_time = start_time + self.window_length
            window_data = df[(df[time_col] >= start_time) & (df[time_col] < end_time)]
            
            if len(window_data) > 10:  # Minimum points per window
                features = self._aggregate_window(window_data)
                windows.append(features)
            
            start_time += self.window_length
        
        return windows
    
    def _aggregate_window(self, window_df):
        """Aggregate window data into statistical features."""
        features = {}
        
        # Aggregate gaze features
        for col in ['x-avg', 'y-avg']:
            if col in window_df.columns:
                features[f'{col}_mean'] = window_df[col].mean()
                features[f'{col}_std'] = window_df[col].std()
                features[f'{col}_min'] = window_df[col].min()
                features[f'{col}_max'] = window_df[col].max()
        
        # Aggregate pupil features
        for col in ['pupil-size-left-avg', 'pupil-size-right-avg']:
            if col in window_df.columns:
                features[f'{col}_mean'] = window_df[col].mean()
                features[f'{col}_std'] = window_df[col].std()
        
        # Confidence features
        for col in ['confidence-gaze-left', 'confidence-gaze-right']:
            if col in window_df.columns:
                features[f'{col}_mean'] = window_df[col].mean()
        
        # Emotion targets (take mean across window)
        for col in window_df.columns:
            if 'emotion' in col.lower():
                features[col] = window_df[col].mean()
        
        return features
