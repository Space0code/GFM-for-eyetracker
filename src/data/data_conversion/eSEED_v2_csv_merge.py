"""
eSEED_v2 Dataset Preprocessing

Merges gaze, pupil, blinks, and annotation data for each recording.
Applies interpolation for small gaps and downsamples to 100Hz.

By default, saves the merged data to the data/raw/eSEEd_v2/merged_csv_files directory.

Dataset structure:
    - sample_{id}, id = 01 to 48
        - gaze_{i}.csv, i = 1 to 10 (10 recordings per sample)
        - pupil_{i}.csv, i = 1 to 10
        - blinks_{i}.csv, i = 1 to 10
        - annotation_{i}.csv, i = 1 to 10
        - questionnaires.csv
        - subject_info.csv
"""

import os
import pandas as pd
import numpy as np
import argparse


def merge_eyetracking_data(sample_dir, recording_id):
    """Merge gaze, pupil, blinks, and annotation data for a single recording."""
    
    # Load all data files
    gaze_df = pd.read_csv(os.path.join(sample_dir, f"gaze_{recording_id}.csv"))
    pupil_df = pd.read_csv(os.path.join(sample_dir, f"pupil_{recording_id}.csv"))
    blinks_df = pd.read_csv(os.path.join(sample_dir, f"blinks_{recording_id}.csv"))
    annotation_df = pd.read_csv(os.path.join(sample_dir, f"annotation_{recording_id}.csv"))
    
    # Rename timestamp columns to standard 'timestamp'
    gaze_df = gaze_df.rename(columns={'gaze_timestamp': 'timestamp'})
    pupil_df = pupil_df.rename(columns={'pupil_timestamp': 'timestamp'})
    
    # Merge gaze and pupil on timestamp
    merged_df = pd.merge(gaze_df, pupil_df, on='timestamp', how='outer', suffixes=('_gaze', '_pupil'))
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    # Initialize blink columns
    merged_df['blink_bool'] = False
    merged_df['blink_confidence'] = None
    merged_df['blink_index'] = None
    
    # Mark blink segments
    for _, blink in blinks_df.iterrows():
        mask = (merged_df['timestamp'] >= blink['start_timestamp']) & \
               (merged_df['timestamp'] <= blink['end_timestamp'])
        merged_df.loc[mask, 'blink_bool'] = True
        merged_df.loc[mask, 'blink_confidence'] = blink['confidence']
        merged_df.loc[mask, 'blink_index'] = blink['id']
    
    # Add annotation columns (constant values for entire recording)
    for col in annotation_df.columns:
        merged_df[col] = annotation_df[col].values[0]
    
    # Interpolate missing values in numeric columns (max 5 consecutive gaps)
    for col in merged_df.columns:
        if merged_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            merged_df[col] = merged_df[col].interpolate(method='linear', limit=5, limit_direction='both')
    
    return merged_df


def downsample_to_100hz(df):
    """Downsample data to 100Hz using timestamp-based resampling."""
    
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['time_seconds'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Create uniform 100Hz time grid (0.01 second intervals)
    max_time = df['time_seconds'].max()
    uniform_time = np.arange(0, max_time, 0.01)
    
    # Create new dataframe with uniform timestamps
    downsampled_df = pd.DataFrame({
        'time_seconds': uniform_time,
        'timestamp': uniform_time + df['timestamp'].iloc[0]
    })
    
    # Interpolate all numeric columns to the new time grid
    for col in df.columns:
        if col not in ['time_seconds', 'timestamp']:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                downsampled_df[col] = np.interp(uniform_time, df['time_seconds'], df[col])
            else:
                # For non-numeric columns, use nearest neighbor
                df_sorted = df[['timestamp', col]].sort_values('timestamp')
                downsampled_df[col] = pd.merge_asof(
                    downsampled_df[['timestamp']], 
                    df_sorted, 
                    on='timestamp', 
                    direction='nearest'
                )[col]
    
    downsampled_df = downsampled_df.drop(columns=['time_seconds'])
    return downsampled_df


def process_recording(sample_id, recording_id, data_dir, output_dir):
    """Process a single recording and save as separate file."""
    
    sample_dir = os.path.join(data_dir, f"sample_{sample_id:02d}")
    
    try:
        # Merge data for this recording
        merged_data = merge_eyetracking_data(sample_dir, recording_id)
        
        # Downsample to 100Hz
        merged_data = downsample_to_100hz(merged_data)
        
        # Save merged data for this recording
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sample_{sample_id:02d}_recording_{recording_id:02d}_merged.csv")
        merged_data.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        return True
    except FileNotFoundError as e:
        print(f"Warning: Missing file for sample_{sample_id:02d}, recording {recording_id}: {e}")
        return False


def process_sample(sample_id, data_dir, output_dir):
    """Process all recordings for a given sample."""
    
    processed_count = 0
    
    # Process each recording (1-10) separately
    for recording_id in range(1, 11):
        if process_recording(sample_id, recording_id, data_dir, output_dir):
            processed_count += 1
    
    if processed_count > 0:
        print(f"Successfully processed {processed_count}/10 recordings for sample_{sample_id:02d}")
    else:
        print(f"No recordings processed for sample_{sample_id:02d}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess eSEED_v2 eyetracking data')
    parser.add_argument('--data_dir', type=str, required=False, default="data/raw/eSEEd_v2/split_mat/export_csv",
                        help='Path to raw data directory (contains sample_XX folders)')
    parser.add_argument('--output_dir', type=str, required=False, default="data/raw/eSEEd_v2/merged_csv_files",
                        help='Path to output directory for merged csv files')
    parser.add_argument('--sample_ids', type=int, nargs='+', default=None,
                        help='Specific sample IDs to process (e.g., 1 2 3). If not provided, processes all (1-48)')
    
    args = parser.parse_args()
    
    # Determine which samples to process
    sample_ids = args.sample_ids if args.sample_ids else range(1, 49)
    
    # Process each sample
    for sample_id in sample_ids:
        print(f"\n{'='*60}")
        print(f"Processing sample {sample_id:02d}")
        print('='*60)
        process_sample(sample_id, args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
