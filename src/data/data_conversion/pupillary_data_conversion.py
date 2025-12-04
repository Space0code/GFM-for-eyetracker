#!/usr/bin/env python3
"""
pupillary_data_conversion.py

Load left-eye pupil data from a MATLAB .mat file and export per-subject CSVs
in a common GFM-style schema.

Expected .mat structure:
  - Variables: t0_control, t0_experimental, t1_control, ..., t4_experimental
  - Each is an array of shape (n_subjects, n_timepoints)

Output directory tree under --output:
  pupillary_data/
    t0_control/
      s_001.csv
      s_002.csv
      ...
    t0_experimental/
      s_001.csv
      ...
    ...

Usage:
python scripts/data_conversion/pupillary_data_conversion.py \
  --input data/raw/pupillary_data/left_eye_pupil_dataset.mat \
  --output data/raw-one-format/pupillary_data
"""

import os
import argparse

import numpy as np
import pandas as pd
import scipy.io


def convert_pupil_mat(mat_path: str, output_dir: str):
    """
    Load the .mat file at mat_path, then for each variable of form
    t{timepoint}_{condition}, write per-subject CSVs to output_dir/timepoint_condition/s_XXX.csv
    """
    # Load .mat (squeeze_me to drop singleton dims)
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    
    # Process each key
    for var_name, arr in mat.items():
        # Skip private/meta fields
        if not var_name.startswith('t') or not isinstance(arr, np.ndarray):
            continue
        
        # Expect var_name like "t0_control"
        try:
            timepoint, condition = var_name.split('_', 1)
        except ValueError:
            print(f"Skipping unexpected variable '{var_name}'")
            continue
        
        # Ensure 2D array: (n_subjects, n_timepoints)
        if arr.ndim != 2:
            print(f"Variable '{var_name}' has shape {arr.shape}, skipping.")
            continue
        
        n_subjects, n_samples = arr.shape
        
        # Create folder for this condition
        out_subdir = os.path.join(output_dir, var_name)
        os.makedirs(out_subdir, exist_ok=True)
        
        # For each row (subject), build a DataFrame and write CSV
        for subj_idx in range(n_subjects):
            subj_id = f"s_{subj_idx+1:03d}"
            series = arr[subj_idx, :]
            
            # Build GFM-style DataFrame:
            # - time-rel-seconds: sample time in seconds (0.000, 0.020, 0.040, ...)
            # - pupil-diameter-left: the value
            df = pd.DataFrame({
                'time-rel-seconds': np.arange(n_samples) * 0.020,
                'pupil-diameter-left': series.astype(float)
            })
            
            out_path = os.path.join(out_subdir, f"{subj_id}.csv")
            df.to_csv(out_path, index=False)
            print(f"Wrote {out_path} ({n_samples} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MATLAB left-eye pupil data to per-subject GFM CSVs"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to left_eye_pupil_dataset.mat"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Base directory for output CSV tree"
    )
    args = parser.parse_args()
    
    convert_pupil_mat(args.input, args.output)