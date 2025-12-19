"""
Convert eSEED_v2 merged CSV files to GFM format.

Reads from data/raw/eSEEd_v2/merged_csv_files and saves to data/raw-one-format/eSEEd_v2.

Example usage:
# Convert all files with default paths
python src/data/data_conversion/eSEED_v2_conversion.py

# Or specify custom paths
python src/data/data_conversion/eSEED_v2_conversion.py \
  --input_dir data/raw/eSEEd_v2/merged_csv_files \
  --output_dir data/raw-one-format/eSEEd_v2 \
  --spec_file specifications/eSEEd_v2_to_GFM_spec.yaml
  --sample_ids 1 2 3
"""

import os
import pandas as pd
import yaml
import argparse
from pathlib import Path


def load_mapping(spec_file):
    """Load column mapping from specification file.
    
    Returns a dictionary where keys are source columns and values are lists of target columns.
    This allows one-to-many mappings.
    """
    with open(spec_file, 'r') as f:
        # Use yaml.load_all or parse manually to handle duplicate keys
        content = f.read()
    
    # Parse YAML manually to handle duplicate keys
    mappings = {}
    in_mappings = False
    
    for line in content.split('\n'):
        line = line.strip()
        if line == 'mappings:':
            in_mappings = True
            continue
        
        if in_mappings and line and not line.startswith('#'):
            # Parse "source": "target" # comment
            if ':' in line:
                parts = line.split(':', 1)
                source = parts[0].strip().strip('"')
                target_part = parts[1].split('#')[0].strip().strip('"')
                
                if source and target_part:
                    # Add to list of targets for this source
                    if source not in mappings:
                        mappings[source] = []
                    mappings[source].append(target_part)
    
    return mappings


def convert_to_gfm(input_file, output_file, mapping):
    """Convert a single CSV file to GFM format.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        mapping: Dictionary with source columns as keys and lists of target columns as values
    """
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Create new dataframe with mapped columns
    gfm_df = pd.DataFrame()
    
    for source_col, target_cols in mapping.items():
        # target_cols is a list, allowing one-to-many mappings
        if source_col in df.columns:
            # Map the same source column to multiple target columns
            for target_col in target_cols:
                gfm_df[target_col] = df[source_col]
        else:
            # Column not in source, fill all targets with None
            for target_col in target_cols:
                print(f"Warning: Column '{source_col}' not found in {os.path.basename(input_file)}. Filling '{target_col}' with None.")
                gfm_df[target_col] = None
    
    # Add time-rel-seconds column by subtracting minimum from time-abs-seconds
    if 'time-abs-seconds' in gfm_df.columns:
        time_rel = gfm_df['time-abs-seconds'] - gfm_df['time-abs-seconds'].min()
        # Insert as first column
        gfm_df.insert(0, 'time-rel-seconds', time_rel)
    
    if "x-left" in gfm_df.columns and "x-right" in gfm_df.columns:
        gfm_df["x-avg"] = gfm_df[["x-left", "x-right"]].mean(axis=1)
    if "y-left" in gfm_df.columns and "y-right" in gfm_df.columns:
        gfm_df["y-avg"] = gfm_df[["y-left", "y-right"]].mean(axis=1)
    
    # Save to output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    gfm_df.to_csv(output_file, index=False)
    print(f"Converted: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")


def main():
    parser = argparse.ArgumentParser(description='Convert eSEED_v2 data to GFM format')
    parser.add_argument('--input_dir', type=str, 
                        default='data/raw/eSEEd_v2/merged_csv_files',
                        help='Input directory with merged CSV files')
    parser.add_argument('--output_dir', type=str, 
                        default='data/raw-one-format/eSEEd_v2',
                        help='Output directory for GFM format files')
    parser.add_argument('--spec_file', type=str, 
                        default='specifications/eSEEd_v2_to_GFM_spec.yaml',
                        help='Path to specification file')
    parser.add_argument('--sample_ids', type=int, nargs='+', default=None,
                        help='Specific sample IDs to process (e.g., 1 2 3). If not provided, processes all (1-48)')
        
    args = parser.parse_args()
    
    # Load mapping
    mapping = load_mapping(args.spec_file)
    
    # Get all CSV files from input directory
    input_files = list(Path(args.input_dir).glob('*.csv'))
    
    if not input_files:
        print(f"No CSV files found in {args.input_dir}")
        return
    
    # Convert each file
    for input_file in sorted(input_files):
        # If sample_ids specified, filter files
        if args.sample_ids is not None:
            sample_id_strs = [f"sample_{sid:02d}_" for sid in args.sample_ids]
            if not any(sid_str in input_file.name for sid_str in sample_id_strs):
                continue
        
        output_file = os.path.join(args.output_dir, input_file.name)
        convert_to_gfm(str(input_file), output_file, mapping)
    
    print(f"\nConverted {len(input_files)} files successfully")


if __name__ == '__main__':
    main()


