"""

Uage:
python scripts/data_conversion/hci_tagging_conversion.py \
    --spec specifications/hci_to_GFM_spec.yaml \
    --input data/raw/hci-tagging \
    --output data/raw-one-format/hci-tagging
"""


import yaml
import pandas as pd
import argparse
import os
import re

PATTERN = re.compile(r'.*-All-Data-.*\.tsv$')

def load_spec(spec_path):
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    return spec.get('mappings', {})

def convert_tsv(input_tsv, spec, output_csv):
    df = pd.read_csv(input_tsv, sep='\t', header=None, dtype=str)
    # Find the row index where "Timestamp" appears (assume it's in the first column)
    header_row_idx = df[df.iloc[:, 0] == "Timestamp"].index[0]
    # Set column names from that row
    df.columns = df.iloc[header_row_idx]
    # Data starts from the next row
    df = df.iloc[header_row_idx + 1:].reset_index(drop=True)

    # print columns for debugging
    #print("HEAD OLD", df.head(3))
    rename_map = {old: new for old, new in spec.items() if new}
    df = df.rename(columns=rename_map)
    target_cols = list(rename_map.values())
    df = df[target_cols]
    if 'time-abs-seconds' in df.columns:
        df['time-abs-seconds'] = df['time-abs-seconds'].astype(float) / 1e6
    if 'time-rel-seconds' in df.columns:
        df['time-rel-seconds'] = df['time-rel-seconds'].astype(float) / 1e6
    df.to_csv(output_csv, index=False)
    print(f"Converted {input_tsv} -> {output_csv} with {len(target_cols)} columns.")
    #print("HEAD NEW", df.head(3))
    

def process_folder(input_root, output_root, spec):
    for dirpath, _, filenames in os.walk(input_root):
        rel_dir = os.path.relpath(dirpath, input_root)
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        for fname in filenames:
            if PATTERN.match(fname):
                print(f"Processing {fname} in {dirpath}")
                if not fname.endswith('.tsv'):
                    print(f"Skipping {fname}, not a TSV file.")
                    continue
                in_path = os.path.join(dirpath, fname)
                out_fname = os.path.splitext(fname)[0] + '.csv'
                out_path = os.path.join(out_dir, out_fname)
                convert_tsv(in_path, spec, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch convert Tobii TSV eye-tracking data to GFM CSV format')
    parser.add_argument('--spec', required=True, help='Path to TOBII_to_GFM_spec.yaml')
    parser.add_argument('--input', required=True, help='Path to input folder')
    parser.add_argument('--output', required=True, help='Path to output folder')
    args = parser.parse_args()
    mappings = load_spec(args.spec)
    process_folder(args.input, args.output, mappings)
