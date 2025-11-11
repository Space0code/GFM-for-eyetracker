# cog_load_conversion.py

"""
Usage:
python scripts/data_conversion/cog_load_conversion.py \
  --spec specifications/cog_load_spec.yaml \
  --input data/raw/cog-load \
  --output data/raw-one-format/cog-load
"""

import os
import yaml
import pandas as pd
import argparse

def load_spec(spec_path):
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    return spec.get('mappings', {})

def convert_cog_load(input_dir, spec, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for subj in os.listdir(input_dir):
        subj_path = os.path.join(input_dir, subj)
        tobii_dir = os.path.join(subj_path, "tobii")
        if not os.path.isdir(tobii_dir):
            continue

        for filename in os.listdir(tobii_dir):
            if not filename.endswith('.tsv'):
                continue
            input_path = os.path.join(tobii_dir, filename)

            # 1) Find header row index by detecting "TimeStamp" start
            header_row_idx = None
            with open(input_path, 'r') as f:
                for i, line in enumerate(f):
                    if line.startswith("TimeStamp"):
                        header_row_idx = i
                        break
            if header_row_idx is None:
                print(f"Header row not found in {input_path}, skipping.")
                continue

            # 2) Read TSV using detected header row
            df = pd.read_csv(
                input_path,
                sep='\t',
                header=header_row_idx,
                dtype=str
            ).reset_index(drop=True)

            # 3) Rename and filter columns based on spec
            rename_map = {old: new for old, new in spec.items() if new}
            df = df.rename(columns=rename_map)
            df = df[list(rename_map.values())]

            # 4) Convert timestamps from milliseconds to seconds
            if 'time-abs-seconds' in df.columns:
                df['time-abs-seconds'] = df['time-abs-seconds'].astype(float) / 1e3
            if 'time-rel-seconds' in df.columns:
                df['time-rel-seconds'] = df['time-rel-seconds'].astype(float) / 1e3

            # 5) Write out CSV
            subj_id = subj
            base, _ = os.path.splitext(filename)
            out_name = f"{subj_id}.csv"
            out_path = os.path.join(output_dir, out_name)
            df.to_csv(out_path, index=False)
            print(f"Converted {input_path} -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Cog-Load Tobii TSV data to GFM CSV format"
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to cog_load_spec.yaml"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to cog-load root directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output directory for CSV files"
    )
    args = parser.parse_args()

    mappings = load_spec(args.spec)
    convert_cog_load(args.input, mappings, args.output)