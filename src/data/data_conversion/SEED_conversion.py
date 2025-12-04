"""
Usage:
python scripts/data_conversion/SEED_conversion.py \
    --spec specifications/SEED_to_GFM_spec.yaml \
    --input data/raw/SEED \
    --output data/raw-one-format/SEED
"""

import yaml
import pandas as pd
import argparse
import os

def load_spec(spec_path):
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    return spec.get('mappings', {})


def convert_files(input_dir, sheet_name, spec, output_csv):
    # Find all .xls files in the input directory
    xls_files = [f for f in os.listdir(input_dir) if f.endswith('.xls')]
    
    if not xls_files:
        print("No .xls files found in the input directory.")
        return

    for xls_file in xls_files:
        input_xls = os.path.join(input_dir, xls_file)
        if sheet_name is None:
            for sheet in pd.ExcelFile(input_xls).sheet_names:
                output_file = os.path.join(output_csv, f"{os.path.splitext(xls_file)[0]}_{sheet}.csv")
                convert_individual_file(input_xls, sheet, spec, output_file)
        else:
            if sheet_name not in pd.ExcelFile(input_xls).sheet_names:
                print(f"Sheet '{sheet_name}' not found in {xls_file}. Skipping.")
                continue
            else:
                print(f"Converting {xls_file} from sheet '{sheet_name}'")
                output_file = os.path.join(output_csv, f"{os.path.splitext(xls_file)[0]}.csv")
                convert_individual_file(input_xls, sheet_name, spec, output_file)

def convert_individual_file(input_xls, sheet_name, spec, output_csv):
    # Read Excel (all columns as-is)
    df = pd.read_excel(input_xls, sheet_name=sheet_name)

    # Rename columns based on spec
    rename_map = {old: new for old, new in spec.items() if new}
    df = df.rename(columns=rename_map)

    # Drop any columns not in the spec target list
    target_cols = list(rename_map.values())
    df = df[target_cols]

    # Write out CSV
    df.to_csv(output_csv, index=False)
    print(f"Converted {input_xls} -> {output_csv} with {len(rename_map)} columns renamed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SEED XLS eye-tracking data to GFM CSV format')
    parser.add_argument('--spec', required=True, help='Path to SEED_to_GFM_spec.yaml')
    parser.add_argument('--input', required=True, help='Path to input dir with .xls files')
    parser.add_argument('--sheet', default=None, help='Name of the sheet to convert (default: Sheet1)')
    parser.add_argument('--output', required=True, help='Path to output .csv file')

    args = parser.parse_args()
    mappings = load_spec(args.spec)
    convert_files(args.input, args.sheet, mappings, args.output)