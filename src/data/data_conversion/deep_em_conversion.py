# deep_em_conversion.py

"""
Usage:
python scripts/data_conversion/deep_em_conversion.py \
    --spec specifications/deep_em_to_GFM_spec.yaml \
    --input data/raw/deep_em_classifier-data \
    --output data/raw-one-format/deep_em_classifier-data
"""

import os
import csv
import yaml
import pandas as pd
import argparse
from typing import Dict, List, Tuple


def load_spec(spec_path: str) -> Dict[str, str]:
    """Load YAML spec as a flat mapping of target->source columns.
    Notes:
    - This script expects a simple top-level mapping like: target_col: source_col
    - Any mapping whose source column is not present in the ARFF data is ignored.
    - Metadata comments in YAML (lines starting with '#') are ignored by the YAML loader.
    """
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f) or {}
    return {str(k): str(v) for k, v in spec.items() if isinstance(k, str) and isinstance(v, str)}


def read_arff_to_dataframe(file_path: str) -> pd.DataFrame:
    """Minimal ARFF reader -> DataFrame.
    - Parses @attribute names (assumes simple names without spaces or quoted names).
    - Reads rows after @data using CSV parsing (comma-separated).
    - Returns a DataFrame with all values as strings; downstream casts selected columns.
    """
    attributes: List[str] = []
    data_rows: List[List[str]] = []
    in_data = False

    with open(file_path, "r", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            lower = line.lower()
            if not in_data:
                if lower.startswith("@attribute"):
                    # Example: @ATTRIBUTE time INTEGER
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[1].strip("'\"")
                        attributes.append(name)
                elif lower.startswith("@data"):
                    in_data = True
                # else: ignore @relation and others
            else:
                if line.startswith("{"):
                    # Sparse ARFF not supported in this minimal parser
                    continue
                # Parse CSV line
                reader = csv.reader([line])
                row = next(reader)
                # Normalize row length to attributes length
                if len(row) < len(attributes):
                    row += [""] * (len(attributes) - len(row))
                elif len(row) > len(attributes):
                    row = row[: len(attributes)]
                data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=attributes)
    return df


def convert_deep_em(input_dir: str, spec: Dict[str, str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    print(f"Converting Deep-EM data from {input_dir} to {output_dir} per spec mapping...")
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(".arff"):
                continue
            input_path = os.path.join(root, filename)

            # Read ARFF
            original_df = read_arff_to_dataframe(input_path)

            # Prepare adjusted confidence (set to 0 for noise code 4 when available)
            adjusted_conf = None
            codes = None
            if "handlabeller_final" in original_df.columns:
                codes = pd.to_numeric(original_df["handlabeller_final"], errors="coerce")
            if "confidence" in original_df.columns:
                adjusted_conf = pd.to_numeric(original_df["confidence"], errors="coerce")
                if codes is not None:
                    adjusted_conf = adjusted_conf.mask(codes == 4, 0)

            # Build output dataframe by copying each target from its source
            data_map: Dict[str, pd.Series] = {}
            target_cols: List[str] = []
            for target, source in spec.items():
                if source not in original_df.columns and not (source == "confidence" and adjusted_conf is not None):
                    continue
                if source == "confidence" and adjusted_conf is not None:
                    series = adjusted_conf
                else:
                    series = original_df[source]
                data_map[target] = series
                if target not in target_cols:
                    target_cols.append(target)

            if not data_map:
                print(f"No matching columns per spec in {input_path}, skipping.")
                continue

            df = pd.DataFrame(data_map)

            # Time conversion: microseconds -> seconds if mapped to 'time-abs-seconds'
            if "time-abs-seconds" in df.columns:
                df["time-abs-seconds"] = pd.to_numeric(df["time-abs-seconds"], errors="coerce") / 1e6

            # Derive events from handlabeller_final (1=fix, 2=saccade, 3=smooth pursuit)
            if codes is not None:
                df["fixation"] = codes == 1
                df["saccade"] = codes == 2
                df["smooth-pursuit"] = codes == 3
            else:
                df["fixation"] = False
                df["saccade"] = False
                df["smooth-pursuit"] = False

            for col in ["fixation", "saccade", "smooth-pursuit"]:
                if col not in target_cols:
                    target_cols.append(col)

            # Reorder columns to target_cols
            df = df[target_cols]

            # Determine output path preserving relative structure
            rel = os.path.relpath(input_path, input_dir)
            rel_no_ext = os.path.splitext(rel)[0]
            out_path = os.path.join(output_dir, f"{rel_no_ext}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            df.to_csv(out_path, index=False)
            converted += 1
            # print(f"Converted {input_path} -> {out_path}")
    print(f"Converted {converted} files to {output_dir}.")
    
    if converted == 0:
        print(f"No ARFF files found under {input_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Deep-EM data to GFM CSV format per spec mapping"
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to deep_em_to_GFM_spec.yaml"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Deep-EM input directory containing CSV/TSV files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output directory for converted CSV files"
    )
    args = parser.parse_args()

    mappings = load_spec(args.spec)
    convert_deep_em(args.input, mappings, args.output)
