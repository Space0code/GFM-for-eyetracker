import argparse
import os
import re
import sys
from glob import glob
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from data.hci_signals import prepare_hci_eye_tracking_signals

"""
Usage:
python src/data/data_preprocess.py
or
python src/data/data_preprocess.py --dataset cog-load
"""

SOURCE_DATA_DIR_ROOT = "data/raw-one-format/"
DEST_DATA_DIR_ROOT = "data/processed/"
SOURCE_DATA_DIRS = {
    # "cog-load": os.path.join(SOURCE_DATA_DIR_ROOT, "cog-load/"),
    "hci-tagging/emotion-elicitation": os.path.join(SOURCE_DATA_DIR_ROOT, "hci-tagging/Sessions/emotion-elicitation/"),
    "hci-tagging/video-tagging": os.path.join(SOURCE_DATA_DIR_ROOT, "hci-tagging/Sessions/video-tagging/"),
    "hci-tagging/image-tagging-1": os.path.join(SOURCE_DATA_DIR_ROOT, "hci-tagging/Sessions/image-tagging-1/"),
    "hci-tagging/image-tagging-2": os.path.join(SOURCE_DATA_DIR_ROOT, "hci-tagging/Sessions/image-tagging-2/"),
    # "deep_em": os.path.join(SOURCE_DATA_DIR_ROOT, "deep_em_classifier-data/"),
    # "eSEEd_v2": os.path.join(SOURCE_DATA_DIR_ROOT, "eSEEd_v2/"),
}

SUBJECT_PATTERN = re.compile(r"(P\d+)")


def _extract_subject_id(filename: str) -> str | None:
    """Extract subject id from filename, e.g. P20."""
    match = SUBJECT_PATTERN.search(filename)
    if match is None:
        return None
    return match.group(1)


def _infer_experiment_type_from_dir(dir_name: str) -> str | None:
    """Infer HCI experiment type from configured directory key."""
    if not dir_name.startswith("hci-tagging/"):
        return None
    return dir_name.split("/", 1)[1]


def _rebuild_hci_emotion_cache(
    processed_emotion_dir: str,
    cache_path: str,
    subset_rows: int = 10_000,
) -> None:
    """Rebuild merged HCI emotion cache CSVs from processed per-section files."""
    files = sorted(glob(os.path.join(processed_emotion_dir, "*.csv")))
    if not files:
        print(f"No files found for cache rebuild: {processed_emotion_dir}")
        return

    cache_file = Path(cache_path)
    subset_file = cache_file.with_name(f"{cache_file.stem}_subset_10K.csv")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        cache_file.unlink()
    if subset_file.exists():
        subset_file.unlink()

    wrote_header = False
    subset_written = 0

    for file_path in files:
        df = pd.read_csv(file_path)
        if df.empty:
            continue

        df.to_csv(cache_file, index=False, mode="a", header=not wrote_header)
        wrote_header = True

        if subset_written < subset_rows:
            take_n = min(subset_rows - subset_written, len(df))
            df.iloc[:take_n].to_csv(
                subset_file,
                index=False,
                mode="a",
                header=subset_written == 0,
            )
            subset_written += take_n

    if not wrote_header:
        print("Cache rebuild skipped: all source files were empty.")
        return

    print(f"Rebuilt merged cache: {cache_file}")
    print(f"Rebuilt subset cache: {subset_file}")


def preprocess_dir(
    dir_name: str,
    source_dir_path: str,
    source_root_path: str,
    dest_data_root: str,
    seen_subjects: set[str] | None = None,
) -> None:
    """Preprocess all CSV files under a directory tree."""
    if seen_subjects is None:
        seen_subjects = set()

    for file in sorted(os.listdir(source_dir_path)):
        file_path = os.path.join(source_dir_path, file)
        if file.endswith(".csv"):
            rel_dir = os.path.relpath(source_dir_path, source_root_path)
            rel_dir = "" if rel_dir == "." else rel_dir
            dest_data_dir = os.path.join(dest_data_root, rel_dir)

            experiment_group = rel_dir.split(os.sep)[0] if rel_dir else None
            subject_id = _extract_subject_id(file)
            if subject_id is not None and subject_id not in seen_subjects:
                if experiment_group:
                    print(f"Starting subject: {subject_id} ({experiment_group})")
                else:
                    print(f"Starting subject: {subject_id}")
                seen_subjects.add(subject_id)

            preprocess_file(dir_name, file, file_path, dest_data_dir)
        elif os.path.isdir(file_path):
            preprocess_dir(
                dir_name,
                file_path,
                source_root_path,
                dest_data_root,
                seen_subjects=seen_subjects,
            )


def preprocess_file(dir_name: str, filename: str, file_path: str, dest_data_dir: str) -> None:
    """Preprocess one CSV file and save it to destination directory."""
    df = pd.read_csv(file_path)

    if "hci-tagging" in dir_name:
        exp_type_from_dir = _infer_experiment_type_from_dir(dir_name)
        if "experiment-type" not in df.columns:
            df["experiment-type"] = exp_type_from_dir
        else:
            df["experiment-type"] = df["experiment-type"].fillna(exp_type_from_dir)
            df["experiment-type"] = df["experiment-type"].replace("", exp_type_from_dir)

        if "media-file" in df.columns:
            df["recording"] = df["media-file"]
            df = df.drop(columns=["media-file"])

    mandatory_columns = [
        "time-rel-seconds",
        "x-avg",
        "y-avg",
        "confidence-gaze-left",
        "confidence-gaze-right",
    ]
    signal_optional_columns = [
        "pupil-size-left-avg",
        "pupil-size-right-avg",
        "distance-left",
        "distance-right",
    ]
    hci_membership_columns = []
    if "hci-tagging" in dir_name:
        hci_membership_columns = [
            "fixation-index",
            "fixation-duration",
            "fixation",
        ]
    metadata_columns = [
        "subject",
        "recording",
        "section",
        "session-id",
        "experiment-type",
        "is-stimulus",
    ]

    for col in mandatory_columns:
        if col not in df.columns:
            raise ValueError(f"Mandatory column '{col}' is missing in the file: {file_path}")

    raw_validity_cols: list[str] = []
    if "hci-tagging" in dir_name:
        raw_validity_cols = [
            col
            for col in ["raw-validity-gaze-left", "raw-validity-gaze-right"]
            if col in df.columns
        ]

    emotion_cols = [col for col in df.columns if col.startswith("emotion-")]
    tag_cols = [col for col in df.columns if col.startswith("tag-")]
    label_cols = emotion_cols + tag_cols

    present_cols = (
        mandatory_columns
        + [col for col in signal_optional_columns if col in df.columns]
        + [col for col in hci_membership_columns if col in df.columns]
        + [col for col in metadata_columns if col in df.columns]
        + raw_validity_cols
        + label_cols
    )
    present_cols = list(dict.fromkeys(present_cols))
    df = df[present_cols].copy()

    df["confidence-gaze-left"] = pd.to_numeric(df["confidence-gaze-left"], errors="coerce")
    df["confidence-gaze-right"] = pd.to_numeric(df["confidence-gaze-right"], errors="coerce")

    if "cog-load" in dir_name:
        condition_good = (df["confidence-gaze-left"] == 1) & (df["confidence-gaze-right"] == 1)
    elif "hci-tagging" in dir_name:
        condition_good = (df["confidence-gaze-left"] == 1) & (df["confidence-gaze-right"] == 1)
    elif "eSEEd_v2" in dir_name:
        condition_good = (df["confidence-gaze-left"] >= 0.75) & (df["confidence-gaze-right"] >= 0.75)
    else:
        raise ValueError(f"Unknown dataset directory name: {dir_name}")

    condition_good = condition_good.fillna(False)

    metadata_present = [col for col in metadata_columns if col in df.columns]
    protected_cols = set(
        ["time-rel-seconds"]
        + metadata_present
        + raw_validity_cols
        + label_cols
        + [col for col in hci_membership_columns if col in df.columns]
    )

    cols_to_nan = [col for col in present_cols if col not in protected_cols]
    df.loc[~condition_good, cols_to_nan] = float("nan")
    if "hci-tagging" in dir_name:
        df = prepare_hci_eye_tracking_signals(df)

    valid_indices = df.index[condition_good]
    if len(valid_indices) == 0:
        print(f"Warning: no valid confidence rows in {file_path}; saving empty processed file.")
        df = df.iloc[0:0].copy()
        dest_filename = os.path.join(dest_data_dir, filename)
        os.makedirs(dest_data_dir, exist_ok=True)
        df.to_csv(dest_filename, index=False)
        print(f"Processed data saved to: {dest_filename}")
        return

    first_valid_index = valid_indices[0]
    last_valid_index = valid_indices[-1]
    df = df.iloc[first_valid_index:last_valid_index + 1]

    df = df.dropna(subset=["time-rel-seconds"]).reset_index(drop=True)
    df["time-rel-seconds"] = df["time-rel-seconds"] - df["time-rel-seconds"].min()

    for col in present_cols:
        if col in protected_cols:
            continue

        series = df[col].infer_objects(copy=False)
        if not pd.api.types.is_numeric_dtype(series):
            continue

        series = pd.to_numeric(series, errors="coerce")
        df[col] = series.interpolate(
            method="linear",
            limit_direction="both",
            limit=10,
            limit_area="inside",
        )
        if "pupil" in col:
            df[col] = df[col].rolling(window=3, min_periods=2, center=True).mean()

    if "hci-tagging" in dir_name:
        df = prepare_hci_eye_tracking_signals(df)

    dest_filename = os.path.join(dest_data_dir, filename)
    os.makedirs(dest_data_dir, exist_ok=True)
    df.to_csv(dest_filename, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="cog-load or hci-tagging to process only one dataset. Otherwise, process all datasets.",
    )
    args = ap.parse_args()

    for dir_name, dir_path in SOURCE_DATA_DIRS.items():
        if args.dataset is not None and dir_name != args.dataset:
            continue
        print("\n" + "=" * 50)
        print(f"Processing directory: {dir_name} at {dir_path}")
        preprocess_dir(
            dir_name=dir_name,
            source_dir_path=dir_path,
            source_root_path=dir_path,
            dest_data_root=os.path.join(DEST_DATA_DIR_ROOT, dir_name),
        )
        if dir_name == "hci-tagging/emotion-elicitation":
            _rebuild_hci_emotion_cache(
                processed_emotion_dir=os.path.join(DEST_DATA_DIR_ROOT, dir_name),
                cache_path=os.path.join(DEST_DATA_DIR_ROOT, "cached_hci_tagging_emotion.csv"),
            )
        print(f"\nFinished processing directory: {dir_name}")
        print("Files saved to:", os.path.join(DEST_DATA_DIR_ROOT, dir_name))
        print("=" * 50 + "\n")
