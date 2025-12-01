import argparse
import os
import pandas as pd

"""
Usage:
python scripts/data_preprocess.py 
or
python scripts/data_preprocess.py --dataset cog-load
"""

SOURCE_DATA_DIR_ROOT = "data/raw-one-format/"
DEST_DATA_DIR_ROOT = "data/processed/"
SOURCE_DATA_DIRS = {
    "cog-load": os.path.join(SOURCE_DATA_DIR_ROOT, "cog-load/"),
    "hci-tagging": os.path.join(SOURCE_DATA_DIR_ROOT, "hci-tagging/"),
}

def preprocess_dir(dir_name, source_dir_path, dest_data_dir):
    """
    Preprocess the data in the given directory.
    This function is not used in this script but can be useful for future extensions.
    """
    for file in os.listdir(source_dir_path):
        if file.endswith(".csv"):
            print(f"Processing file: {file}")
            filename = os.path.join(source_dir_path, file)
            preprocess_file(dir_name, file, filename, dest_data_dir)
        # elif is directory, recursion
        elif os.path.isdir(os.path.join(source_dir_path, file)):
            preprocess_dir(dir_name, os.path.join(source_dir_path, file), dest_data_dir)

def preprocess_file(dir_name, filename, file_path, dest_data_dir):

    df = pd.read_csv(file_path)
    dfOG = df.copy()
    # print("COLUMNS of original df:", df.columns)

    MANDATORY_COLUMNS = [
        "time-rel-seconds",
        "x-avg",
        "y-avg",
        "confidence-gaze-left",
        "confidence-gaze-right",
    ]
    OPTIONAL_COLUMNS = [
        "pupil-size-left-avg",
        "pupil-size-right-avg",
    ]
    # check if all mandatory columns are present
    for col in MANDATORY_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Mandatory column '{col}' is missing in the file: {file_path}")

    present_cols = MANDATORY_COLUMNS + [col for col in OPTIONAL_COLUMNS if col in df.columns]

    df = dfOG[present_cols].copy()
    # print("COLUMNS of processed df:", df.columns)

    if "cog-load" in dir_name:
        # rows that have both confidence = 0, put x-avg and y-avg to float('nan')
        condition_good = (df["confidence-gaze-left"] == 1) & (df["confidence-gaze-right"] == 1)
    elif "hci-tagging" in dir_name:
        # rows that have at least one confidence > 0, put x-avg and y-avg to float('nan')
        condition_good = (df["confidence-gaze-left"] == 0) & (df["confidence-gaze-right"] == 0)

    cols_to_nan = [col for col in present_cols if col != "time-rel-seconds"]
    df.loc[~condition_good, cols_to_nan] = float('nan')

    # drop all rows until the first row with good confidence
    first_valid_index = df[condition_good].index[0]
    last_valid_index = df[condition_good].index[-1]
    df = df.iloc[first_valid_index:last_valid_index + 1]

    # drop where time is nan
    df = df.dropna(subset=["time-rel-seconds"]).reset_index(drop=True)

    df["time-rel-seconds"] = df["time-rel-seconds"] - df["time-rel-seconds"].min()

    df_to_compare = df.copy()

    # interpolate missing values but only a few points in each direction (limit=30 => max 1s window interpolation)
    for col in present_cols:
        if col != "time-rel-seconds":
            df[col] = df[col].interpolate(method="linear", limit_direction="both", limit=10, limit_area="inside")
            if "pupil" in col:
                # don't smooth (x,y) coordinates because we could lose subtle gaze path details
                df[col] = df[col].rolling(window=3, min_periods=2, center=True).mean() 

    # stime = 1081.11
    # etime = 1081.42
    # print("Sample df_to_compare:")
    # print(df_to_compare.loc[df_to_compare["time-rel-seconds"].between(stime, etime), :])
    # print("Sample df:")
    # print(df.loc[df["time-rel-seconds"].between(stime, etime), :])

    # plot comparision of original and processed data
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(df["time-rel-seconds"], df["x-avg"], label="Processed x-avg", color='orange')
    # plt.plot(df_to_compare["time-rel-seconds"], df_to_compare["x-avg"], label="Original x-avg", alpha=0.5)
    # plt.title("X Average Comparison")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("X Average")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # DON'T normalize x-avg and y-avg (DATA LEAK) -- normalize later in data loader
    # screen_min_x = df["x-avg"].min()
    # screen_max_x = df["x-avg"].max()
    # screen_min_y = df["y-avg"].min()
    # screen_max_y = df["y-avg"].max()
    # df["x-avg"] = (df["x-avg"] - screen_min_x) / (screen_max_x - screen_min_x)
    # df["y-avg"] = (df["y-avg"] - screen_min_y) / (screen_max_y - screen_min_y)


    # print("Processed DataFrame:")
    # print(df.head(3))
    # print(df.describe())

    # save the processed dataframe to a new CSV file
    dest_filename = os.path.join(dest_data_dir, filename)
    os.makedirs(dest_data_dir, exist_ok=True)
    df.to_csv(dest_filename, index=False)
    print(f"Processed data saved to: {dest_filename}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None,
                    help="cog-load or hci-tagging to process only one dataset. Otherwise, process all datasets.")
    args = ap.parse_args()

    for dir_name, dir_path in SOURCE_DATA_DIRS.items():
        if args.dataset is not None and dir_name != args.dataset:
            continue
        print("\n" + "=" * 50)
        print(f"Processing directory: {dir_name} at {dir_path}")
        preprocess_dir(dir_name, dir_path, os.path.join(DEST_DATA_DIR_ROOT, dir_name))
