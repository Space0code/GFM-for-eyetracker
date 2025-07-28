import pandas as pd

filename = "data/raw-one-format/cog-load/s_001_user_2024-08-08$14-22-19-252173_2024-08-08$15-20-22-460601.csv"
df = pd.read_csv(filename, sep=",")
dfOG = df.copy()

# filter the columns, preserve only time-rel-seconds, x-avg, y-avg and confidence-left, confidence-right
df = dfOG[["time-rel-seconds", "x-avg", "y-avg", "confidence-gaze-left", "confidence-gaze-right"]]

# rows that have both confidence = 0, put x-avg and y-avg to float('nan')
df.loc[(df["confidence-gaze-left"] == 0) & (df["confidence-gaze-right"] == 0), ["x-avg", "y-avg"]] = float('nan')

# drop all rows until the first row with confidence > 0
first_valid_index = df[(df["confidence-gaze-left"] > 0) | (df["confidence-gaze-right"] > 0)].index[0]
last_valid_index = df[(df["confidence-gaze-left"] > 0) | (df["confidence-gaze-right"] > 0)].index[-1]
df = df.iloc[first_valid_index:last_valid_index + 1].reset_index(drop=True)

df["time-rel-seconds"] = df["time-rel-seconds"] - df["time-rel-seconds"].min()

df_to_compare = df.copy()

# interpolate missing values in x-avg and y-avg but only a few points in each direction (limit=30 => max 1s window interpolation)
df["x-avg"] = df["x-avg"].interpolate(method="linear", limit_direction="both", limit=30, limit_area="inside")
df["y-avg"] = df["y-avg"].interpolate(method="linear", limit_direction="both", limit=30, limit_area="inside")

# smooth out the x-avg and y-avg using a rolling window 
df["x-avg"] = df["x-avg"].rolling(window=3, min_periods=2, center=True).mean()
df["y-avg"] = df["y-avg"].rolling(window=3, min_periods=2, center=True).mean()

# plot comparision of original and processed data
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df_to_compare["time-rel-seconds"], df_to_compare["x-avg"], label="Original x-avg", alpha=0.5)
plt.plot(df["time-rel-seconds"], df["x-avg"], label="Processed x-avg", color='orange')
plt.title("X Average Comparison")
plt.xlabel("Time (seconds)")
plt.ylabel("X Average")
plt.legend()
plt.tight_layout()
plt.show()

# normalize x-avg and y-avg 
screen_min_x = 0
screen_max_x = 1920
screen_min_y = 0
screen_max_y = 1080
df["x-avg"] = (df["x-avg"] - screen_min_x) / (screen_max_x - screen_min_x)
df["y-avg"] = (df["y-avg"] - screen_min_y) / (screen_max_y - screen_min_y)


print(df.head())

print(df.describe())