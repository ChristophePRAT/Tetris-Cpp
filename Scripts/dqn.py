import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt
import argparse

# file_name is the first argument parsed

parser = argparse.ArgumentParser(description="Plot scores from a CSV file of a DQN.")
parser.add_argument("file_name", type=str, help="Path to the CSV file containing scores.")

file_name = parser.parse_args().file_name


df = pd.read_csv(file_name)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Example parameters
cutoff = 0.05  # Adjust for more/less smoothing
fs = 1.0       # Sampling frequency
df['y_smooth_filter'] = butter_lowpass_filter(df['linesCleared'], cutoff, fs)

df['y_smooth'] = savgol_filter(df['linesCleared'], window_length=51, polyorder=5)

df["smooth"] = df['linesCleared'].ewm(span=50).mean()
df["smooth_simple"] = df['linesCleared'].rolling(window=30, center=True).mean()

# Smooth the curve using an exponential moving average
# plt.plot(df["step"], df["score"], label="Scores", alpha=1)
# 2 plots side by side

# f = plt.figure()
# f.add_subplot(1, 1, 1)
# plt.plot(df["step"], df["score"], label="Scores", alpha=1)
# f.add_subplot(1, 2, 2)
plt.plot(df["step"], df["linesCleared"], c="b", alpha=0.3)
# plt.plot(df["step"], df["smooth_simple"], label="Simple Moving Average", c="r", alpha=1.0)
# plt.plot(df["step"], df["smooth"], label="Exponential Moving Average", alpha=1.0)
plt.plot(df["step"], df["smooth_simple"], label="Évolution moyenne", alpha=1.0)
# plt.plot(df["step"], df["y_smooth_filter"], label="Butterworth Lowpass Filter", alpha=1.0)

plt.xlabel("Itération")
plt.ylabel("Lignes effacées")
plt.legend()
plt.show()

# df["index"] = range(1, len(df) + 1)
# new_df = pd.DataFrame(columns=["population", "best_score", "mean", "median", "worst_score"])

# for i in range(df["population"].max() + 1):
    # population = df.loc[df["population"] == i]

    # Set i-th index of new_df["best_score"] to the population with the highest score]
    # new_df.loc[i, "best_score"] = population["score"].max()
    # new_df.loc[i, "mean"] = population["score"].mean()
    # new_df.loc[i, "median"] = population["score"].median()
    # new_df.loc[i, "population"] = i
    # new_df.loc[i, "worst_score"] = population["score"].min()
# plt.plot(new_df["population"] * 20, new_df["best_score"], label="Best score")
# plt.plot(new_df["population"] * 20, new_df["mean"], label="Mean")
# plt.plot(new_df["population"] * 20, new_df["median"], label="Median")
# plt.plot(new_df["population"] * 20, new_df["worst_score"], label="Worst score")
# plt.plot(df["index"], df["score"], label="All scores", alpha=0.3)
# plt.legend()
# plt.show()
