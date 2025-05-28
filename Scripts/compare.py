import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time
from scipy.signal import savgol_filter, butter, filtfilt
import numpy as np


parser = argparse.ArgumentParser(
    description="Compare the 2."
)
parser.add_argument(
    "file1",
    type=str,
    help="Path to the first input file."
)
parser.add_argument(
    "file2",
    type=str,
    help="Path to the second input file."
)

args = parser.parse_args()

file1 = args.file1
file2 = args.file2

fig, ax = plt.subplots()
# line_data, = ax.plot([], [], label='Données brutes', c="b", alpha=0.2)
line_genNN, = ax.plot([], [], label='Mutation génétique')
line_dqn, = ax.plot([], [], label='Apprentissage par renforcement')
# line_best, = ax.plot([], [], label='Meilleur par génération')
# ax.set_title('Comparaison entre Mutation génétique et Apprentissage par renforcement', fontsize=20)
ax.set_xlabel('Nombre de parties jouées', fontsize=20)
ax.set_ylabel('Lignes effacées', fontsize=20)
ax.legend(fontsize=20)

df = pd.read_csv(file1)
# df = df.head(800)
df["index"] = range(1, len(df) + 1)

df2 = pd.read_csv(file2)

df2['y_smooth'] = savgol_filter(df2['linesCleared'], window_length=33, polyorder=5)

new_df = pd.DataFrame(columns=["population", "best_score", "mean", "median", "worst_score"])

metric = "linesCleared"

for i in range(df["population"].max() + 1):
    population = df.loc[df["population"] == i]
    # Set i-th index of new_df["best_score"] to the population with the highest score
    new_df.loc[i, "best_score"] = population[metric].max()
    new_df.loc[i, "mean"] = population[metric].mean()
    best, median, worst = population[metric].quantile([0.75, 0.5, 0.25])

    new_df.loc[i, "best_quartile"] = best
    new_df.loc[i, "median"] = median
    new_df.loc[i, "worst_quartile"] = worst

    new_df.loc[i, "population"] = i

new_df["population"] = pd.to_numeric(new_df["population"], errors='coerce')
new_df["best_quartile"] = pd.to_numeric(new_df["best_quartile"], errors='coerce')
new_df["worst_quartile"] = pd.to_numeric(new_df["worst_quartile"], errors='coerce')
max_pop = new_df["population"].max()



x = df2["step"][500:]
y = df2["y_smooth"][500:]
coeffs = np.polyfit(x, y, 1)
line_reg = np.poly1d(coeffs)

plt.plot(x, line_reg(x), label="Régression linéaire (RL)", linestyle="--", color="red")

x1 = pd.to_numeric(new_df["population"].iloc[7:], errors='coerce')  # Convert to numeric, coercing errors to NaN
y1 = pd.to_numeric(new_df["mean"].iloc[7:], errors='coerce')        # Convert to numeric, coercing errors to NaN

# Drop NaN values and ensure x1 and y1 are the same size
valid_indices = ~x1.isna() & ~y1.isna()  # Keep only valid (non-NaN) indices
x1 = x1[valid_indices]
y1 = y1[valid_indices]

# Ensure x1 and y1 are the same size
min_length = min(len(x1), len(y1))
x1 = x1[:min_length]
y1 = y1[:min_length]

# Perform linear regression
coeffs1 = np.polyfit(x1, y1, 1)  # Linear regression (degree=1)
line_reg1 = np.poly1d(coeffs1)

plt.plot(x1 * len(df) / max_pop, line_reg1(x1), label="Régression linéaire (GM)", linestyle="--", color="blue")



# line_data.set_data(df['index'], df['linesCleared'])
line_genNN.set_data(new_df['population']  * len(df) / max_pop, new_df['mean'])
ax.fill_between(new_df['population']  * len(df) / max_pop,
                new_df['worst_quartile'],
                new_df['best_quartile'],
                color='b',
                alpha=0.1,
                label='50% des invididus se trouvent dans cette aire')
ax.legend(fontsize=20)
line_dqn.set_data(df2['step'], df2['y_smooth'])
# line_best.set_data(new_df['population'] * len(df) / max_pop, new_df['best_score'])

ax.relim()
ax.autoscale_view()

plt.show()
