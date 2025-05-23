import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time

# Parse the arguments which are the file path
parser = argparse.ArgumentParser(description='Real-time plot of the scores')
parser.add_argument('file', type=str, help='Path to the CSV file')
args = parser.parse_args()

# Path to your CSV file
csv_file = args.file

fig, ax = plt.subplots()
line_data, = ax.plot([], [], label='Données brutes', c="b", alpha=0.2)
line_mean, = ax.plot([], [], label='Moyenne par génération')
line_best, = ax.plot([], [], label='Meilleur par génération')
ax.set_title('Mutation génétique')
ax.set_xlabel('Individu')
ax.set_ylabel('Lignes effacées')
ax.legend()

df = pd.read_csv(csv_file)
df["index"] = range(1, len(df) + 1)
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

line_data.set_data(df['index'], df['linesCleared'])
line_mean.set_data(new_df['population'] * len(df) / max_pop, new_df['mean'])
line_best.set_data(new_df['population'] * len(df) / max_pop, new_df['best_score'])

ax.relim()
ax.autoscale_view()

plt.show()
