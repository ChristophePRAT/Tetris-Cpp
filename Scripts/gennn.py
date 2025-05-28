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



fig, ax = plt.subplots(figsize=(12, 6))

# Plot the line chart with individual data points
line_data, = ax.plot(df['index'], df['linesCleared'], label='Données brutes', c="black", alpha=0.5, linewidth=0.5)

# Calculate bar positions to align with line chart
# Each bar should cover exactly 64 individuals
bar_positions = []
for pop in new_df["population"]:
    # Center of each generation (64 individuals per generation)
    start_idx = pop * 64 + 1  # +1 because index starts at 1
    end_idx = (pop + 1) * 64
    center = (start_idx + end_idx) / 2
    bar_positions.append(center)

# Create bar charts with proper positioning
bar_width = 64  # Full width to remove padding between bars
bar_mean = ax.bar(bar_positions, new_df["mean"], width=bar_width, label='Moyenne par génération', color='blue', alpha=0.2, align='center')
bar_best = ax.bar(bar_positions, new_df["best_score"], width=bar_width, label='Meilleur par génération', color='g', alpha=0.1, align='center')

# ax.set_title('Mutation génétique', fontsize=20)
ax.set_xlabel('Individu', fontsize=20)
ax.set_ylabel('Lignes effacées', fontsize=20)
ax.legend(fontsize=20)

# Set x-axis limits to show all data
# ax.set_xlim(0, len(df) + 1)

plt.tight_layout()
plt.show()
