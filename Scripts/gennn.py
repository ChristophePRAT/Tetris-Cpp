import matplotlib.pyplot as plt
import pandas as pd

file_name = "../build/scores_gennn_preview-next-block.csv"

df = pd.read_csv(file_name)
# Add index column
df["index"] = range(1, len(df) + 1)
new_df = pd.DataFrame(columns=["population", "best_score", "mean", "median", "worst_score"])

metric = "linesCleared"

for i in range(df["population"].max() + 1):
    population = df.loc[df["population"] == i]
    # Set i-th index of new_df["best_score"] to the population with the highest score]
    new_df.loc[i, "best_score"] = population[metric].max()
    new_df.loc[i, "mean"] = population[metric].mean()
    # new_df.loc[i, "median"] = population[metric].median()
    # new_df.loc[i, "worst_score"] = population[metric].min()
    best, median, worst = population[metric].quantile([0.75, 0.5, 0.25])

    new_df.loc[i, "best_quartile"] = best
    new_df.loc[i, "median"] = median
    new_df.loc[i, "worst_quartile"] = worst

    new_df.loc[i, "population"] = i

new_df["population"] = pd.to_numeric(new_df["population"], errors='coerce')
new_df["best_quartile"] = pd.to_numeric(new_df["best_quartile"], errors='coerce')
new_df["worst_quartile"] = pd.to_numeric(new_df["worst_quartile"], errors='coerce')

plt.plot(new_df["population"], new_df["best_score"], label="Best" + metric)
plt.fill_between(new_df["population"], new_df["best_quartile"], new_df["worst_quartile"], alpha=0.3)

plt.plot(new_df["population"], new_df["mean"], label="Mean")
plt.plot(new_df["population"], new_df["median"], label="Median")
# plt.plot(df["index"], df[metric], label="All " + metric, alpha=0.3)
plt.legend()
plt.show()
