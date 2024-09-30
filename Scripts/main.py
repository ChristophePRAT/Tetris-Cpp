import matplotlib.pyplot as plt
import pandas as pd

file_name = "../build/scores.csv"

df = pd.read_csv(file_name)
# Add index column
df["index"] = range(1, len(df) + 1)
new_df = pd.DataFrame(columns=["population", "best_score", "mean", "median", "worst_score"])

for i in range(df["population"].max() + 1):
    population = df.loc[df["population"] == i]

    # Set i-th index of new_df["best_score"] to the population with the highest score]
    new_df.loc[i, "best_score"] = population["score"].max()
    new_df.loc[i, "mean"] = population["score"].mean()
    new_df.loc[i, "median"] = population["score"].median()
    new_df.loc[i, "population"] = i
    new_df.loc[i, "worst_score"] = population["score"].min()
plt.plot(new_df["population"] * 20, new_df["best_score"], label="Best score")
plt.plot(new_df["population"] * 20, new_df["mean"], label="Mean")
plt.plot(new_df["population"] * 20, new_df["median"], label="Median")
plt.plot(new_df["population"] * 20, new_df["worst_score"], label="Worst score")
plt.plot(df["index"], df["score"], label="All scores", alpha=0.3)
plt.legend()
plt.show()
