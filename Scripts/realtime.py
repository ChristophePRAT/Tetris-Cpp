import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import argparse

# Parse the arguments which are the file path
parser = argparse.ArgumentParser(description='Real-time plot of the scores')
parser.add_argument('file', type=str, help='Path to the CSV file')
args = parser.parse_args()


# Path to your CSV file
# csv_file = "../build/scores_gennn_seeded.csv"
csv_file = args.file

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(
        id='interval-component',
        interval=10000,  # Refresh every 10 second
        n_intervals=0
    )
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
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
        max_pop = new_df["population"].max()

    # Create the graph
        figure = {
            'data': [
                {'x': df['index'], 'y': df['linesCleared'], 'type': 'line', 'name': 'Data'},
                {'x': new_df['population'] * len(df)/max_pop, 'y': new_df['mean'], 'type': 'line', 'name': 'Mean'},
                {'x': new_df['population'] * len(df)/max_pop, 'y': new_df['best_score'],'type': 'line', 'name': 'Best' }
            ],
            'layout': {
                'title': 'Real-Time Data Plot'
            }
        }
        return figure

    except Exception as e:
        print(f"Error reading the file {e}")
        return {'data': [], 'layout': {'title': f"Error: {e}"}}

if __name__ == '__main__':
    app.run(debug=True)
