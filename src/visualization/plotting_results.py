# Inspired by the following notebook: https://www.kaggle.com/mykolazotko/3d-visualization-of-molecules-with-plotly
# Imports
import pandas as pd
import numpy as np
import plotly.graph_objs as gobj


def create_summary_df(results_mean):
    """
    INPUT:
        results_mean: numpy array with scores in order given by index
                      and mean of these scores.
    OUTPUT:
        df: pandas dataframe with columns: 0 up to #columns-1 in input and 'ticktext'
    """
    index = ['3JHH', '3JHC', '2JHC', '2JHH', '1JHC', 'mean']

    df = pd.DataFrame(results_mean)
    df.loc[:,'ticktext'] = index
    return df


def plot_results(df):
    """
    INPUT:
        df: pandas dataframe. Created with 'create_summary_df' function
            (need columns: 'baseline', 'top model', 'ticktext')
    OUTPUT:
        fig: plotly object
    """
    keyword_dict = {0: 'without topological features',
                    1: 'with topological features'}
    fig = gobj.Figure()

    for c in (set(df.columns)-set(['ticktext'])):
        fig.add_trace(
            gobj.Scatter(
                mode='markers',
                name=keyword_dict[c],
                x=df.index,
                y=df[c],
                marker=dict(
                    size=10,
                ),
                showlegend=True
            )
        )

    fig.update_layout(
        yaxis_title="score",
        xaxis = dict(
            tickmode = 'array',
            tickvals = df.index,
            ticktext = df['ticktext'],
        ),
        yaxis = dict(autorange = 'reversed')

    )

    return fig
