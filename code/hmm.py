import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import math
import warnings

from data_handler import CSVHandler
from utils import *

import plotly.graph_objects as go
from plotly.graph_objs.scatter.marker import Line
from plotly.subplots import make_subplots

class RegimeDetection:

 
    def get_regimes_hmm(self, input_data, params):

        hmm_model = self.initialise_model(GaussianHMM(), params).fit(input_data)

        return hmm_model


    def initialise_model(self, model, params):

        for parameter, value in params.items():

            setattr(model, parameter, value)

        return model


def plot_hidden_states(hidden_states, prices_df):

    '''
    Input:
    hidden_states(numpy.ndarray) - array of predicted hidden states
    prices_df(df) - dataframe of close prices

    Output:
    Graph showing hidden states and prices

    '''

    colors = ['blue', 'green']
    n_components = len(np.unique(hidden_states))
    fig = go.Figure()

    for i in range(n_components):
        mask = hidden_states == i
        print('Number of observations for State ', i,":", len(prices_df.index[mask]))
        
        fig.add_trace(go.Scatter(x=prices_df.index[mask], y=prices_df[f"{prices_df.columns.name}"][mask],
                    mode='markers',  name='Hidden State ' + str(i), marker=dict(size=4,color=colors[i])))
        
    fig.update_layout(height=400, width=900, legend=dict(
            yanchor="top", y=0.99, xanchor="left",x=0.01), margin=dict(l=20, r=20, t=20, b=20)).show()


if __name__ == '__main__':
    regime_detector = RegimeDetection()
    
    features, labels = CSVHandler.csv_to_np(CSVHandler, filepath='../prepared_data/resampled.csv')
    log_returns = features.iloc[:, 3].values
    prices_array = features.iloc[:, 2].values

    print(prices_array.shape)
    params = {'n_components':2, 'covariance_type':"full", 'random_state':100}

    hmm_model = regime_detector.get_regimes_hmm(features, params)

    hmm_states = hmm_model.predict(features)
    print(hmm_states)
    
    plot_hidden_states(np.array(hmm_states), features[['^GSPC']])


            