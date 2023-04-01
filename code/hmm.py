from data_handler import CSVHandler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from hmmlearn.hmm import GaussianHMM
from scipy.signal import medfilt


def calc_hmm():


    #make background grey
    sns.set_style('darkgrid')

    features, labels = CSVHandler.csv_to_np(CSVHandler, filepath='../prepared_data/resampled.csv')
    X = features
    y = features['^GSPC']


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    n_regimes = 3 # normal, stress and crisis

    model = GaussianHMM(n_components=n_regimes, covariance_type='full', random_state=42)
    model.fit(X_scaled)

    regimes = model.predict(X_scaled)
    X['regimes'] = regimes

    # filter out noise
    X['regimes'] = medfilt(X['regimes'], kernel_size=131)
    X['regimes'] = X['regimes'].shift(-2)

    # calculate mean and std-dev for each regime annuazlized
    means = X.groupby('regimes')['log_ret'].mean() * 252
    means.sort_values(inplace=True)
    stds = X.groupby('regimes')['log_ret'].std() * 252
    #sort stds in descending order
    stds.sort_values(inplace=True, ascending=False)
    srs = means / stds
    srs.sort_values(inplace=True)

    print(type(means))
    print(f'Annualized mean returns: {means}\nAnnualized std-dev: {stds}\nSharpe ratio: {srs}')

    X['market_light_hmm'] = 0
    for i, std in enumerate(stds):
        if i == 0:
            print(f'Std: {std}. Regime: {stds.index[0]}. Assigning red.')
        elif i == 1:
            print(f'Std: {std}. Regime: {stds.index[1]}. Assigning yellow.')
        elif i == 2:
            print(f'Std: {std}. Regime: {stds.index[2]}. Assigning green.')

    X['market_light_hmm'] = X['regimes'].apply(lambda x: -1 if x == stds.index[0] else 0 if x == stds.index[1] else 1)


    print(X)
    #transform X to dataframe
    X = pd.DataFrame(X)
    X.to_csv('../prepared_data/resampled_hmm.csv')
        
    print(X['market_light_hmm'].shape)
    print(f'Type of hmm: {type(X)}')


    #fig, ax = plt.subplots(figsize=(16, 9))

    #scatterplot, compare with regime_mapping and plot in the following colors:

    # ax.scatter(X.index, X['^GSPC'], c=X['regimes'].apply(lambda x: 'red' if x == stds.index[0] else 'yellow' if x == stds.index[1] else 'green'), s=1)

    # plt.show()
    
    
    return X['market_light_hmm']



if __name__ == '__main__':
    calc_hmm()