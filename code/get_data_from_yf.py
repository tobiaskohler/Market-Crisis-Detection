import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# add moving_average parameter, default to False

def get_data(start: str, end: str, ticker: str, indicators: bool=False, plot: bool=False) -> pd.DataFrame:


    data = yf.download(ticker, start=start, end=end, interval='1d')
    data = data[['Adj Close']]
    #calculate daily log returns
    data.columns = [ticker]
    data['log_ret'] = np.log(data[ticker] / data[ticker].shift(1))
    
    # remove timezone info from index
    data.index = data.index.tz_localize(None)
    
    if indicators:
        
        # EXPONENTIAL WEIGHTED MOVING AVERAGES
        data['ewma200'] = data[ticker].ewm(span=200, adjust=False).mean()
        data['ewma50'] = data[ticker].ewm(span=50, adjust=False).mean()
        data['ewma20'] = data[ticker].ewm(span=20, adjust=False).mean()
        data['ewma10'] = data[ticker].ewm(span=10, adjust=False).mean()
        
        # MOMENTUM, DEFINED AS CLOSE[t]/CLOSE[t-n]
        data['mom200'] = data[ticker] / data[ticker].shift(200)
        data['mom50'] = data[ticker] / data[ticker].shift(50)
        data['mom20'] = data[ticker] / data[ticker].shift(20)
        data['mom10'] = data[ticker] / data[ticker].shift(10)
        
        # MOMENTUM REVERSAL, DEFINED AS CLOSE[t-n]/CLOSE[t]
        data['momrev200'] = data[ticker].shift(200) / data[ticker]
        data['momrev50'] = data[ticker].shift(50) / data[ticker]
        data['momrev20'] = data[ticker].shift(20) / data[ticker]
        data['momrev10'] = data[ticker].shift(10) / data[ticker]
        
        # VOLATILITY (STANDARD DEVIATION)
        data['vola200'] = data['log_ret'].rolling(200).std() * np.sqrt(200)
        data['vola50'] = data['log_ret'].rolling(50).std() * np.sqrt(50)
        data['vola20'] = data['log_ret'].rolling(20).std() * np.sqrt(20)
        data['vola10'] = data['log_ret'].rolling(10).std() * np.sqrt(10)
        
        data = data.dropna()
    
    #save data to csv using ; as separator
    data.to_csv(f'{ticker}_{start}_{end}.csv', sep=';')
    
    if plot:
        
        
        # plot all data, except for columns beginning with "mom" on first plot, and columns beginning with "mom" on second plot
        plt.style.use('dark_background')        

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
        
        ax1.plot(data[ticker], label=ticker)
        ax1.plot(data['ewma200'], label='ewma200')
        ax1.plot(data['ewma50'], label='ewma50')
        ax1.plot(data['ewma20'], label='ewma20')
        ax1.plot(data['ewma10'], label='ewma10')
        ax1.legend(loc='upper left')
        ax1.set_title(f'{ticker} EWMAs ({start} to {end})')

        ax2.plot(data['mom200'], label='mom200')
        ax2.plot(data['mom50'], label='mom50')
        ax2.plot(data['mom20'], label='mom20')
        ax2.plot(data['mom10'], label='mom10')
        ax2.plot(data['momrev200'], label='momrev200')
        ax2.plot(data['momrev50'], label='momrev50')
        ax2.plot(data['momrev20'], label='momrev20')
        ax2.plot(data['momrev10'], label='momrev10')
        ax2.legend(loc='upper left')
        ax2.set_title(f'{ticker} Momentum Indicators ({start} to {end})')
        
        # add vola
        ax3.plot(data['vola200'], label='vola200')
        ax3.plot(data['vola50'], label='vola50')
        ax3.plot(data['vola20'], label='vola20')
        ax3.plot(data['vola10'], label='vola10')
        ax3.legend(loc='upper left')
        ax3.set_title(f'{ticker} Volatility Indicators ({start} to {end})')
        
        plt.show()

    
    print(data.loc['2005-01-01':'2005-01-31'])
    print(data.loc['2023-01-01':'2023-01-31'])

if __name__ == '__main__':
    
    
    start = '2002-01-01'
    end = '2023-03-26'
    ticker = '^GSPC'
    
    get_data(start, end, ticker, indicators=True, plot=True)