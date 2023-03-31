import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CSVHandler():
    
    def __init__(self) -> None:
        
        self.parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.filepath_daily = os.path.join(self.parent_dir, 'prepared_data', 'daily')
        self.filepath_weekly = os.path.join(self.parent_dir, "prepared_data", "weekly")
        self.filepath_monthly = os.path.join(self.parent_dir, "prepared_data", "monthly")
        self.filepath_quarterly = os.path.join(self.parent_dir, "prepared_data", "quarterly")
        
        print(f'Initializing CSVHandler with the following paths:\n {self.filepath_daily}\n {self.filepath_weekly}\n {self.filepath_monthly}\n {self.filepath_quarterly}')
        
    
    def _get_files(self, filepath: str) -> list:
        
        #only select files with .csv extension
        
        filenames = [f for f in os.listdir(filepath) if f.endswith('.csv')]
        no_files = len(filenames)
        
        print(f'Filenames in {filepath}:\n {filenames} ({no_files} files))')
        
        return filenames
    
    
    def _csv_to_pandas(self, filepath: str) -> pd.DataFrame:
        
        files = CSVHandler._get_files(self, filepath)
        df_list = []
        
        if not os.path.exists(f'{filepath}'):
            raise FileNotFoundError(f'Filepath {filepath} does not exist.')
        
        for file in files:
            
            filename = os.path.join(filepath, file)
            print(f'Converting {filename} to pandas dataframe.')
            
            df = pd.read_csv(filename, delimiter=';')
            
            if filename==self.filepath_monthly:
                # adjust index to only contain year and month
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)
                df.index = df.index.to_period('M')
            
            else:
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)
            
            print(df)
            
            #make all columns numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            
            print(df.head(2))
            print(df.columns)
            print(df.index)
            print(df.shape)
            df_list.append(df)

        return df_list
    
    
    def _merge_dataframes(self, df_list: list) -> pd.DataFrame:
        
        df = pd.concat(df_list, axis=1)

        return df

    
    
    def _get_data_as_panda(self, filepath: str) -> pd.DataFrame:
        
        '''
        One-in-all function that returns a pandas df with all data from the given filepath.
        '''
        
        df_list = CSVHandler._csv_to_pandas(self, filepath)
        df = CSVHandler._merge_dataframes(self, df_list)
        
        print(f'Resulting pd.Dataframe for {filepath}:\n Head:{df.head(3)}\n Tail: {df.tail(3)}\n Shape:{df.shape}\n Index: {df.index}\n Columns: {df.columns}')
        
        print(df)
        return df
    
    
    def get_resampled_df(self) -> pd.DataFrame:
        
        daily_df = CSVHandler._get_data_as_panda(self, self.filepath_daily)
        weekly_df = CSVHandler._get_data_as_panda(self, self.filepath_weekly)
        monthly_df = CSVHandler._get_data_as_panda(self, self.filepath_monthly)
        quarterly_df = CSVHandler._get_data_as_panda(self, self.filepath_quarterly)
        
        df = pd.merge(daily_df, weekly_df , how='outer', left_index=True, right_index=True)
        df = pd.merge(df, monthly_df , how='outer', left_index=True, right_index=True)
        df = pd.merge(df, quarterly_df , how='outer', left_index=True, right_index=True)

        df = df.resample('D').asfreq()

        df = df.fillna(method='ffill')
        df = df.dropna()

        print("Properties of resampled dataframe:\n")
        print(df.head(3))
        print(df.tail(3))
        print(df.shape)
        print(df.index)
        print(df.columns)
                
        # AUGMENT DATA WITH NEW FEATURES
        lags = 14
        
        # CALCULATE DIFFERENCES
        
        df['diff_T10Y2Y'] = df['T10Y2Y'].pct_change()
        df['diff_T10Y3M'] = df['T10Y3M'].pct_change()
        df['diff_OFR FSI'] = df['OFR FSI'].pct_change()
        df['diff_GDP'] = df['GDP'].pct_change()
        df['diff_EUGDP'] = df['EUGDP'].pct_change()
        df['diff_Bullish'] = df['Bullish'].pct_change()
        df['diff_Bearish'] = df['Bearish'].pct_change()
        df['diff_Neutral'] = df['Neutral'].pct_change()
        df['diff_UMCSENT'] = df['UMCSENT'].pct_change()
        df['diff_BCI'] = df['BCI'].pct_change()
        df['diff_UNRATE'] = df['UNRATE'].pct_change()
        
        # DATE-RELATED FEATURES
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_month_end'] = df.index.is_month_end
        df['is_month_start'] = df.index.is_month_start
        df['is_quarter_end'] = df.index.is_quarter_end
        df['is_quarter_start'] = df.index.is_quarter_start
        
        
        # LAGGED VARIABLES
        for i in range(1,lags+1):
            df[f'T10Y2Y_{i}'] = df['T10Y2Y'].shift(i)

        for i in range(1,lags+1):
            df[f'T10Y3M_{i}'] = df['T10Y2Y'].shift(i)

        for i in range(1,lags+1):
            df[f'market_light_{i}'] = df['market_light'].shift(i)
            
        for i in range(1,lags+1):
            df[f'OFR FSI_{i}'] = df['OFR FSI'].shift(i)
            
        for i in range(1,lags+1):
            df[f'Bullish_{i}'] = df['Bullish'].shift(i)
            df[f'Neutral_{i}'] = df['Neutral'].shift(i)
            df[f'Bearish_{i}'] = df['Bearish'].shift(i)
            
        # ROLLING WINDOWS, WEIGHTED AVERAGE, MAX, MIN and STD-DEV
        
        window_size = 30
        
        # T10Y2Y
        
        #calculate weighted moving average
        df['T10Y2Y_rolling_mean'] = df['T10Y2Y'].rolling(window=window_size).mean()
        df['T10Y2Y_rolling_max'] = df['T10Y2Y'].rolling(window=window_size).max()
        df['T10Y2Y_rolling_min'] = df['T10Y2Y'].rolling(window=window_size).min()
        df['T10Y2Y_rolling_std'] = df['T10Y2Y'].rolling(window=window_size).std()
        
        # T10Y3M
        df['T10Y3M_rolling_mean'] = df['T10Y3M'].rolling(window=window_size).mean()
        df['T10Y3M_rolling_max'] = df['T10Y3M'].rolling(window=window_size).max()
        df['T10Y3M_rolling_min'] = df['T10Y3M'].rolling(window=window_size).min()
        df['T10Y3M_rolling_std'] = df['T10Y3M'].rolling(window=window_size).std()
        
        # OFR FSI
        df['OFR FSI_rolling_mean'] = df['OFR FSI'].rolling(window=window_size).mean()
        df['OFR FSI_rolling_max'] = df['OFR FSI'].rolling(window=window_size).max()
        df['OFR FSI_rolling_min'] = df['OFR FSI'].rolling(window=window_size).min()
        df['OFR FSI_rolling_std'] = df['OFR FSI'].rolling(window=window_size).std()
        
        # Bullish   
        
        df['Bullish_rolling_mean'] = df['Bullish'].rolling(window=window_size).mean()
        df['Bullish_rolling_max'] = df['Bullish'].rolling(window=window_size).max()
        df['Bullish_rolling_min'] = df['Bullish'].rolling(window=window_size).min()
        df['Bullish_rolling_std'] = df['Bullish'].rolling(window=window_size).std()
        
        # Neutral
        
        df['Neutral_rolling_mean'] = df['Neutral'].rolling(window=window_size).mean()
        df['Neutral_rolling_max'] = df['Neutral'].rolling(window=window_size).max()
        df['Neutral_rolling_min'] = df['Neutral'].rolling(window=window_size).min()
        df['Neutral_rolling_std'] = df['Neutral'].rolling(window=window_size).std()
        
        # Bearish
        
        df['Bearish_rolling_mean'] = df['Bearish'].rolling(window=window_size).mean()
        df['Bearish_rolling_max'] = df['Bearish'].rolling(window=window_size).max()
        df['Bearish_rolling_min'] = df['Bearish'].rolling(window=window_size).min()
        df['Bearish_rolling_std'] = df['Bearish'].rolling(window=window_size).std()
        
            
        #print if na
        print(df.isna().sum())
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # save to csv
        df.to_csv(os.path.join(self.parent_dir, 'prepared_data', 'resampled.csv'))
        print(f'Saved to {os.path.join(self.parent_dir, "prepared_data", "resampled.csv")}')
        return df
    
    

    def csv_to_np(self, filepath: str) -> np.array:
        
        df = pd.read_csv(filepath)
        
        features = df.drop(columns=['Date'])
        features = features[:-2] # remove last two rows, since no prediction is available
        
        labels = df['market_light'].shift(-2) # Market Light of the day after tomorrow
        labels = labels[:-2] # remove last two rows, since no prediction is available
        
        print(f'Shape of features: {features.shape}')
        print(f'Shape of labels: {labels.shape}')
        
        # # save features and labels to csv files
        # features.to_csv(os.path.join(self.parent_dir, 'prepared_data', 'features.csv'))
        # labels.to_csv(os.path.join(self.parent_dir, 'prepared_data', 'labels.csv'))
        
        return features, labels
    


if __name__ == '__main__':
    
    csvHandler = CSVHandler()

    df = csvHandler.get_resampled_df()

    f = '../prepared_data/resampled.csv'
    features, labels = csvHandler.csv_to_np(f)