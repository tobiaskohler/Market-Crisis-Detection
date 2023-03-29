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
        
        # DATE-RELATED FEATURES
        # calculate day of week
        df['day_of_week'] = df.index.dayofweek
        
        # calculate month
        df['month'] = df.index.month
        
        # calculate is month end
        df['is_month_end'] = df.index.is_month_end
        
        # calculate is month start
        df['is_month_start'] = df.index.is_month_start
        
        # calculate is quarter end
        df['is_quarter_end'] = df.index.is_quarter_end

        # calculate is quarter start
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
        
        

        #print if na
        print(df.isna().sum())
        df = df.dropna()
        
        # save to csv
        df.to_csv(os.path.join(self.parent_dir, 'prepared_data', 'resampled.csv'))

        return df
    
    

    def csv_to_np(self, filepath: str) -> np.array:
        
        df = pd.read_csv(filepath)
        features = df.drop(columns=['market_light', 'Date'])
        labels = df['market_light_1']
        
        print(f'Shape of features: {features.shape}')
        print(f'Shape of labels: {labels.shape}')
        
        return features, labels
    


if __name__ == '__main__':
    
    csvHandler = CSVHandler()

    df = csvHandler.get_resampled_df()

    f = '../prepared_data/resampled.csv'
    features, labels = csvHandler.csv_to_np(f)