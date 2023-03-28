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

    
    def get_data_as_panda(self, filepath: str) -> pd.DataFrame:
        
        df_list = CSVHandler._csv_to_pandas(self, filepath)
        df = CSVHandler._merge_dataframes(self, df_list)
        
        df = df.dropna(axis=0, how='any')
        
        print(f'Resulting pd.Dataframe for {filepath}:\n Head:{df.head(3)}\n Tail: {df.tail(3)}\n Shape:{df.shape}\n Index: {df.index}\n Columns: {df.columns}')
        
        return df
    
    def plot_panda(self, df: pd.DataFrame) -> None:

        df.plot(figsize=(10, 10))
        plt.show()
    
        return None
    
      

        
if __name__ == '__main__':
    
    csvHandler = CSVHandler()
    df = csvHandler.get_data_as_panda(csvHandler.filepath_weekly)
    csvHandler.plot_panda(df)