import os

class CSVHandler():
    
    def __init__(self) -> None:
        
        self.filepath_daily = os.path.join(os.path.dirname(__file__), "..", "prepared_data", "daily")
        self.filepath_weekly = os.path.join(os.path.dirname(__file__), "..", "prepared_data", "weekly")
        self.filepath_monthly = os.path.join(os.path.dirname(__file__), "..", "prepared_data", "monthly")
        self.filepath_quarterly = os.path.join(os.path.dirname(__file__), "..", "prepared_data", "quarterly")
        
        print(f'Initializing CSVHandler with the following paths:\n {self.filepath_daily}\n {self.filepath_weekly}\n {self.filepath_monthly}\n {self.filepath_quarterly}')
        
    def get_files(self, filepath: str) -> list:
        
        #only select files with .csv extension
        
        filenames = [f for f in os.listdir(filepath) if f.endswith('.csv')]
        no_files = len(filenames)
        
        print(f'Filenames in {filepath}:\n {filenames} ({no_files} files))')
        
        return filenames
    
    
        
        
if __name__ == '__main__':
    
    csvHandler = CSVHandler()
    csvHandler.get_filenames(csvHandler.filepath_daily)
    