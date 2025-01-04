import os
import pandas as pd
from datetime import datetime, timedelta
from config import DATA_PATH

def ensure_data_dir():
    os.makedirs(DATA_PATH, exist_ok=True)

def get_data_filename(symbol):
    return os.path.join(DATA_PATH, f'{symbol}_data.csv')

def get_daily_filename(symbol, date_str):
    return os.path.join(DATA_PATH, f'{symbol}_{date_str}.csv')

def save_data(df, symbol):
    ensure_data_dir()
    filename = get_data_filename(symbol)
    df.to_csv(filename)

def load_data(symbol):
    filename = get_data_filename(symbol)
    if os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=['index'])
    return None

def load_last_6months_data(symbol):
    today = datetime.now()
    dates = [(today - timedelta(days=x)).strftime('%Y%m%d') 
             for x in range(180)]  # 약 6개월
    
    dfs = []
    for date_str in dates:
        filename = get_daily_filename(symbol, date_str)
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df.index)
                df.set_index('timestamp', inplace=True)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    if dfs:
        return pd.concat(dfs, axis=0).sort_index()
    return None
