import pyupbit
import pandas as pd
from datetime import datetime, timedelta
from config import SYMBOLS, DATA_PATH
from utils import ensure_data_dir, get_daily_filename

def collect_daily_data():
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    
    ensure_data_dir()
    for symbol_name, symbol in SYMBOLS.items():
        try:
            # 전일 시간별 데이터 수집
            df = pyupbit.get_ohlcv(symbol, interval='minute60', count=24)
            
            if df is not None and not df.empty:
                filename = get_daily_filename(symbol, yesterday.strftime('%Y%m%d'))
                df.to_csv(filename)
                print(f"✅ {symbol} 데이터 수집 완료: {yesterday.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"❌ {symbol} 데이터 수집 실패: {str(e)}")

if __name__ == "__main__":
    print("데이터 수집 시작...")
    collect_daily_data()
    print("데이터 수집 완료!")
