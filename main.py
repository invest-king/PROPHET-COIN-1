import pyupbit
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
from config import SYMBOLS, FORECAST_HOURS
from utils import load_last_6months_data

def analyze_crypto(symbol_name, symbol):
    try:
        # 6개월 데이터 로드
        df = load_last_6months_data(symbol)
        if df is None or df.empty:
            raise ValueError(f"데이터가 없습니다: {symbol}")
        
        # Prophet 데이터 준비
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['close'].astype(float)  # 명시적 형변환 추가
        }).reset_index(drop=True)
        
        # 결측치 제거
        prophet_df = prophet_df.dropna()
        
        # 모델 학습
        model = Prophet(
            changepoint_prior_scale=0.05,
            yearly_seasonality=False,  # 6개월 데이터로는 연간 계절성 분석이 어려움
            weekly_seasonality=True,
            daily_seasonality=True,
            interval_width=0.95
        )
        
        model.fit(prophet_df)
        
        # 예측
        future = model.make_future_dataframe(periods=FORECAST_HOURS, freq='H')
        forecast = model.predict(future)
        
        return prophet_df, forecast
        
    except Exception as e:
        print(f"❌ {symbol} 분석 실패: {str(e)}")
        return None, None

def main():
    print("암호화폐 분석 시작...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    for i, (symbol_name, symbol) in enumerate(SYMBOLS.items()):
        prophet_df, forecast = analyze_crypto(symbol_name, symbol)
        
        if prophet_df is not None and forecast is not None:
            # 그래프 그리기
            ax = axes[i]
            ax.plot(prophet_df['ds'], prophet_df['y'], 'k.', label='실제 가격')
            ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='예측')
            ax.fill_between(forecast['ds'], 
                          forecast['yhat_lower'], 
                          forecast['yhat_upper'],
                          color='b', alpha=0.2, label='신뢰 구간')
            ax.set_title(f"{symbol} 가격 예측")
            ax.legend()
            
            # 현재 가격과 예측 가격 비교
            current_price = prophet_df['y'].iloc[-1]
            predicted_price = forecast['yhat'].iloc[-1]
            print(f"\n{symbol} 분석 결과:")
            print(f"현재 가격: {current_price:,.0f}")
            print(f"예측 가격: {predicted_price:,.0f}")
            print(f"신뢰 구간: {forecast['yhat_lower'].iloc[-1]:,.0f} ~ {forecast['yhat_upper'].iloc[-1]:,.0f}")
    
    plt.tight_layout()
    plt.show()
    print("\n분석 완료!")

if __name__ == "__main__":
    main()
