import pyupbit
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from config import SYMBOLS, FORECAST_HOURS

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def get_crypto_data(symbol: str) -> pd.DataFrame:
    try:
        df = pyupbit.get_ohlcv(symbol, interval="day", count=365)
        if df is None or df.empty:
            logger.error(f"데이터가 없습니다: {symbol}")
            return None
        
        logger.info(f"{symbol} 데이터 {len(df)} 개 로드 완료")
        return df
    except Exception as e:
        logger.error(f"{symbol} 데이터 로드 오류: {str(e)}")
        return None

def analyze_crypto(symbol_name, symbol):
    try:
        logger.info(f"{symbol} 데이터 로딩 중...")
        
        # Verify symbol format
        if not symbol.startswith("KRW-"):
            symbol = f"KRW-{symbol}"
            logger.info(f"심볼 형식 수정: {symbol}")
        
        # Get the data
        df = get_crypto_data(symbol)
        if df is None:
            raise ValueError(f"데이터를 가져올 수 없습니다: {symbol}")
        
        # Detailed error checking
        if df.empty:
            logger.error(f"데이터가 비어있습니다: {symbol}")
            raise ValueError(f"데이터가 비어있습니다: {symbol}")
        
        logger.info(f"{symbol} 데이터 {len(df)}개 레코드 로드 완료")
        
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
            yearly_seasonality=True,  # 1년 데이터를 사용하므로 연간 계절성 활성화
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
        logger.error(f"오류 발생: {symbol} - {str(e)}")
        logger.error("상세 오류 정보:", exc_info=True)
        return None, None

def main():
    logger.info("암호화폐 가격 예측 분석을 시작합니다...")
    
    try:
        fig, axes = plt.subplots(len(SYMBOLS), 1, figsize=(15, 6*len(SYMBOLS)))
        if len(SYMBOLS) == 1:
            axes = [axes]
        
        for i, (symbol_name, symbol) in enumerate(SYMBOLS.items()):
            prophet_df, forecast = analyze_crypto(symbol_name, symbol)
            
            if prophet_df is not None and forecast is not None:
                ax = axes[i]
                ax.plot(prophet_df['ds'], prophet_df['y'], 'k.', label='실제 가격')
                ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='예측')
                ax.fill_between(forecast['ds'], 
                              forecast['yhat_lower'], 
                              forecast['yhat_upper'],
                              color='b', alpha=0.2, label='신뢰 구간')
                ax.set_title(f"{symbol} 가격 예측")
                ax.legend()
                
                current_price = prophet_df['y'].iloc[-1]
                predicted_price = forecast['yhat'].iloc[-1]
                print(f"\n{symbol} 분석 결과:")
                print(f"현재 가격: {current_price:,.0f}")
                print(f"예측 가격: {predicted_price:,.0f}")
                print(f"신뢰 구간: {forecast['yhat_lower'].iloc[-1]:,.0f} ~ {forecast['yhat_upper'].iloc[-1]:,.0f}")
        
        plt.tight_layout()
        plt.show()
        print("\n분석이 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"메인 실행 중 오류 발생: {str(e)}")
        logger.error("상세 오류 정보:", exc_info=True)

if __name__ == "__main__":
    main()
