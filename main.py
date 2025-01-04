import pyupbit
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to fetch historical data
def fetch_ohlcv(ticker, interval='minute60', count=4380):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    # 종가와 거래량만 선택
    return df[['close', 'volume']]

# Function to prepare data for Prophet
def prepare_prophet_data(df):
    prophet_df = df[['close']].copy()
    prophet_df.index.name = 'ds'
    prophet_df.reset_index(inplace=True)
    prophet_df.columns = ['ds', 'y']
    return prophet_df

# Function to create and train Prophet model
def train_prophet_model(df):
    model = Prophet(
        changepoint_prior_scale=0.05,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.95
    )
    model.fit(df)
    return model

# Function to generate trading signals
def generate_signals(forecast):
    signals = pd.DataFrame()
    signals['ds'] = forecast['ds']
    signals['yhat'] = forecast['yhat']
    signals['yhat_lower'] = forecast['yhat_lower']
    signals['yhat_upper'] = forecast['yhat_upper']
    
    # Generate buy/sell signals
    signals['signal'] = 0
    signals.loc[signals['y'] <= signals['yhat_lower'], 'signal'] = 1  # Buy signal
    signals.loc[signals['y'] >= signals['yhat_upper'], 'signal'] = -1  # Sell signal
    
    return signals

def main():
    # Fetch data for BTC and ETH from Upbit
    btc_df = fetch_ohlcv("KRW-BTC")
    eth_df = fetch_ohlcv("KRW-ETH")
    
    # Prepare data for Prophet
    btc_prophet_df = prepare_prophet_data(btc_df)
    eth_prophet_df = prepare_prophet_data(eth_df)
    
    # Train models
    btc_model = train_prophet_model(btc_prophet_df)
    eth_model = train_prophet_model(eth_prophet_df)
    
    # Make predictions
    future_btc = btc_model.make_future_dataframe(periods=24, freq='H')
    forecast_btc = btc_model.predict(future_btc)
    
    future_eth = eth_model.make_future_dataframe(periods=24, freq='H')
    forecast_eth = eth_model.predict(future_eth)
    
    # Generate signals
    btc_signals = generate_signals(forecast_btc)
    eth_signals = generate_signals(forecast_eth)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot BTC
    ax1.plot(btc_prophet_df['ds'], btc_prophet_df['y'], 'k.', label='Actual')
    ax1.plot(forecast_btc['ds'], forecast_btc['yhat'], 'b-', label='Prediction')
    ax1.fill_between(forecast_btc['ds'], forecast_btc['yhat_lower'], forecast_btc['yhat_upper'], 
                     color='b', alpha=0.2, label='Confidence Interval')
    ax1.set_title('BTC/USDT Prediction')
    ax1.legend()
    
    # Plot ETH
    ax2.plot(eth_prophet_df['ds'], eth_prophet_df['y'], 'k.', label='Actual')
    ax2.plot(forecast_eth['ds'], forecast_eth['yhat'], 'r-', label='Prediction')
    ax2.fill_between(forecast_eth['ds'], forecast_eth['yhat_lower'], forecast_eth['yhat_upper'], 
                     color='r', alpha=0.2, label='Confidence Interval')
    ax2.set_title('ETH/USDT Prediction')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
