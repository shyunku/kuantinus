import os
import joblib
import numpy as np
import pandas as pd
from upbit_types import Markets, Market
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import logger as log
import json
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

coin = 'XRP'
model_idx = 52
unit = 1
window_size = 100
horizon = 20
candle_cnt = 50000
random_state = 30

# model directory
model_name = f'model_{model_idx}'

try:
    # configure markets
    markets = Markets()
    markets.load_markets()

    xrp_market: Market = markets.get(coin)
    log.info("Loaded", xrp_market)
    candles = xrp_market.get_recent_candles(unit, count=candle_cnt, allow_upbit_omission=True)

    # 데이터 추출 및 배열 생성
    data = np.array([c.close_price for c in candles])

    # 차분(differencing)을 통해 시계열 데이터의 안정성 확인
    diff_data = np.diff(data, n=1)
    log.info(f'Original data: {data[:5]}')
    
    # ACF와 PACF 플롯
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(diff_data, lags=50, ax=axes[0])
    plot_pacf(diff_data, lags=50, ax=axes[1])
    plt.show()

    # ARIMA 모델 학습
    log.info("fitting ARIMA model...")
    model = ARIMA(data, order=(window_size, 1, 0))  # (p, d, q) 값 설정
    fit_model = model.fit()
    
    # 학습 과정 시각화
    fit_model.plot_diagnostics(figsize=(16, 8))
    plt.show()

    # 잔차 분석
    residuals = fit_model.resid

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(residuals, bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Residuals Histogram")
    plot_acf(residuals, lags=50, ax=axes[1])
    axes[1].set_title("ACF of Residuals")
    plt.show()

    # 예측
    # forecast, stderr, conf_int = fit_model.forecast(steps=horizon, alpha=0.05)
    forecast = fit_model.forecast(steps=horizon)
    
    # 평가
    test_data = data[-horizon:]  # 마지막 horizon 길이의 데이터를 테스트 데이터로 사용
    mse = mean_squared_error(test_data, forecast)
    r2 = r2_score(test_data, forecast)
    
    log.info(f'Mean Squared Error: {mse}')
    log.info(f'R-squared: {r2}')
    
    # make directory
    if not os.path.exists(f'models'):
        os.makedirs(f'models')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    # save model
    log.info(f'saving model...')
    fit_model.save(f'models/{model_name}/arima_model.pkl')
    
    # save properties
    props = {
        'coin': coin,
        'unit': unit,
        'window_size': window_size,
        'horizon': horizon,
        'candle_cnt': candle_cnt,
        'random_state': random_state,
        'mse': mse,
        'r2': r2,
        'model_order': (5, 1, 0)
    }
    # save properties to file
    with open(f'models/{model_name}/props.json', 'w') as f:
        json.dump(props, f, indent=4)

except Exception as e:
    log.error('Failed to train model:', e)