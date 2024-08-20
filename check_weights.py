from upbit_types import *
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import json

model_idx = 78

# 초기값 설정 부분
model_name = f'model_{model_idx}'

with open(f'models/{model_name}/props.json', 'r') as f:
    props = json.load(f)
    print(props)  # 디버깅용 출력
    coin = props['coin']
    unit = props['unit']
    window_size = props['window_size']
    fluctuation = props['fluctuation']
    input_columns = props['input_columns']

try:
    # 마켓 초기화 및 캔들 데이터 가져오기
    markets = Markets()
    markets.load_markets()
    market = markets.get(coin)

    candles = market.get_recent_candles(unit, count=window_size + 30, allow_upbit_omission=True)

    # 모델 로드
    model = load_model(f'models/{model_name}')
    # model.compile(run_eagerly=True)  # run_eagerly 옵션 추가
    
    if fluctuation:
        fluctuationRates = []
        for i in range(1, len(candles)):
            fluctuationRates.append(candles[i].close_price - candles[i-1].close_price)
        X_sample = np.array([[f] for f in fluctuationRates[:window_size]])
        X_sample = X_sample.reshape((1, X_sample.shape[0], 1))  # 3D 텐서로 변환
    else:
        X_sample = np.array([[getattr(c, col) for col in input_columns] for c in candles[:window_size]])
        X_sample = X_sample.reshape((1, X_sample.shape[0], len(input_columns)))
    
    # 각 시퀀스 인덱스의 중요도 계산
    importances = np.zeros(window_size)
    baseline_pred = model.predict(X_sample)

    for i in range(window_size):
        X_sample_perturbed = X_sample.copy()
        X_sample_perturbed[0, i, 0] = 0  # 특정 인덱스의 값을 0으로 변경
        perturbed_pred = model.predict(X_sample_perturbed)
        importances[i] = np.mean(np.abs(baseline_pred - perturbed_pred))
    
    # 중요도 정규화
    importances = importances / np.max(importances)
    
    print(importances)

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(range(window_size), importances, color='blue')
    plt.xlabel('Sequence Index')
    plt.ylabel('Importance (0-1)')
    plt.title('Feature Importance in LSTM Predictions')
    plt.show()

except Exception as e:
    print(f"Error: {e}")
