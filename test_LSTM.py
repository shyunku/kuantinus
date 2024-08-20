from upbit_types import *
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import json

model_idx = 78
showing_candles = 200

# 초기값 설정 부분
model_name = f'model_{model_idx}'
x_scaler_method = None
y_scaler_method = None

with open(f'models/{model_name}/props.json', 'r') as f:
    props = json.load(f)
    print(props)  # 디버깅용 출력
    coin = props['coin']
    unit = props['unit']
    window_size = props['window_size']
    horizon = props['horizon']
    fluctuation = props['fluctuation']
    input_columns = props['input_columns']
  
    x_scaler_method = props.get('x_scaler_method', None)
    y_scaler_method = props.get('y_scaler_method', None)
  
    scaler_X = load(f'models/{model_name}/scaler_x.gz') if x_scaler_method == 'minmax' else None
    scaler_y = load(f'models/{model_name}/scaler_y.gz') if y_scaler_method == 'minmax' else None

try:
    # 마켓 초기화 및 캔들 데이터 가져오기
    markets = Markets()
    markets.load_markets()
    market: Market = markets.get(coin)

    candles = market.get_recent_candles(unit, count=window_size + showing_candles + 1, allow_upbit_omission=True)
    last_candle = candles[-1 - showing_candles]

    # 모델 로드
    model = load_model(f'models/{model_name}')
    # model.compile(run_eagerly=True)  # run_eagerly 옵션 추가
    
    if fluctuation:
        # extract difference
        fluctuationRates = []
        for i in range(1, len(candles)):
            fluctuationRates.append((candles[i].close_price - candles[i-1].close_price) / candles[i-1].close_price)
        X = np.array([[f] for f in fluctuationRates])
        y = np.array(fluctuationRates)
    else:
        X = np.array([[getattr(c, col) for col in input_columns] for c in candles])
        y = np.array([c.close_price for c in candles])
    
    num_iterations = showing_candles // horizon

    print(num_iterations)
    def create_sequences(X, y, window_size, horizon):
        X_seq, y_seq = [], []
        for i in range(showing_candles):
          idx = i
          X_seq.append(X[idx:idx + window_size])
          y_seq.append(y[idx + window_size:idx + window_size + horizon])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X, y, window_size, horizon)
    print(f'x_shape={X_seq.shape}, y_shape={y_seq.shape}')  # 디버깅용 출력

    # 데이터 스케일링
    if scaler_X is not None:
        X_seq = scaler_X.transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)
    
    if scaler_y is not None:
        y_seq = scaler_y.transform(y_seq.reshape(-1, horizon)).reshape(-1, horizon, 1)
    
    actual = []
    calculated_actuals = []
    predictions = []

    # 예측 및 역변환 과정
    log.debug("prediction start")
    calculated_actual_price = last_candle.close_price
    for i in range(len(X_seq)):
        X_batch = X_seq[i:i+1]
        if X_batch.shape[0] == 0:
            continue
        
        real_idx = i + window_size + 1

        y_pred = model.predict(X_batch)
        
        if scaler_y is not None:
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, horizon))
        y_pred = y_pred.reshape(-1, horizon)
        
        if fluctuation:
            predicted_fluctation = y_pred[0][0]
            actual_fluctuation = y_seq[i][0]
            
            prev_actual_price = candles[real_idx-1].close_price
            actual_price = candles[real_idx].close_price
            
            predicted_price = calculated_actual_price * (1 + predicted_fluctation)
            calculated_actual_price *= (1 + actual_fluctuation)
            
            print(f"predicted: {predicted_fluctation}, actual:",actual_fluctuation)
            # print(f"predicted: {predicted_price}, actual:", actual_price)
            predictions.append(predicted_price)
            actual.append(actual_price)
            calculated_actuals.append(calculated_actual_price)
        else:
            predictions.append(y_pred)
            actual.append(y_seq[i:i+1])

    # predictions와 actual을 2D 배열로 변환
    if not fluctuation:
        predictions = np.concatenate(predictions, axis=0).flatten()
        actual = np.concatenate(actual, axis=0).reshape(-1, horizon)
        if scaler_y is not None:
            actual = scaler_y.inverse_transform(actual).flatten()
        else:
            actual = actual.flatten()

    # 플롯에 예측 및 실제 값 추가
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(actual, label='actual', color='blue')
    ax.plot(predictions, label='predicted', color='red')
    # ax.plot(calculated_actuals, label='calculated actual', color='green')
    ax.legend()
    plt.show()

except Exception as e:
    print(f"Error: {e}")
