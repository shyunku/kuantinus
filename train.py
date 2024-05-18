from sklearn.metrics import mean_squared_error
import os
import joblib

from common_types import DynamicLRScheduler, PlotLearning
from upbit_types import Markets
from upbit_types import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import logger as log

os.environ["PATH"]+=os.pathsep+'C:/Program Files/Graphviz/bin/'

coin = 'XRP'
model_idx = 47
unit = 15
window_size = 500
horizon = 10
candle_cnt = 10000
batch_size = 256
epochs = 500
learning_rate = 0.006
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
    # X = np.array([[c.open_price, c.low_price, c.high_price, c.candle_acc_trade_volume, c.close_price] for c in candles])
    X = np.array([[c.close_price] for c in candles])
    y = np.array([c.close_price for c in candles])

    def create_sequences(X, y, window_size, horizon):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size - horizon + 1):
            X_seq.append(X[i:(i + window_size)])
            y_seq.append(y[i + window_size:i + window_size + horizon]) 
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X, y, window_size, horizon)
    log.debug(f'x_shape={X_seq.shape}, y_shape={y_seq.shape}')

    # 데이터 스케일링
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_seq.reshape(-1, X_seq.shape[2]))
    X_scaled = X_scaled.reshape(X_seq.shape)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_seq.reshape(-1, horizon))
    y_scaled = y_scaled.reshape(-1, horizon, 1)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=random_state)

    # LSTM 모델 구축
    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh', input_shape=(window_size, X_train.shape[2])),
        # Dropout(0.01),
        LSTM(64, return_sequences=False),
        # Dropout(0.01),
        Dense(horizon)
    ])

    # 모델 컴파일
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

    model.summary()

    # 모델 학습
    plot_losses = PlotLearning(model_idx=model_idx)
    dynamic_lr = DynamicLRScheduler(base_lr=learning_rate)
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, callbacks=[plot_losses, dynamic_lr])

    # 예측
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, horizon))

    mse = mean_squared_error(y_test, y_pred)
    log.debug(f'Mean Squared Error: {mse}')

    evaluate_value = model.evaluate(X_test, y_test, verbose=0)
    log.debug("MAE:", evaluate_value[1])
    log.debug("MAPE:", sum(abs(y_test-y_pred)/y_test)/len(X_test))
    r2 = r2_score(y_test, y_pred)
    log.debug(f'R-squared: {r2}')
    
    # make directory
    if not os.path.exists(f'models'):
        os.makedirs(f'models')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    # save model and scalers
    log.info(f'saving model and scalers...')
    model.save(f'models/{model_name}', save_format='tf')
    joblib.dump(scaler_X, f'models/{model_name}/scaler_x.gz')
    joblib.dump(scaler_y, f'models/{model_name}/scaler_y.gz')

    # save properties
    props = {
        'coin': coin,
        'unit': unit,
        'window_size': window_size,
        'horizon': horizon,
        'candle_cnt': candle_cnt,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'random_state': random_state,
        'mse': mse,
        'r2': r2,
    }
    # save properties to file
    with open(f'models/{model_name}/props.json', 'w') as f:
        json.dump(props, f, indent=4)
except Exception as e:
    log.error('Failed to train model:', e)