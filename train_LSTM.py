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

coin = 'BTC'
model_idx = 86
unit = 240
# input size
window_size = 100
# output size
horizon = 1
candle_cnt = 20000
batch_size = 512
epochs = 300
learning_rate = 0.00001
random_state = 30
input_columns = ['open_price', 'close_price', 'low_price', 'high_price', 'candle_acc_trade_volume']
scaler_method = 'minmax'

x_scaler_method = scaler_method
y_scaler_method = scaler_method
fluctuation = True

# model directory
model_name = f'model_{model_idx}'
scaler_X = None
scaler_y = None
stdtime = datetime(2024, 1, 1)
stdtime_code = get_time_code(stdtime.timestamp(), unit)

try:
    # configure markets
    markets = Markets()
    markets.load_markets()

    market: Market = markets.get(coin)
    log.info("Loaded", market)
    
    log.info("====================== Model information ======================")
    log.info("Model name: ", model_name)
    log.info("Coin: ", coin)
    log.info("Unit: ", unit)
    log.info("Window size: ", window_size)
    log.info("Horizon: ", horizon)
    log.info("Candle count: ", candle_cnt)
    log.info("Batch size: ", batch_size)
    log.info("Epochs: ", epochs)
    log.info("Learning rate: ", learning_rate)
    log.info("Random state: ", random_state)
    log.info("X scaler method: ", x_scaler_method)
    log.info("Y scaler method: ", y_scaler_method)
    log.info("Input columns: ", input_columns)
    log.info("Fluctuation: ", fluctuation)
    log.info("===============================================================")
    
    # candles = xrp_market.get_recent_candles(unit, count=candle_cnt, allow_upbit_omission=True)
    candles = market.get_candles_before(unit, stdtime_code, count=candle_cnt, allow_upbit_omission=True)

    # create input data
    if fluctuation:
        # extract difference
        fluctuationRates = []
        for i in range(1, len(candles)):
            fluctuationRates.append((candles[i].close_price - candles[i-1].close_price)/candles[i-1].close_price)
        X = np.array([[f] for f in fluctuationRates])
        y = np.array([f for f in fluctuationRates])
    else:
        X = np.array([[getattr(c, col) for col in input_columns] for c in candles])
        y = np.array([c.close_price for c in candles])

    def create_sequences(X, y, window_size, horizon):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size - horizon + 1):
            X_seq.append(X[i:(i + window_size)])
            y_seq.append(y[i + window_size:i + window_size + horizon]) 
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X, y, window_size, horizon)
    log.debug(f'x_shape={X_seq.shape}, y_shape={y_seq.shape}')

    # scale data
    if x_scaler_method == 'minmax':
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_scaled = X_scaled.reshape(X_seq.shape)
    else:
        X_scaled = X_seq
    if y_scaler_method == 'minmax':
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y_seq.reshape(-1, horizon))
        y_scaled = y_scaled.reshape(-1, horizon, 1)
    else:
        y_scaled = y_seq

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=random_state)

    # LSTM 모델 구축
    model = Sequential([
        LSTM(64, return_sequences=True, activation='tanh', input_shape=(window_size, X_train.shape[2])),
        Dropout(0.01),
        LSTM(32, return_sequences=False),
        Dropout(0.01),
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
    if scaler_y is None:
        y_pred = y_pred_scaled
    else:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test = scaler_y.inverse_transform(y_test.reshape(-1, horizon))

    mse = mean_squared_error(y_test, y_pred)
    log.info(f'Mean Squared Error: {mse}')

    evaluate_value = model.evaluate(X_test, y_test, verbose=0)
    log.debug("evaluate value", evaluate_value)
    log.debug("X_test length", len(X_test))
    
    log.info("MAE:", evaluate_value[1])
    log.info("MAPE:", sum(abs(y_test-y_pred)/y_test)/len(X_test))
    r2 = r2_score(y_test, y_pred)
    log.info(f'R-squared: {r2}')
    
    # make directory
    if not os.path.exists(f'models'):
        os.makedirs(f'models')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    # save model and scalers
    log.info(f'saving model and scalers...')
    model.save(f'models/{model_name}', save_format='tf')
    model.save_weights(f'models/{model_name}/model_weights.h5')
    
    if scaler_X is not None:
        joblib.dump(scaler_X, f'models/{model_name}/scaler_x.gz')
    if scaler_y is not None:
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
        'x_scaler_method': x_scaler_method,
        'y_scaler_method': y_scaler_method,
        'input_columns': input_columns,
        'fluctuation': fluctuation
    }
    # save properties to file
    with open(f'models/{model_name}/props.json', 'w') as f:
        json.dump(props, f, indent=4)
except Exception as e:
    log.error(e)