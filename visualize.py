from sklearn.metrics import mean_squared_error
import os
import joblib

from common_types import PlotLearning
from upbit_types import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import graphviz
import logger as log

os.environ["PATH"]+=os.pathsep+'C:/Program Files/Graphviz/bin/'

model_name = 'model_1'
candle_cnt = 1000
time_steps = 50
epochs = 200

# configure markets
markets = Markets()
markets.load_markets()

xrp_market: Market = markets.get('XRP')
candles = xrp_market.get_recent_candles(5, count=candle_cnt, include_now=True)

# 데이터 추출 및 배열 생성
X = np.array([[c.timestamp, c.open_price, c.low_price, c.high_price, c.candle_acc_trade_volume] for c in candles])
y = np.array([c.close_price for c in candles])

# make pd
df = pd.DataFrame(X, columns=["timestamp", "open_price", "low_price", "high_price", "candle_acc_trade_volume"])
df['close_price'] = y
# drop everything except closing
df.drop(['timestamp', 'open_price', 'low_price', 'high_price', 'candle_acc_trade_volume'], axis=1, inplace=True)
df['120_mv'] = df['close_price'].rolling(window=120).mean()
df['300_mv'] = df['close_price'].rolling(window=300).mean()

span = 120
ses = SimpleExpSmoothing(df['close_price']).fit(smoothing_level=2/(span+1), optimized=False)
ses.predict(start=0, end=len(df)-1).plot(label='SES')
ses = ses.fittedvalues.shift(-1)
df['120_ses'] = ses

span = 300
ses = SimpleExpSmoothing(df['close_price']).fit(smoothing_level=2/(span+1), optimized=False)
ses.predict(start=0, end=len(df)-1).plot(label='SES')
ses = ses.fittedvalues.shift(-1)
df['300_ses'] = ses

log.debug(df.head())
df.plot(figsize=(12, 6))
plt.show()