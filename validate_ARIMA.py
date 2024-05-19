from upbit_types import *
from joblib import load
from statsmodels.tsa.arima.model import ARIMAResults
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.dates import AutoDateLocator, DateFormatter
import pandas as pd
import random
import mplfinance as mpf
import matplotlib.dates as mdates
import threading
import signal
import math
import json
import logger as log

# signal handler
def safe_close():
    plt.close('all')

def signal_handler(sig, frame):
    log.info('Exiting...')
    safe_close()
    exit(0)
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

model_idx = 51
initial_seed = 10000
invest_rate = 0.3

seed = initial_seed
seed_last_calculate_timecode = 0

# load props
model_name = f'model_{model_idx}'
with open(f'models/{model_name}/props.json', 'r') as f:
    props = json.load(f)
    log.verbose(props)
    coin = props['coin']
    unit = props['unit']
    window_size = props['window_size']
    horizon = props['horizon']

# model directory
model_name = f'model_{model_idx}'

# initialize market
markets = Markets()
markets.load_markets()
market: Market = markets.get(coin)

# load model
log.info(f'Loading ARIMA model...')
fit_model = ARIMAResults.load(f'models/{model_name}/arima_model.pkl')

predictions = {}
correctness = 0

def predict():
    candles = market.get_recent_candles(unit, count=window_size, allow_upbit_omission=True)
    last_candle: Candle = candles[-1]
    last_time = last_candle.kst_dt
    predict_times = [last_time + timedelta(minutes=unit * (i + 1)) for i in range(horizon)]
    log.debug(f'candles count={len(candles)}, last_time={last_time}, predict_times={predict_times}')

    # 데이터 추출 및 배열 생성
    data = np.array([c.close_price for c in candles])
    model = ARIMA(data, order=(props['model_order'][0], props['model_order'][1], props['model_order'][2]))
    fit_model = model.fit()


    # 예측
    forecast = fit_model.forecast(steps=horizon)
    
    # set predictions
    for i, t in enumerate(predict_times):
        timecode = get_time_code(t.timestamp(), unit)
        predictions[timecode] = forecast[i]

def get_predictions():
    global predictions
    # flatten
    predict_times = [datetime.fromtimestamp(get_timestamp_from_code(k, unit)) for k in predictions.keys()]
    predict_prices = list(predictions.values())
    return predict_times, predict_prices

# initial predict
predict()

# realtime data
max_candles = 50
candles = market.get_recent_candles(unit, max_candles)

def update_predictions():
    global candles, predictions, lock
    last_predict_timecode = get_time_code(datetime.now().timestamp(), unit)
    while True:
        if not plt.fignum_exists(fig.number):
            break
        
        time.sleep(1)
        
        current_timecode = get_time_code(datetime.now().timestamp(), unit)
        current_timecode_time = get_timestamp_from_code(current_timecode, unit)
        latency = datetime.now().timestamp() - current_timecode_time
        if current_timecode != last_predict_timecode and latency > 5:
            with lock:
                log.info('prediction updating...')
                predict()
                last_predict_timecode = current_timecode


def update_candles():
    global candles, lock
    while True:
        if not plt.fignum_exists(fig.number):
            break
        
        time.sleep(1)
        
        with lock:
            candles = market.get_recent_candles(unit, max_candles, include_now=True, allow_upbit_omission=True)

def candles_to_df(candles):
    data = {
        'date': [c.utc_dt for c in candles],
        'Open': [c.open_price for c in candles],
        'Close': [c.close_price for c in candles],
        'Low': [c.low_price for c in candles],
        'High': [c.high_price for c in candles],
        'Volume': [c.candle_acc_trade_volume for c in candles]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('Asia/Seoul')
    df.set_index('date', inplace=True)
    return df

df = candles_to_df(candles)

mc = mpf.make_marketcolors(up="r", down="b")
s = mpf.make_mpf_style(base_mpf_style='starsandstripes', marketcolors=mc)
fig, ax1 = plt.subplots(figsize=(16, 8))
fig.canvas.manager.window.setWindowTitle(f'Kuant Model #{model_idx}')

def format_tick(x, pos):
    return '{:.0f}'.format(x)

# hour_loc = mdates.HourLocator(interval=6)
# hour_loc.MAXTICKS = 5000
# minute_loc = mdates.MinuteLocator(byminute=[0,15,30,45])
# minute_loc.MAXTICKS = 2000

# 차트를 업데이트하는 함수
def update(frame):
    global df, lock, market, unit, seed_last_calculate_timecode, seed, initial_seed
    
    with lock:
        # DataFrame 업데이트
        df = candles_to_df(candles)
        ax1.clear()
        # ax1.xaxis.set_major_locator(hour_loc)
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax1.xaxis.set_minor_locator(minute_loc)
        ax1.yaxis.set_major_formatter(FuncFormatter(format_tick))
        
        ax1.xaxis.set_major_locator(AutoDateLocator())
        # ax1.set_xlim(df.index[0], df.index[-1] + pd.Timedelta(minutes=15*horizon))
        # ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # grid
        ax1.grid(True)
        
        now = datetime.now()
        value = seed + market.order_book.wallet * market.order_book.current_price
        current_profit = value - initial_seed
        current_profit_rate = current_profit / initial_seed
        mpf.plot(df, type='candle', ax=ax1, style=s, mav=(15,30,60), volume=False, 
            axtitle=f'Kuant#{model_idx} {coin} [{unit}m] Price Prediction {now.strftime("%Y-%m-%d %H:%M:%S")}' +
            f' Seed({value:.3f}) [{market.order_book.wallet:.2f}{coin}] ({profitify_rate(current_profit_rate)})')
        
        # 종가만 표시
        ax1.plot([e for e in range(len(candles))], df['Close'], color='magenta', marker='o', linestyle='dashed', linewidth=1, markersize=3)
        
        # 예측 가격 표시
        predict_times, predicted_prices = get_predictions()
        
        seed += market.order_book.tick()
        
        pred_x = [len(candles) + int((t.timestamp() - df.index[-1].timestamp()) // (60 * unit)) - 1 for t in predict_times]
        ax1.plot(pred_x, predicted_prices, color='lime', marker='o', linestyle='dashed', linewidth=1, markersize=3)
        for i, y in enumerate(predicted_prices):
            t = predict_times[i]
            timecode = get_time_code(t.timestamp(), unit)
            candle = market.get_candle(unit, timecode)
            prev_candle = market.get_candle(unit, timecode-1)
            y_offset = [0.0001, -0.0001][i % 2]
            
            if candle is not None:
                accuracy = (1 - (abs(y - candle.close_price) / candle.close_price)) * 100                
                ax1.text(pred_x[i], predicted_prices[i] * (1 + y_offset), f'{accuracy:.5f}%', color='orange', fontsize=8)
            else:
                # current candle is none
                not_judged = timecode > seed_last_calculate_timecode
                if prev_candle is not None and not_judged:
                    current_price = prev_candle.close_price
                    
                    if current_price < y:
                        buy = current_price
                        sell = y
                        volume = seed * invest_rate / buy
                        if volume <= 0:
                            log.debug('Order failed: seed not enough')
                        else:
                            seed += market.order_book.order(buy, sell, volume)
                    else:
                        log.debug(f'Expected price will be lower than current price: expected({y}) <= current({current_price})')
                    
                    seed_last_calculate_timecode = timecode
        
        fig.canvas.draw()
    return ax1,

lock = threading.Lock()
load_candle_thread = threading.Thread(target=update_candles)
load_candle_thread.daemon = True
load_candle_thread.start()

predict_thread = threading.Thread(target=update_predictions)
predict_thread.daemon = True
predict_thread.start()

# 애니메이션 실행
ani = FuncAnimation(fig, update, blit=False, interval=1000, save_count=50)
plt.show()