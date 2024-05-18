import string
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from util import *
from upbit_req import *
import sqlite3
import pytz
import logger as log

available_candle_units = [1, 3, 5, 10, 15, 30, 60, 240]

conn = sqlite3.connect('datafiles/db.sqlite3')
c = conn.cursor()

kst = pytz.timezone('Asia/Seoul')

class Candle:
    def __init__(self, unit, utc_dt, kst_dt, open_price, close_price, low_price, high_price, timestamp, candle_acc_trade_price, candle_acc_trade_volume):
        self.minute_unit = unit
        self.utc_dt = None
        self.kst_dt = None
        
        if utc_dt is not None:
            self.utc_dt = datetime.fromisoformat(utc_dt.replace('Z', '+00:00'))
        if kst_dt is not None:
            if type(kst_dt) == str:
                self.kst_dt = datetime.fromisoformat(kst_dt.replace('+09:00', ''))
            elif type(kst_dt) == datetime:
                self.kst_dt = kst_dt
        
        self.open_price = open_price
        self.close_price = close_price
        self.low_price = low_price
        self.high_price = high_price
        self.timestamp = timestamp
        self.candle_acc_trade_price = candle_acc_trade_price # 누적 거래 금액
        self.candle_acc_trade_volume = candle_acc_trade_volume

        self.local_time_str = self.kst_dt.strftime("%Y.%m.%d %H:%M:%S")
        
    def timecode(self):
        return get_time_code(self.kst_dt.timestamp(), self.minute_unit)

    def __str__(self):
        return f'[{self.minute_unit}m] {self.local_time_str} ({self.candle_acc_trade_volume}) {self.open_price} -> {self.close_price} ({self.low_price} ~ {self.high_price})'


class SellOrder:
    def __init__(self, buy_price, sell_price, volume):
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.volume = volume

    def __str__(self):
        return f'{self.buysell} {self.price} {self.volume}'

class OrderBook:
    def __init__(self, coin):
        self.coin = coin
        self.current_price = 0
        self.sells: Dict[float, List[SellOrder]] = {}
        self.wallet = 0
        
    def order(self, buy_price, sell_price, volume):
        self.wallet += volume
        self.sells[sell_price] = self.sells.get(sell_price, [])
        self.sells[sell_price].append(SellOrder(buy_price, sell_price, volume))
        log.debug(f'Buy: {buy_price} -> {sell_price} ({volume}) -{buy_price * volume}')
        return -buy_price * volume
    
    # exclusive bound
    def tick(self):
        profit = 0
        for price, orders in self.sells.items():
            if price > self.current_price:
                continue
            for order in orders:
                profit += order.sell_price * order.volume
                self.wallet -= order.volume
                log.debug(f'Sell: {order.buy_price} -> {order.sell_price} ({order.volume}) +{order.sell_price * order.volume}')
            self.sells[price] = []
        return profit
    

    def __str__(self):
        str = ''
        for price, orders in self.buys.items():
            str += f'BUY {price}: {len(orders)}\n'
        for price, orders in self.sells.items():
            str += f'SELL {price}: {len(orders)}\n'
        return str


class Market:
    def __init__(self, market_name, korean_name, english_name):
        self.market_name = market_name
        self.korean_name = korean_name
        self.english_name = english_name
        self.quote_currency, self.base_currency = market_name.split('-')
        
        self.order_book = OrderBook(self.base_currency)
        self.candles_group: Dict[string, Dict[int, Candle]] = {}
        

    def __str__(self):
        return f'[{self.korean_name}] {self.base_currency}/{self.quote_currency} ({self.english_name})'
    
    def subscribe(self):
        # implement
        return
    
    def update_current_price(self, price):
        self.order_book.current_price = price
    
    # DB에서 time_code 빈 구간 찾기
    def find_missing_intervals_on_db(self, unit, start=None, end=None) -> List[int]:
        if start is None:
            # found most oldest
            c.execute('''
                SELECT time_code FROM candles
                WHERE quote=? AND base=? AND unit=? ORDER BY time_code ASC LIMIT 1
            ''', (self.quote_currency, self.base_currency, unit))
            result = c.fetchone()
            if result is not None:
                start = result[0]
            else:
                return []
        if end is None:
            # found most recent
            c.execute('''
                SELECT time_code FROM candles
                WHERE quote=? AND base=? AND unit=? ORDER BY time_code DESC LIMIT 1
            ''', (self.quote_currency, self.base_currency, unit))
            result = c.fetchone()
            if result is not None:
                end = result[0]
            else:
                return []
        
        if start > end:
            return []
        
        # load candles from db
        c.execute('''
            SELECT time_code FROM candles
            WHERE quote=? AND base=? AND unit=? AND time_code>=? AND time_code<=?
        ''', (self.quote_currency, self.base_currency, unit, start, end))
        candles = c.fetchall()
        total_len, expected_len = len(candles), end - start + 1
        
        if total_len == expected_len:
            return []
        
        if total_len == 0:
            return [(start, end)]
        
        # find missing intervals with binary split
        missings = []
        s1, e1 = start, start + (end - start) // 2
        s2, e2 = e1 + 1, end
        
        m1 = self.find_missing_intervals_on_db(unit, s1, e1)
        m2 = self.find_missing_intervals_on_db(unit, s2, e2)
        
        if len(m1) > 0:
            missings.extend(m1)
        if len(m2) > 0:
            missings.extend(m2)
        
        # sort by start time
        missings = sorted(missings, key=lambda x: x[0])
        
        # combine if possible
        combined_missings = []
        for interval in missings:
            if combined_missings and combined_missings[-1][1] + 1 >= interval[0]:
                combined_missings[-1] = (combined_missings[-1][0], max(combined_missings[-1][1], interval[1]))
            else:
                combined_missings.append(interval)
        
        return combined_missings
    
    # Memory에서 time_code 빈 구간 찾기
    def find_missing_intervals(self, unit, start=None, end=None) -> List[int]:
        candle_map = self.candles_group.get(unit)
        if candle_map is None:
            return []
        
        if start is None:
            # found most oldest
            start = min(candle_map.keys())
        if end is None:
            # found most recent
            end = max(candle_map.keys())
        
        if start > end:
            return []
        
        if unit not in available_candle_units:
            log.error(f'Available candle units: {available_candle_units}')
            return []
        
        # linear search
        missings = []
        c_start, c_end = None, None

        for timecode in range(start, end + 1):
            if candle_map.get(timecode) is None:
                if c_start is None:
                    c_start = timecode
                c_end = timecode
            else:
                if c_start is not None and c_end is not None:
                    missings.append((c_start, c_end))
                    c_start, c_end = None, None

        if c_start is not None and c_end is not None:
            missings.append((c_start, c_end))

        return missings
    
    def load_candles_from_db(self, unit, start=None, end=None):
        if self.candles_group.get(unit) is None:
            self.candles_group[unit] = {}
        
        c.execute('''
            SELECT time_code, low_price, high_price, open_price, close_price, volume
            FROM candles
            WHERE quote=? AND base=? AND unit=? AND time_code>=? AND time_code<=?
        ''', (self.quote_currency, self.base_currency, unit, start, end))
        candles = c.fetchall()
        for cc in candles:
            kst_dt = datetime.fromtimestamp(get_timestamp_from_code(cc[0], unit)).strftime('%Y-%m-%d %H:%M:%S')
            candle = Candle(unit, None, kst_dt, cc[3], cc[4], cc[1], cc[2], None, None, cc[5])
            self.candles_group[unit][cc[0]] = candle
        
        log.info(f'Loaded {len(candles)} candles from db')
        
    
    # 최근 캔들 리스트 반환 (현재 시간 캔들 제외)
    def get_recent_candles(self, unit, count=None, include_now=False, allow_upbit_omission=False, silent=False) -> List[Candle]:
        current_timecode = get_time_code(datetime.now().timestamp(), unit)
        t_end = current_timecode
        if not include_now:
            t_end -= 1            
        t_start = t_end - count + 1
        log.verbose(f'Searching for {unit}m candles ({t_start} ~ {t_end})...')
        
        # search on memory
        candle_group = self.candles_group.get(unit)
        if candle_group is not None:
            log.verbose(f'Found memory candle group: {unit}m')
            missings = self.find_missing_intervals(unit, t_start, t_end)
            
            if len(missings) > 0:
                result_string = ', '.join([f"{start}~{end}" if start != end else f"{start}" for start, end in missings])
                log.verbose(f'Missing memory candle intervals detected: {result_string}')
                
                # fetch missings (from db)
                db_missings_combined = []
                for start, end in missings:
                    # search on db
                    db_missings = self.find_missing_intervals_on_db(unit, start, end)
                    db_missings_combined.extend(db_missings)
                
                if len(db_missings_combined) > 0:
                    result_string = ', '.join([f"{start}~{end}" if start != end else f"{start}" for start, end in db_missings_combined])
                    log.verbose(f'Missing db candle intervals detected: {result_string}')
                    
                    # fetch missings (from api)
                    for start, end in db_missings_combined:
                        self.get_candles_from_api(unit, end=end, count=end-start+1, silent=silent)
                else:
                    for start, end in missings:
                        # load from db
                        self.load_candles_from_db(unit, start, end)
            else:
                log.verbose(f'all candles are loaded on memory')
        else:
            log.verbose(f'Missing memory candle group: {unit}m, searching on db...')
            
            # search on db
            db_missings = self.find_missing_intervals_on_db(unit, t_start, t_end)
            if len(db_missings) > 0:
                result_string = ', '.join([f"{start}~{end}" if start != end else f"{start}" for start, end in db_missings])
                log.verbose(f'Missing db candle intervals detected: {result_string}')
                
                # fetch missings (from api)
                for start, end in db_missings:
                    self.get_candles_from_api(unit, end=end, count=end-start+1, silent=silent)
            
            # load from db
            self.load_candles_from_db(unit, t_start, t_end)
        
        # test continousity
        candle_group = self.candles_group.get(unit)
        if candle_group is None:
            raise Exception(f'failed to load candles even after safe-load for {self.market_name}')
        
        missings = self.find_missing_intervals(unit, t_start, t_end)
        if len(missings) > 0:
            result_string = ', '.join([f"{start}~{end} ({get_timestring_from_code(start, unit)} ~ {get_timestring_from_code(end, unit)})" if start != end else f"{start} ({get_timestring_from_code(start, unit)})" for start, end in missings])
            if not allow_upbit_omission:
                raise Exception(f'missing candles detected even after safe-load for {self.market_name}: {result_string}')
            else:
                log.warn(f'missing candles detected even after safe-load for {self.market_name}: {result_string}, but passing by allowance.')
        
        # validated
        result = []
        for timecode in range(t_start, t_end):
            candle = candle_group.get(timecode)
            if candle is not None:
                result.append(candle)
        
        return result
        
    # API에서 캔들 리스트 반환 (count 무제한)
    def get_candles_from_api(self, unit, end=None, count=100, silent=False) -> List[Candle]:
        prev_batch_start_time = end
        remains = count
        
        max_candle_batch = 200 # const.
        load_batch = count < max_candle_batch and count or max_candle_batch

        while remains > 0:
            try:
                candles = self.get_candle_bundle_from_api(unit, end=prev_batch_start_time, count=load_batch)
                if len(candles) == 0:
                    log.warn(f'No candles for {self.market_name}')
                    return
                prev_batch_start_time = candles[0].timecode() - 1
                remains -= len(candles)
                if remains < 0:
                    remains = 0
                if not silent:
                    log.verbose(f'Fetched {len(candles)} candles ({candles[0].timecode()}~{candles[-1].timecode()}), {remains}/{count} remains')
            except Exception as e:
                log.error(f'Failed to fetch candles for {self.market_name}', e)
                return

        if not silent:
            log.info(f'Fetched {count} candles')
    
    
    # API에서 캔들 번들 반환 (count 200 제한) + memory와 db에 저장
    def get_candle_bundle_from_api(self, unit, end: int = None, count=200) -> List[Candle]:
        current_timecode = get_time_code(datetime.now().timestamp(), unit)
        
        if not(unit in available_candle_units):
            log.warn(f'Available candle units: {available_candle_units}')
            return

        if self.candles_group.get(unit) is None:
            self.candles_group[unit] = {}

        params = {
            'market': self.market_name,
            'count': count,
        }
        if end is not None:
            timestamp = get_timestamp_from_code(end + 1, unit)
            kst_dt = datetime.fromtimestamp(timestamp, kst)
            utc_dt = kst_dt.astimezone(pytz.utc)
            params['to'] = utc_dt.strftime('%Y-%m-%d %H:%M:%S')

        candle_batch = []
        new_candles = []
        resp = upbit_get('/v1/candles/minutes/' + str(unit), params)
        for item in resp.json():
            candle = Candle(
                item['unit'],
                item['candle_date_time_utc'],
                item['candle_date_time_kst'],
                item['opening_price'],
                item['trade_price'],
                item['low_price'],
                item['high_price'],
                item['timestamp'],
                item['candle_acc_trade_price'],
                item['candle_acc_trade_volume']
            )
            timecode = get_time_code(candle.kst_dt.timestamp(), unit)
            self.candles_group[unit][timecode] = candle
            candle_batch.append(candle)
            new_candles.append(candle)

        # sort by timestamp, asc
        # self.candles_group[unit] = sorted(self.candles_group[unit], key=lambda x: x.timestamp)
        candle_batch = sorted(candle_batch, key=lambda x: x.timestamp)
        
        # save new candles
        conn.execute('BEGIN')
        try:
            for candle in new_candles:
                if candle.timecode() == current_timecode:
                    continue
                c.execute('''
                    INSERT INTO candles (quote, base, unit, time_code, low_price, high_price, open_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(quote, base, unit, time_code) DO UPDATE SET
                        low_price=excluded.low_price,
                        high_price=excluded.high_price,
                        open_price=excluded.open_price,
                        close_price=excluded.close_price,
                        volume=excluded.volume
                ''', (self.quote_currency, self.base_currency, candle.minute_unit, candle.timecode(), candle.low_price, candle.high_price, candle.open_price, candle.close_price, candle.candle_acc_trade_volume))
            conn.commit()
        except Exception as e:
            log.error('Failed to save candle', e)
            conn.rollback()
            raise e
        
        # update current price
        self.update_current_price(candle_batch[-1].close_price)

        return candle_batch


class Markets:
    def __init__(self):
        self.markets = {}

    def __str__(self):
        return f'{self.markets}'

    def load_markets(self):
        resp = upbit_get("/v1/market/all")
        for item in resp.json():
            market = Market(item['market'], item['korean_name'], item['english_name'])
            if market.quote_currency != 'KRW':
                continue
            self.add(market)

    def add(self, market):
        self.markets[market.base_currency] = market

    def get(self, base_currency):
        return self.markets[base_currency]