from upbit_types import Market, Markets

unit = 5

try:
  markets = Markets()
  markets.load_markets()
  market: Market = markets.get('XRP')

  candles = market.get_recent_candles(unit=unit, count=500)
except Exception as e:
  print(e)