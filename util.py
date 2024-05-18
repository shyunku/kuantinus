
from datetime import datetime


def get_time_code(timestamp: float, unit: int) -> int:
    # check is milli
    if timestamp > 9999999999:
        timestamp = timestamp // 1000
    return int(timestamp // (60 * unit))

# return timestamp in seconds
def get_timestamp_from_code(timecode: int, unit: int) -> float:
    return float(timecode * 60 * unit)

def get_timestring_from_code(timecode: int, unit: int) -> str:
    return datetime.fromtimestamp(get_timestamp_from_code(timecode, unit)).strftime('%Y-%m-%d %H:%M:%S')

def profitify(num=0):
    if num > 0:
        return f'+{num:.2f}'
    return f'{num:.2f}'

def profitify_rate(num=0):
    rate = num * 100
    if rate > 0:
        return f'+{rate:.5f}%'
    return f'{rate:.5f}%'