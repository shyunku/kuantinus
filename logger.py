from datetime import datetime
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

log_level_limit = int(os.getenv('LOG_LEVEL', 0))

class LogLevel:
    DEBUG = 0
    VERBOSE = 1
    INFO = 2
    WARN = 3
    ERROR = 4

class Colors:
    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GREY = '\033[38;5;240m'
    ORANGE = '\033[38;5;208m'

def cwrite(color, *args):
    text_parts = []
    for arg in args:
        if isinstance(arg, Exception):
            exception_text = ''.join(traceback.format_exception(None, arg, arg.__traceback__))
            text_parts.append(f"\n{exception_text}")
        else:
            text_parts.append(str(arg))
    text = ' '.join(text_parts)
    return f'{color}{text}{Colors.RESET}'

def cprint(color, *args):
    print(cwrite(color, *args), flush=True)

def log(color, log_level, level, *args):
    if log_level < log_level_limit:
        return

    now = datetime.now()
    timestring = now.strftime('%Y-%m-%d %H:%M:%S') + f".{now.microsecond:06d}"
    timestring = cwrite(Colors.GREY, timestring)
    padded_level = level.ljust(7)
    padded_level = cwrite(color, f'[{padded_level}]')
    final = f'{padded_level} {timestring} {" ".join(map(str, args))}'
    print(final, flush=True)

def debug(*args):
    log(Colors.MAGENTA, LogLevel.DEBUG, "DEBUG", *args)

def verbose(*args):
    log(Colors.BLUE, LogLevel.VERBOSE, "VERBOSE", *args)

def info(*args):
    log(Colors.GREEN, LogLevel.INFO, "INFO", *args)

def warn(*args):
    log(Colors.YELLOW, LogLevel.WARN, "WARN", *args)

def error(*args):
    log(Colors.RED, LogLevel.ERROR, "ERROR", *args)