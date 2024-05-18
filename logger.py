from datetime import datetime
import traceback

class Colors:
    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

def cprint(color, *args):
    now = datetime.now()
    current_timestring = now.strftime('%Y-%m-%d %H:%M:%S') + f".{now.microsecond}"
    
    text_parts = []
    for arg in args:
        if isinstance(arg, Exception):
            exception_text = ''.join(traceback.format_exception(None, arg, arg.__traceback__))
            text_parts.append(f"\n{exception_text}")
        else:
            text_parts.append(str(arg))
    text = ' '.join(text_parts)
    print(f'{current_timestring} {color}{text}{Colors.RESET}')

def debug(*args):
    cprint(Colors.MAGENTA, *args)

def verbose(*args):
    cprint(Colors.BLUE, *args)

def info(*args):
    cprint(Colors.GREEN, *args)

def warn(*args):
    cprint(Colors.YELLOW, *args)

def error(*args):
    cprint(Colors.RED, *args)