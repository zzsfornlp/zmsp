#

# colorful printing

__all__ = [
    "wrap_color",
]

try:
    from colorama import init as colorama_init
    colorama_init()
    from colorama import Fore, Back, Style
    RESET_ALL = Style.RESET_ALL
except:
    class Fore:
        BLACK = '\033[30m'
        RED = '\033[31m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN = '\033[36m'
        WHITE = '\033[37m'
        RESET = '\033[39m'

    class Back:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
        BLUE = '\033[44m'
        MAGENTA = '\033[45m'
        CYAN = '\033[46m'
        WHITE = '\033[47m'
        RESET = '\033[49m'
    RESET_ALL = '\033[0m'

class Special:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def wrap_color(s: str, fcolor: str=None, bcolor: str=None, smode: str=None):
    f_prefix = ("" if fcolor is None else getattr(Fore, str.upper(fcolor)))
    b_prefix = ("" if bcolor is None else getattr(Back, str.upper(bcolor)))
    s_prefix = ("" if smode is None else getattr(Special, str.upper(smode)))
    return "".join([f_prefix, b_prefix, s_prefix, s, RESET_ALL])
