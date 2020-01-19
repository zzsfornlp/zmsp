# assert

from .log import zlog
from collections import Callable

# Checkings

def zfatal(ss=""):
    zlog(ss, func="fatal")
    raise RuntimeError(ss)

def zwarn(ss="", **kwargs):
    zlog(ss, func="warn", **kwargs)

def _get_v(ff):
    return ff() if isinstance(ff, Callable) else ff

def zcheck(ff, ss, fatal=True, level=3):
    if level >= Checker._level:
        check_val, check_ss = _get_v(ff), _get_v(ss)
        if not check_val:
            if fatal:
                zfatal(check_ss)
            else:
                zwarn(check_ss)

# should be used when debugging or only fatal ones, comment out if real usage
class Checker(object):
    _level = 3

    @staticmethod
    def init(level=3):
        Checker.set_level(level)

    @staticmethod
    def set_level(level):
        if level != Checker._level:
            zlog(f"Change check level from {Checker._level} to {level}")
        Checker._level = level
