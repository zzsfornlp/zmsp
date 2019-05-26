# assert

from .log import zlog
from collections import Callable

# Checkings: also including some shortcuts for convenience

class ZException(RuntimeError):
    def __init__(self, s):
        super().__init__(s)

def zfatal(ss=""):
    zlog(ss, func="fatal")
    raise ZException(ss)

def zwarn(ss=""):
    zlog(ss, func="warn")

def zcheck(ff, ss, func="error", forced=False):
    if Checker.enabled(func) or forced:
        Checker.check(ff, ss, func)

# should be used when debugging or only fatal ones, comment out if real usage
class Checker(object):
    _checker_filters = {"warn": True, "error": True, "fatal": True}
    _checker_handlers = {"warn": (lambda: 0), "error": (lambda: zfatal("ERROR")), "fatal": (lambda: zfatal("FATAL-ERROR"))}

    @staticmethod
    def init():
        # todo(0): this may be enough
        pass

    @staticmethod
    def _get_v(v):
        if isinstance(v, Callable):
            return v()
        else:
            return v

    @staticmethod
    def check(expr, ss, func):
        if not Checker._get_v(expr):
            zlog(Checker._get_v(ss), func=func)
            Checker._checker_handlers[func]()

    @staticmethod
    def enabled(func):
        return Checker._checker_filters[func]
