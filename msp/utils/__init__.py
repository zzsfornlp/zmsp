#

from .algo import AlgoHelper
from .check import zwarn, zfatal, zcheck, Checker
from .color import wrap_color
from .conf import Conf
from .log import zopen, zlog, printing, Logger
from .math import MathHelper
from .random import Random
from .seria import JsonRW, PickleRW
from .system import system, dir_msp, get_statm, FileHelper, extract_stack
from .task import Timer, StatRecorder
from .utils import Constants, Helper, NumHelper, StrHelper, ZObject

from sys import stderr, argv
from platform import uname

def auto_init():
    Logger.init([stderr])       # default only writing stderr
    Checker.init()
    Timer.init()

# if not calling init explicitly, then just use default auto one
auto_init()

# Calling once at start, manually init after the auto one, could override the auto one
def init(extra_file=None, msp_seed=None):
    #
    flist = [stderr]
    if extra_file:
        flist.append(extra_file)
    Logger.init(flist)
    Random.init(msp_seed)
    # init_print
    zlog("Start!! After manually init.")
    zlog("*cmd: %s" % ' '.join(argv), func="config")
    zlog("*platform: %s" % ' '.join(uname()), func="config")

# global recorder
GLOBAL_RECORDER = StatRecorder(False)
