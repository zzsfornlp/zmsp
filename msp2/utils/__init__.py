#

# msp2.utils: some basic utils

from .algo import *
from .color import *
from .conf import *
from .file import *
from .log import *
from .math import *
from .random import *
from .reg import *
from .seria import *
from .system import *
from .task import *
from .utils import *

from typing import Union, Iterable, IO, List
from sys import stderr, argv
from platform import uname
import numpy as np
import time

def auto_init():
    Logger.init([stderr])       # default only writing stderr
    Random.init(quite=True)
    Timer.init()

# todo(note): first init with the default auto one
auto_init()

# to init conf
class Msp2UtilsConf(Conf):
    def __init__(self):
        # logging
        self.log_stderr = True  # use stderr
        self.log_file = ""  # by default no LOG
        self.log_files = []
        self.log_magic_file = False  # add magic file
        self.log_level = 0
        # random
        self.msp_seed = Random._init_seed
        # conf writing
        self.conf_output = ""  # if writing conf_output
        # numpy
        self.np_raise = True

# Calling once at start, manually init after the auto one, could override the auto one
def init(utils_conf: Msp2UtilsConf, extra_files: List=None):
    # re-init things!!
    flist = []
    if utils_conf.log_stderr:
        flist.append(stderr)
    flist.append(utils_conf.log_file)
    flist.extend(utils_conf.log_files)
    if extra_files is not None:
        flist.extend(extra_files)
    Logger.init(flist, utils_conf.log_level, utils_conf.log_magic_file)
    Random.init(utils_conf.msp_seed)
    Timer.init()
    # numpy
    if utils_conf.np_raise:
        np.seterr(all='raise')
    # init_print
    zlog(f"Start!! After manually init at {time.ctime()}")
    zlog(f"*cmd: {' '.join(argv)}", func="config")
    zlog(f"*platform: {' '.join(uname())}", func="config")

# ====
# init everything for msp2

def init_everything(main_conf: Conf, args: Iterable[str], add_utils=True, add_nn=True):
    list_args = list(args)  # store it!
    gconf = get_singleton_global_conf()
    # utils?
    if add_utils:
        # first we try to init a Msp2UtilsConf to allow logging!
        utils_conf = Msp2UtilsConf()
        utils_conf.update_from_args(list_args, quite=True, check=False, add_global_key='')
        init(utils_conf)
        # add to gconf!
        gconf.add_subconf("utils", Msp2UtilsConf())
    # nn?
    if add_nn:
        from msp2.nn import BK
        gconf.add_subconf("nn", BK.BKNIConf())
    # --
    # then actual init
    all_argv = main_conf.update_from_args(list_args)
    # --
    # init utils
    if add_utils:
        # write conf?
        if utils_conf.conf_output:
            with zopen(utils_conf.conf_output, 'w') as fd:
                for k, v in all_argv.items():
                    # todo(note): do not save this one!!
                    if k.split(".")[-1] not in ["conf_output", "log_file", "log_files"]:
                        fd.write(f"{k}:{v}\n")
        # no need to re-init
    # --
    # init nn
    if add_nn:
        from msp2.nn import init as nn_init
        nn_init(gconf.nn)
    # --
    return main_conf
