#

# mspx.utils: some basic utils

from .color import *
from .conf import *
from .file import *
from .log import *
from .math import *
from .random import *
from .reg import *
from .seria import *
from .stream import *
from .system import *
from .task import *
from .utils import *

import os
import sys
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
class MspUtilsConf(Conf):
    def __init__(self):
        # logging
        self.log_stderr = True  # use stderr
        self.log_file = ""  # by default no LOG
        self.log_files = []
        self.log_magic_file = False  # add magic file
        self.log_level = 0
        self.log_cached = False
        self.log_append = False
        # random
        self.msp_seed = Random._init_seed
        # conf writing
        self.conf_output = ""  # if writing conf_output
        # ignoring prefixes
        self.ignored_prefixes = []  # ignoring prefixes
        # numpy
        self.np_raise = True
        # special conf base
        self.conf_sbase = {}
        self.conf_sbase2 = {}  # allow easier extra ones!
        self.conf_sbase3 = {}  # allow easier extra ones!
        # main dir
        self.zdir = ""
        # global cache dir
        self.global_cache_dir = "___cache"  # for transformers/datasets/...

# Calling once at start, manually init after the auto one, could override the auto one
def init(utils_conf: MspUtilsConf, extra_files: List=None):
    # re-init things!!
    if utils_conf.zdir:  # change running dir!
        mkdir_p(utils_conf.zdir)
        os.chdir(utils_conf.zdir)
    flist = []
    if utils_conf.log_stderr:
        flist.append(stderr)
    flist.append(utils_conf.log_file)
    flist.extend(utils_conf.log_files)
    if extra_files is not None:
        flist.extend(extra_files)
    Logger.init(flist, level=utils_conf.log_level, use_magic_file=utils_conf.log_magic_file,
                log_cached=utils_conf.log_cached, log_append=utils_conf.log_append)
    Random.init(utils_conf.msp_seed)
    Timer.init()
    # numpy
    if utils_conf.np_raise:
        np.seterr(all='raise')
    # init_print
    zlog(f"Start!! After manually init at {time.ctime()}")
    zlog(f"*cmd: {' '.join(argv)}", func="config")
    zlog(f"*system: {get_sysinfo()}", func="config")
    check_env_info()

# ====
# init everything for mspx

def strip_quotes(args):
    ret = [a[1:-1] if (len(a)>2 and a.startswith("'") and a.endswith("'")) else a for a in args]
    return ret

def extra_args_for_mp(args, rank: int):
    ext = []
    if rank != 0:
        ext.extend(["log_stderr:0", "log_magic_file:0", "conf_output:"])
        log_file_item = None
        for a in args:
            if "log_file:" in a:
                log_file_item = a
        if log_file_item is not None:
            ext.extend([f"{log_file_item}{rank}"])  # put into another file!
    return ext

def check_env_info():
    res = {}
    for key in ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'MASTER_ADDR', 'MASTER_PORT']:
        res[key] = os.environ.get(key)
    zlog(f"ENV_INFO: ARGS = {sys.argv}, Env = {res}")

def init_everything(main_conf: Conf, args: Iterable[str]=None, add_utils=True, add_nn=True, sbase_getter=None, rank=None):
    if args is None:
        args = sys.argv[1:]
    list_args = list(args)  # store it!
    if rank is not None:
        list_args.extend(extra_args_for_mp(list_args, rank))
    gconf = get_singleton_global_conf()
    # utils?
    _ignored_prefixes = None
    if add_utils:
        # first we try to init a MspUtilsConf to allow logging!
        utils_conf = MspUtilsConf()
        utils_conf.update_from_args(list_args, quite=True, check=False, add_global_key='')
        init(utils_conf)
        # add to gconf!
        gconf.add_subconf("utils", MspUtilsConf())
        # --
        # if add special ones
        _conf_sbase = utils_conf.conf_sbase.copy()
        _conf_sbase.update(utils_conf.conf_sbase2)  # overwrite
        _conf_sbase.update(utils_conf.conf_sbase3)  # overwrite
        if len(_conf_sbase) > 0:
            zlog(f"Add sbase of {_conf_sbase}")
            sbase_args = strip_quotes(sbase_getter(**_conf_sbase))
            if len(list_args)>0 and ':' not in list_args[0]:  # add after _conf
                list_args = [list_args[0]] + sbase_args + list_args[1:]
            else:  # add to front!!
                list_args = sbase_args + list_args
        _ignored_prefixes = utils_conf.ignored_prefixes
        # --
    # nn?
    if add_nn:
        from mspx.nn import BK
        gconf.add_subconf("nn", BK.BKNIConf())
    # --
    # then actual init
    all_argv = main_conf.update_from_args(list_args, ignored_prefixes=_ignored_prefixes)
    # --
    # init utils
    if add_utils:
        # write conf?
        from mspx.nn import BK
        if utils_conf.conf_output and (rank is None or BK.is_main_process()):
            with zopen(utils_conf.conf_output, 'w') as fd:
                for k, v in all_argv.items():
                    # todo(note): do not save this one!!
                    if k.split(".")[-1] not in ["conf_output", "log_file", "log_files", "conf_sbase", "conf_sbase2", "conf_sbase3", "zdir"]:
                        fd.write(f"{k}:{v}\n")
        # no need to re-init
    # --
    # init nn
    if add_nn:
        from mspx.nn import init as nn_init
        nn_init(gconf.nn)
    # --
    return main_conf
