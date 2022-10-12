#

# some common functions for running, go!

__all__ = [
    "ZOverallConf",
]

from typing import List
from msp2.utils import Conf, init_everything
from msp2.nn.l3 import ZmodelConf
from .task_center import TaskCenterConf
from .data_center import DataCenterConf
from .run_center import RunCenterConf

# --

class SpecialTestingConf(Conf):
    def __init__(self):
        self.stn = ""  # simply for better printing!
        self.st_args = []  # list of extra args to try

class ZOverallConf(Conf):
    def __init__(self):
        self.tconf = TaskCenterConf()  # task conf
        self.dconf = DataCenterConf()  # data conf
        self.mconf = ZmodelConf()  # general model conf
        self.rconf = RunCenterConf()  # run conf
        self.st_conf = SpecialTestingConf()  # special testing
        # --
