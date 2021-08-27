#

# some common functions for running, go!

__all__ = [
    "ZOverallConf",
]

from typing import List
from msp2.utils import Conf, init_everything
from .task_center import TaskCenterConf, TaskCenter
from .data_center import DataCenterConf, DataCenter
from .run_center import RunCenterConf, RunCenter
from ..zmod import ZModelConf, ZModel

# --

class ZOverallConf(Conf):
    def __init__(self):
        self.tconf = TaskCenterConf()  # task conf
        self.dconf = DataCenterConf()  # data conf
        self.rconf = RunCenterConf()  # run conf
        self.mconf = ZModelConf()  # model conf
