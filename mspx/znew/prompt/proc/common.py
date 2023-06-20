#

# common ones for train/test

from mspx.utils import Conf, zlog, init_everything, ConfEntryCallback, default_json_serializer
from ..model import ModelConf
from ..task import TaskConf
from .run import RunnerConf

class CommonConf(Conf):
    def __init__(self):
        self.model_conf = ModelConf()
        self.task_conf = ConfEntryCallback((lambda s: TaskConf.find_conf(s)))
        self.run_conf = RunnerConf()
        self.accelerate_conf = AccelerateConf()
        # --
        # data
        from ..data import DatasetConf, MyDataset
        self.train0 = DatasetConf.direct_conf(wset='train0')
        self.dev0 = DatasetConf.direct_conf(wset='dev0')
        self.test0 = DatasetConf.direct_conf(wset='test0')


class AccelerateConf(Conf):
    def __init__(self):
        self.mixed_precision = None

    def get_accelerator(self):
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision=self.mixed_precision)
        return accelerator

    @staticmethod
    def get_acce(args):
        _conf = AccelerateConf()
        _conf.update_from_args(args, quite=True, check=False, add_global_key='')
        ret = _conf.get_accelerator()
        zlog(f"Get accelerator:{ret.state}")
        return ret
