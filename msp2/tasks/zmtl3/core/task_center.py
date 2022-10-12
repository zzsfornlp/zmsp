#

# collection of tasks

__all__ = [
    "TaskCenterConf", "TaskCenter",
]

from typing import List
from collections import OrderedDict
from msp2.utils import Conf, ConfEntryChoices, zlog
from .task import ZTask, ZTaskConf
from .data_center import DataCenter

# --
class TaskCenterConf(Conf):
    def __init__(self):
        # vocab dir
        self.vocab_save_dir = "./"  # place to store all vocabs
        self.vocab_load_dir = ""  # place to load pre-built vocabs
        self.vocab_force_rebuild = False  # force rebuild all vocabs
        # --
        from ..mod.extract import ZTaskTrigConf, ZTaskArgConf, ZTaskMlmConf
        self.evt0 = ConfEntryChoices({"yes": ZTaskTrigConf(name='evt0'), "no": None}, "no")
        self.evt2 = ConfEntryChoices({"yes": ZTaskTrigConf(name='evt2'), "no": None}, "no")
        self.arg0 = ConfEntryChoices({"yes": ZTaskArgConf(name='arg0'), "no": None}, "no")
        self.arg1 = ConfEntryChoices({"yes": ZTaskArgConf(name='arg1'), "no": None}, "no")
        self.arg2 = ConfEntryChoices({"yes": ZTaskArgConf(name='arg2'), "no": None}, "no")
        self.mlm0 = ConfEntryChoices({"yes": ZTaskMlmConf(name='mlm0'), "no": None}, "no")
        # --
        from ..mat import ZTaskMatMConf, ZTaskMatRConf
        self.matM = ConfEntryChoices({"yes": ZTaskMatMConf(name='matM'), "no": None}, "no")
        self.matR = ConfEntryChoices({"yes": ZTaskMatRConf(name='matR'), "no": None}, "no")
        # --

    def get_all_tconfs(self):
        ret = []
        ret += [z for n,z in self.__dict__.items() if (n!="enc" and isinstance(z, ZTaskConf))]
        return ret

    @classmethod
    def _get_type_hints(cls):
        return {"vocab_load_dir": "zglob1"}  # easier finding!

class TaskCenter:
    def __init__(self, conf: TaskCenterConf):
        self.conf = conf
        # --
        # build them
        self.tasks = OrderedDict()
        for tconf in conf.get_all_tconfs():
            task: ZTask = tconf.build_task()
            assert task.name not in self.tasks, "Repeated task!!"
            self.tasks[task.name] = task
        # --
        zlog(f"Build TaskCenter ok: {self}")

    def __repr__(self):
        return f"TaskCenter with: {list(self.tasks.keys())}"

    def build_vocabs(self, d_center: DataCenter, try_load_vdir=None, save_vdir=None):
        # first try load vocabs
        if try_load_vdir is not None:
            load_info = self.load_vocabs(try_load_vdir, quiet=True)
        else:
            load_info = OrderedDict()
        load_names = [k for k,v in load_info.items() if v]
        # then build for those not loaded!
        build_names = []
        for n, t in self.tasks.items():
            if not load_info.get(n, False):  # if not loaded!
                t_datasets = d_center.get_datasets(task=n)  # obtain by task name!!
                # --
                assert t.vpack is None
                t.vpack = t.build_vocab(t_datasets)
                # --
                build_names.append(n)
                if save_vdir is not None:
                    t.save_vocab(save_vdir)
        zlog(f"Build vocabs: load {load_names} from {try_load_vdir}, build {build_names}")
        # --

    def load_vocabs(self, v_dir: str, quiet=False):
        info = OrderedDict()
        for n, t in self.tasks.items():
            info[n] = t.load_vocab(v_dir)
        if not quiet:
            zlog(f"Load vocabs from {v_dir}, success={info}")
        return info
        # --

    def save_vocabs(self, v_dir: str):
        for n, t in self.tasks.items():
            t.save_vocab(v_dir)
        zlog(f"Save vocabs to {v_dir}")
        # --

    def prepare_datasets(self, datasets: List):
        # prepare them all
        for dataset in datasets:
            for task in dataset.tasks:
                self.tasks[task].prepare_dataset(dataset)
        # --

    def build_mods(self, model):
        for t in self.tasks.values():
            assert t.mod is None
            t.mod = t.build_mod(model)  # "build_mod" only needs to return a built one!!
            model.add_mod(t.mod)
        model.finish_mods()
        # --
