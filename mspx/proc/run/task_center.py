#

__all__ = [
    "ZTaskCenterConf", "ZTaskCenter",
]

import os
from collections import OrderedDict, Counter, defaultdict
from mspx.nn import BK, ZmodelConf, Zmodel, ZTask, ZTaskConf
from mspx.utils import zlog, Conf, zwarn, default_json_serializer, ConfEntryCallback

# --
# ztask-center: confs for all the mods

class ZTaskCenterConf(Conf):
    def __init__(self):
        # vocab
        self.vocab_save_dir = "./"  # place to store all vocabs
        self.vocab_load_dir = ""  # place to load pre-built vocabs
        self.vocab_force_rebuild = False  # force rebuild all vocabs
        # model
        self.mconf = ZmodelConf()
        # task confs
        self.tcs = ConfEntryCallback(lambda s: self.callback_entries(s, T=ZTaskConf))
        # --

    # return all task confs
    def get_all_tconfs(self):
        ret = []
        names = [z[0] for z in self.tcs]
        ret += [(n, getattr(self, n)) for n in names]
        return ret

class ZTaskCenter:
    def __init__(self, conf: ZTaskCenterConf):
        self.conf = conf
        # --
        # build them
        self.tasks = OrderedDict()
        for tname, tconf in conf.get_all_tconfs():
            tconf.name = tname  # note: simply the same name!
            task: ZTask = tconf.make_node(tc=self)
            assert task.name not in self.tasks, "Repeated task!!"
            self.tasks[task.name] = task
        # --
        self.model = None  # should build later!
        zlog(f"Build TaskCenter ok: {self}")

    def __repr__(self):
        return f"TaskCenter with: {list(self.tasks.keys())}"

    def get_task(self, task_name: str, df=None):
        return self.tasks.get(task_name, df)

    # --
    # vocabs

    def build_vocabs(self, d_center, try_load_vdir=None, save_vdir=None):
        # first try load vocabs
        if try_load_vdir is not None:
            load_info = self.load_vocabs(try_load_vdir, quite=True)
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

    def load_vocabs(self, v_dir: str = None, quite=False):
        if not v_dir:
            v_dir = self.conf.vocab_load_dir
        info = OrderedDict()
        for n, t in self.tasks.items():
            info[n] = t.load_vocab(v_dir)
        if not quite:
            zlog(f"Load vocabs from {v_dir}, success={info}")
        return info
        # --

    def save_vocabs(self, v_dir: str = None):
        if v_dir is None:
            v_dir = self.conf.vocab_save_dir
        for n, t in self.tasks.items():
            t.save_vocab(v_dir)
        zlog(f"Save vocabs to {v_dir}")
        # --

    # --
    # models & self

    def make_model(self, load_name='', preload_name='', quite=True):
        assert self.model is None
        model = Zmodel(self.conf.mconf)
        for t in self.tasks.values():
            assert t.mod is None
            t.mod = t.build_mod(model)  # "build_mod" only needs to return a built one!!
            model.add_mod(t.mod)
        if preload_name:
            model.load(preload_name, quite=quite)
        model.finish_mods()
        model.finish_build()
        if load_name:
            model.load(load_name, quite=quite)
        self.model = model
        return model

    @classmethod
    def load(cls, load_name: str, quite=True, tc_kwargs=None):
        path_json = BK.change_slname(load_name, add_suffix='.json', rm_suffixes=['.m'], rm_specs=True)
        path_model = BK.change_slname(load_name, add_suffix='.m', rm_suffixes=['.m'], rm_specs=False)
        conf = default_json_serializer.from_file(path_json)
        assert isinstance(conf, ZTaskCenterConf)
        if not conf.vocab_load_dir:  # assume same dir as model
            conf.vocab_load_dir = os.path.dirname(path_model)
        if tc_kwargs:
            conf.update_from_dict(tc_kwargs, _quite=quite)
        tc = cls(conf)  # make a tc
        tc.load_vocabs(quite=quite)  # load vocabs!
        tc.make_model()  # blank model
        tc.model.load(path_model, quite=quite)  # load model (note: simply load afterwards!)
        if not quite:
            zlog(f"Load {tc} from {load_name} ...")
        return tc

    def save(self, save_name: str, quite=True):
        path_json = BK.change_slname(save_name, add_suffix='.json', rm_suffixes=['.m'], rm_specs=True)
        path_model = BK.change_slname(save_name, add_suffix='.m', rm_suffixes=['.m'], rm_specs=False)
        default_json_serializer.to_file(self.conf, path_json)
        self.model.save(path_model, quite=quite)
        if not quite:
            zlog(f"Save {self} to {save_name} ...")
        # --
