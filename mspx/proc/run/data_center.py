# --

__all__ = [
    "ZDatasetGroupConf", "ZDataCenterConf", "ZDataCenter",
]

from collections import OrderedDict
import numpy as np
from mspx.utils import Conf, Configurable, zglob1, zglobs, ConfEntryCallback, zlog, Random
from .data import *
from .helper import *

# --
# data center

class ZDatasetGroupConf(ZDatasetConf):
    def __init__(self):
        super().__init__()
        self.group_files = []  # List: # -> "input_file" or Dict: sub_name -> "input_file"
        self.group_sep = True  # separate individual files for the group!
        self.group_sample_rate = SVConf().direct_update(val=1.)  # group (outside) sample rate
        self.group_sample_betas = [1.]  # inside sample rate
        self.group_eval_weight = 1.  # weight for final eval

class ZDataCenterConf(Conf):
    def __init__(self):
        # base ones
        _DATA_MAXN=10  # this should be enough!!
        for ii in range(_DATA_MAXN):
            for wset in ["train", "dev", "test"]:
                setattr(self, f"{wset}{ii}", ZDatasetGroupConf())
        self.testM = ZDatasetGroupConf()  # shortcut: main-test that can override others!
        # --
        # special default ones
        self.d_input_dir = ""
        self.d_tasks = []
        # --

    def get_dconfs(self, prefix='', try_main=True):
        # --
        if try_main:  # note: special shortcut for main ones!
            ret0 = self.get_dconfs(prefix+'M', False)
            if len(ret0) > 0:
                return ret0
        # --
        d_input_dir = self.d_input_dir
        d_tasks = self.d_tasks
        ret = OrderedDict()
        for kk, vv in self.__dict__.items():
            if kk.startswith(prefix) and isinstance(vv, ZDatasetGroupConf) and vv.group_files:
                # assign in these cases!
                if d_input_dir and not vv.input_dir:
                    vv.input_dir = d_input_dir
                if d_tasks and not vv.tasks:
                    vv.tasks = d_tasks
                # --
                ret[kk] = vv
        return ret

class ZDataCenter:
    def __init__(self, conf: ZDataCenterConf, eager_prefixes=('train', 'dev', 'test')):
        self.conf = conf
        # --
        # load and prepare them
        self.dataset_groups = OrderedDict()  # note: (group_name, [sub_name])
        self.train_sample_svs = OrderedDict()  # sv:sample_rate for training
        # --
        for pp in eager_prefixes:
            self.get_datasets(prefix=pp)
        # --
        zlog(f"Build DataCenter ok: {self}")

    def __repr__(self):
        ss = [f"{k}:{len(v)}" for k,v in self.dataset_groups.items()]
        return f"DataCenter({','.join(ss)})"

    def get_scheduled_values(self):
        return OrderedDict([(f"_sr_{k}", v[0]) for k,v in self.train_sample_svs.items()])

    def add_dataset_group(self, name: str, conf: ZDatasetGroupConf):
        group = []
        if not conf.group_sep:
            assert isinstance(conf.group_files, list)
            d = ZDataset(conf, name=name, input_file=conf.group_files)
            group.append(d)
        else:  # find files with sep
            sub_iter = list(conf.group_files.items()) if \
                isinstance(conf.group_files, dict) else [(i, z) for i,z in enumerate(conf.group_files)]
            for one_ii, (one_subname, one_filename) in enumerate(sub_iter):
                d = ZDataset(conf, name=f"{name}_{one_subname}", input_file=one_filename)
                group.append(d)
        # add group
        assert name not in self.dataset_groups
        self.dataset_groups[name] = group
        if name.startswith("train"):
            _betas = conf.group_sample_betas
            if len(_betas) < len(group):  # note: fill the last one!
                _betas = _betas + [_betas[-1]] * (len(group) - len(_betas))
            self.train_sample_svs[name] = \
                (ScheduledValue(conf.group_sample_rate, name=f"sr_{name}"), _betas)
        # --
        return group

    def get_dataset_groups(self, prefix=''):
        ret = OrderedDict()
        dconfs = self.conf.get_dconfs(prefix)
        for kk, cc in dconfs.items():
            if kk not in self.dataset_groups:
                group = self.add_dataset_group(kk, cc)
                self.dataset_groups[kk] = group
            ret[kk] = self.dataset_groups[kk]
        return ret

    def get_datasets(self, prefix='', task='', extra_filters=None):
        # filter by name
        candidates = sum(self.get_dataset_groups(prefix).values(), [])
        # filter by task
        if task:
            candidates = [z for z in candidates if (task in z.tasks)]
        # filter by filter
        if extra_filters:
            for ff in extra_filters:
                candidates = [z for z in candidates if ff(z)]
        return candidates

    # yield the next dataset (loop forever)
    def yield_dataset(self, prefix='train'):
        dg = self.get_dataset_groups(prefix)
        all_ds = []  # List[List[Dataset]]
        all_svs = []
        all_inner_rates = []
        for group_name, group_datasets in dg.items():
            _sv, _betas = self.train_sample_svs[group_name]
            all_ds.append(group_datasets)   # directly put dataset!
            one_inner_rates = np.asarray(_betas[:len(group_datasets)])
            all_inner_rates.append(one_inner_rates / one_inner_rates.sum())  # inner sample
            all_svs.append(_sv)
        _gen = Random.get_generator('stream')
        _n_groups = len(all_svs)
        while True:
            # choose the outer
            if len(all_svs) == 1:
                cur_gidx = 0  # simply 1
            else:
                pvals = np.asarray([z.value for z in all_svs])
                pvals = pvals / pvals.sum()
                cur_gidx = _gen.choice(_n_groups, p=pvals)  # choose group
            # choose the inner
            pvals2 = all_inner_rates[cur_gidx]
            if len(pvals2) == 1:
                cur_iidx = 0
            else:
                cur_iidx = _gen.choice(len(pvals2), p=pvals2)  # choose inner one
            # choose that one!
            chosen = all_ds[cur_gidx][cur_iidx]
            yield chosen  # yield it!
        # --

    # yield batches
    def yield_batches(self, prefix='train', each_time=1, loop=True):
        dataset_yielders = {}  # id(dataset) -> yielder
        for dataset in self.yield_dataset(prefix):
            if id(dataset) not in dataset_yielders:
                dataset_yielders[id(dataset)] = dataset.yield_batches(loop=loop)
            cur_yielder = dataset_yielders[id(dataset)]
            for t in range(each_time):
                yield next(cur_yielder)
        # --
