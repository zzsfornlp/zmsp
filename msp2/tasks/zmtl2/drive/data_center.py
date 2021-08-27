#

# collection of datasets

__all__ = [
    "DataCenterConf", "DataCenter",
]

from collections import OrderedDict
from typing import List
from copy import deepcopy
import numpy as np
from msp2.utils import Conf, ConfEntryChoices, zlog, Random
from msp2.proc import ScheduledValue
from msp2.data.inst import Sent, Frame, yield_frames, yield_sents
from ..core import ZDatasetConf, ZDataset, ZDataPreprocessor
from .ztables import parse_filename_with_shortcut

# --
_DATA_MAXN=10  # this should be enough!!
class DataCenterConf(Conf):
    def __init__(self):
        # could be multiple ones!!
        for ii in range(_DATA_MAXN):
            for wset in ["train", "dev", "test"]:
                setattr(self, f"{wset}{ii}", ZDatasetConf())
        self.testM = ZDatasetConf()  # shortcut: main-test that can override others!
        # --
        # shortcut data tables

    def get_all_dconfs(self, wset: str):
        return [getattr(self, f"{wset}{ii}") for ii in range(_DATA_MAXN)]

class DataCenter:
    def __init__(self, conf: DataCenterConf, specified_wset=None):
        self.conf = conf
        # --
        # load and prepare them
        self.datasets = OrderedDict()  # note: three-layered naming! (wset, group_name, [sub_name])
        self.train_sample_svs = OrderedDict()  # sv:sample_rate for training
        if specified_wset is None:
            specified_wset = ["train", "dev", "test"]  # by default read all!
        for wset in specified_wset:
            trg = OrderedDict()
            if wset == "test" and len(conf.testM.group_files)>0:
                dconfs = [conf.testM]
            else:
                dconfs = conf.get_all_dconfs(wset)
            for dconf in dconfs:
                # --
                if len(dconf.group_files) <= 0:
                    continue
                # --
                one_group_name = dconf.group_name
                assert one_group_name not in trg
                trg[one_group_name] = []
                # --
                # note: can be list or dict!!
                sub_iter = dconf.group_files.items() if isinstance(dconf.group_files, dict) else [(z,z) for z in dconf.group_files]
                for one_subname, one_filename in sub_iter:
                    copied_dconf: ZDatasetConf = deepcopy(dconf)
                    sc_filename = parse_filename_with_shortcut(one_filename)
                    copied_dconf.input_file = one_filename if sc_filename is None else sc_filename  # assign here!
                    d = ZDataset(copied_dconf, f"{one_group_name}_{one_subname}", wset)
                    trg[one_group_name].append(d)
                # join them?
                if dconf.group_joint:
                    jd = ZDataset.join_datasets(dconf, one_group_name, wset, trg[one_group_name])
                    trg[one_group_name] = [jd]
                # --
                if wset == "train":
                    self.train_sample_svs[one_group_name] = ScheduledValue(f"sr_{one_group_name}", dconf.group_sample_rate)
            self.datasets[wset] = trg
        # --
        zlog(f"Build DataCenter ok: {self}")

    def __repr__(self):
        ss = [f"{k}:{len(v)}" for k,v in self.datasets.items()]
        return f"DataCenter({','.join(ss)})"

    def get_scheduled_values(self):
        return OrderedDict([(f"_sr_{k}",v) for k,v in self.train_sample_svs.items()])

    def get_datasets(self, wset=None, group=None, task=None, extra_filter=None):
        # filter by wset
        if wset is None:
            cand_groups: List[OrderedDict] = list(self.datasets.values())
        else:
            cand_groups: List[OrderedDict] = [self.datasets[wset]]
        # filter by group
        if group is None:
            candidates: List[ZDataset] = sum([z for g in cand_groups for z in g.values()], [])
        else:
            candidates: List[ZDataset] = sum([g[group] for g in cand_groups], [])
        # filter by task
        if task is not None:
            candidates = [z for z in candidates if (task in z.tasks)]
        # filter by filter
        if extra_filter is not None:
            candidates = [z for z in candidates if extra_filter(z)]
        return candidates

    # yield yielder to further yield batches (loop forever)
    def yield_train_yielder(self):
        all_yielders = []
        all_svs = []
        all_inner_rates = []
        for group_name, group_datasets in self.datasets["train"].items():
            all_yielders.append([z.yield_batches() for z in group_datasets])
            one_inner_rates = np.asarray([(len(z.items)**z.conf.group_sample_alpha) for z in group_datasets])
            all_inner_rates.append(one_inner_rates / one_inner_rates.sum())  # inner sample
            all_svs.append(self.train_sample_svs[group_name])
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
            chosen_yielder = all_yielders[cur_gidx][cur_iidx]
            yield chosen_yielder
        # --

# --
# define some preprocessors here!!
@ZDataPreprocessor.reg_decorator("approx_prev_next")
def _approx_prev_next(insts: List):
    if len(insts) > 0 and isinstance(insts[0], Sent):
        for ii in range(len(insts) - 1):
            Sent.assign_prev_next(insts[ii], insts[ii + 1])
    return insts

@ZDataPreprocessor.reg_decorator("pb_delete_noarg")
def _pb_delete_noarg(insts: List):  # delete no args frames!
    for sent in yield_sents(insts):
        for evt in list(sent.events):  # note: remember to copy!!
            if len([a for a in evt.args if a.role not in ["V", "C-V"]]) == 0:
                sent.delete_frame(evt, 'evt')
    return insts

@ZDataPreprocessor.reg_decorator("pb_delete_argv")
def _pb_delete_argv(insts: List):  # delete argv for srl!
    for evt in yield_frames(insts):
        for a in list(evt.args):  # note: remember to copy!
            if a.role in ["V", "C-V"]:
                a.delete_self()
    return insts

@ZDataPreprocessor.reg_decorator("ace_merge_time")
def _ace_merge_time(insts: List):  # merge Time* args
    for evt in yield_frames(insts):
        for a in list(evt.args):  # note: remember to copy!
            if a.role.lower().startswith("time"):
                a.set_label(a.role[:4])
    return insts

@ZDataPreprocessor.reg_decorator("ace_exclude_cts_un")
def _ace_exclude_cts_un(insts: List):  # exclude cts and un, following (Zhang et al., 2019)
    # note: dataset specific!
    rets = []
    for doc in insts:
        if any(str.isupper(c) for c in doc.id):  # for convenience, a quick pattern!
            if doc.id not in ["Austin-Grad-Community_20050212.2454", "Integritas-Group-Community-Forum_20050110.0557"]:
                rets.append(doc)
    return rets

@ZDataPreprocessor.reg_decorator("evt_exclude_empty")
def _evt_exclude_empty(insts: List):  # exclude empty sents!
    rets = []
    for sent in insts:
        if len(sent.events) > 0:
            rets.append(sent)
    return rets
# --

# --
# b msp2/tasks/zmtl2/drive/data_center:122
