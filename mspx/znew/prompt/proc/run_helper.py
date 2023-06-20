#

__all__ = [
    "OptimizerConf", "get_optim",
    "TrainingProgressRecord", "SVConf", "ScheduledValue",
    "CheckpointManager",
]

from typing import List
from collections import Counter, OrderedDict
import numpy as np
from mspx.utils import Conf, Registrable, Serializable, ZResult, InfoField, Constants, zlog, Conf, ZHelper, zwarn

# --
# optimizer

class OptimizerConf(Conf):
    def __init__(self):
        self.optim_type = "adamw"
        self.adam_betas = (0.9, 0.98)  # for adam/adamw
        self.init_lrate = 2e-5  # init lrate
        self.optim_kwargs = {}  # other ones
        # --

def get_optim(conf: OptimizerConf, model):
    import torch
    # --
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # --
    # note: simply use the default ones for others if not specified
    _optim = conf.optim_type
    _kwargs = conf.optim_kwargs
    if _optim == 'adamw':
        optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=conf.init_lrate, betas=conf.adam_betas, **_kwargs)
    elif _optim == 'adam':
        optim = torch.optim.Adam(optimizer_grouped_parameters, lr=conf.init_lrate, betas=conf.adam_betas, **_kwargs)
    else:
        raise NotImplementedError(f"UNK optimizer of {_optim}")
    return optim

# --
# training process

# Record of the training process
@Registrable.rd('tpr')
class TrainingProgressRecord(Serializable):
    def __init__(self):
        # idxes (counts: how many has past up to now!)
        self.eidx = 0  # epochs: increase end-of-epoch
        self.iidx = 0  # insts: increase after feeding
        self.uidx = 0  # updates: increase after updating
        self.cidx = 0  # checkpoints: increase end-of-validate(check)
        self.aidx = 0  # anneal: increase if anneal with bad validate-result
        # idx counter (from which dataset?)
        self.iidx_counter = Counter()
        self.uidx_counter = Counter()
        # checkpoints: len==self.cidx
        self.chp_names: List[str] = []
        self.train_results: List[ZResult] = []
        self.dev_results: List[ZResult] = []
        # all with dev results
        # track best point
        self.overall_best_result = ZResult()  # overall best, regardless of 'record_best' or not
        self.overall_best_point = -1
        self.best_result = ZResult()  # recorded best
        self.best_point = -1  # recorded best point
        # deal with anneal with all bests
        self.bad_counter = 0  # current bad counter
        self.bad_points = []  # all bad points
        self.anneal_points = []  # all anneal points

    @classmethod
    def _info_fields(cls):
        return {
            'iidx_counter': InfoField(from_f=lambda v: Counter(v)),
            'uidx_counter': InfoField(from_f=lambda v: Counter(v)),
            'train_results': InfoField(inner_type=ZResult, wrapper_type=list),
            'dev_results': InfoField(inner_type=ZResult, wrapper_type=list),
        }

    def current_suffix(self, brief_uidx=True):
        # 4 digits should be enough
        if brief_uidx:
            sname = f".c{self.cidx:03d}-e{self.eidx}-u{self.uidx//1000}k"
        else:
            sname = f".c{self.cidx:03d}-e{self.eidx}-u{self.uidx}"
        return sname

    def info_best(self):
        if self.best_point < 0:
            return [-1, "None", ZResult()]
        else:
            return [self.best_point, self.chp_names[self.best_point], self.best_result]

    def info_overall_best(self):
        if self.overall_best_point < 0:
            return [-1, "None", ZResult()]
        else:
            return [self.overall_best_point, self.chp_names[self.overall_best_point], self.overall_best_result]

    # simple updates for plain idxes
    def update_eidx(self, d: int):
        self.eidx += d

    def update_iidx(self, d: int, dname: str = None):
        self.iidx += d
        self.iidx_counter[dname] += d

    def update_uidx(self, d: int, dname: str = None):
        self.uidx += d
        self.uidx_counter[dname] += d

    # special update at checkpoint: no_bad means no recording bad, patience is used for anneal
    def update_checkpoint(self, train_result: ZResult, dev_result: ZResult,
                          no_bad=False, record_best=True, patience=Constants.INT_PRAC_MAX):
        sname = self.current_suffix()
        train_result = ZResult() if train_result is None else train_result
        dev_result = ZResult() if dev_result is None else dev_result
        # ----
        if_overall_best = if_best = if_anneal = False
        # --
        # for overall best
        if float(dev_result) > float(self.overall_best_result):
            self.overall_best_result = dev_result
            self.overall_best_point = self.cidx
            if_overall_best = True
        # --
        if float(dev_result) > float(self.best_result):
            if record_best:
                self.bad_counter = 0  # clear bad counter!
                self.best_result = dev_result
                self.best_point = self.cidx
                if_best = True
        else:
            # [cur-name, best-score-idx, best-score-name, best-score]
            cur_info = [sname, self.best_point, self.chp_names[self.best_point], str(self.best_result)] \
                if self.best_point>0 else [sname, -1, "None", "-inf"]
            if no_bad:
                zlog(f"Bad point (not recorded), now bad/anneal is {self.bad_counter}/{self.aidx}.", func="report")
            else:
                self.bad_counter += 1
                self.bad_points.append(cur_info)
                zlog(f"Bad point ++, now bad/anneal is {self.bad_counter}/{self.aidx}.", func="report")
                if self.bad_counter >= patience:
                    # clear bad count!
                    self.bad_counter = 0
                    # anneal ++
                    if_anneal = True
                    self.anneal_points.append(cur_info)
                    # there can be chances of continuing when restarts enough (min training settings)
                    self.aidx += 1
                    zlog(f"Anneal++, now {self.aidx}.", func="report")
        # record others
        self.chp_names.append(sname)
        self.train_results.append(train_result)
        self.dev_results.append(dev_result)
        self.cidx += 1
        return if_overall_best, if_best, if_anneal

# =====
# scheduled values

class SVConf(Conf):
    def __init__(self):
        self.val = 0.  # basic value
        # how to schedule the value
        self.which_idx = "cidx"  # count steps on which: aidx, eidx, iidx, uidx
        self.val_range = [0., 1.]  # [min, max]
        self.ff = "1."  # i as 'idx': lambda i: ...
        # --
        # transform on idx
        self.idx_bias = 0
        self.idx_scale = 1.0
        self.idx_int = False  # ensure transformed idx is int?
        # --

# note: SV should be stateless, that is, its value can be decided by obj at one step!
class ScheduledValue:
    def __init__(self, conf: SVConf, name: str = None):
        self.conf = conf
        self.name = name
        self._bv = conf.val  # base val
        self._minv, self._maxv = conf.val_range
        self.cur_val: float = None
        _ff = conf.ff.strip()
        self.changeable = "i" in _ff  # involving "i"
        self.ff = ZHelper.eval_ff(_ff, 'i')
        # -- init
        self._set(0)
        zlog(f"Init scheduled value {self.name} as {self.cur_val} (changeable={self.changeable}).")

    @property
    def value(self): return self.cur_val
    def __repr__(self): return "SV-%s=%s" % (self.name, self.cur_val)
    def __float__(self): return float(self.cur_val)
    def __int__(self): return int(self.cur_val)

    def transform_idx(self, idx: int):
        _conf = self.conf
        v = max(0, idx - _conf.idx_bias) / _conf.idx_scale
        if _conf.idx_int:
            v = int(v)
        return v

    def _set(self, the_idx: int):
        _conf = self.conf
        # --
        new_idx = self.transform_idx(the_idx)
        vv = self.ff(new_idx)  # ff
        vv = min(max(self._minv, vv), self._maxv)  # clamp
        vv = self._bv * vv  # base val
        # --
        old_val = self.cur_val
        self.cur_val = vv
        return old_val, self.cur_val

    # adjust at checkpoint
    def adjust_at_ckp(self, sname: str, obj: object, extra_info: str = ""):
        the_idx = getattr(obj, self.conf.which_idx)
        old_val, new_val = self._set(the_idx)
        if self.cur_val != old_val:
            zlog(f"Change scheduled value {self.name}({extra_info}) at {sname}: {old_val} => {self.cur_val}.")
        else:
            zlog(f"Keep scheduled value {self.name}({extra_info}) at {sname} as {self.cur_val}.")

# --
class CheckpointManager:
    def __init__(self, save_bestn: int):
        self.n = save_bestn
        self.bestn_states = []  # List[(name, score, state_dict)]
        # --

    def __repr__(self):
        return f"{[z[:2] for z in self.bestn_states]}"

    def add(self, name: str, score: float, model):
        if self.n <= 0:
            return
        if len(self.bestn_states) < self.n or score > self.bestn_states[-1][1]:
            zlog(f"Add checkpoint {name}[{score}].")
            _s = OrderedDict([(k, v.to('cpu').clone()) for k, v in model.tosave_state_dict().items()])
            self.bestn_states.append((name, score, _s))
            self.bestn_states.sort(key=(lambda x: x[1]), reverse=True)  # highest first
            self.bestn_states = self.bestn_states[:self.n]  # store only bestn!
        else:
            zlog(f"No adding worse checkpoint: {name}[{score}]")
        zlog(f"Current nbest checkpoints: {self}")

    def average_model(self):
        import torch
        if len(self.bestn_states) <= 0:
            zlog(f"No states available, skip: {self}!")
            return
        ds = [z[2] for z in self.bestn_states]
        avg_d = OrderedDict()
        for d in ds:
            for k, v in d.items():
                if k not in avg_d:
                    avg_d[k] = [v]
                else:
                    avg_d[k].append(v)
        for k in list(avg_d.keys()):
            vs = avg_d[k]
            if len(vs) < len(ds):
                zwarn(f"When average_model, key is not full: {len(vs)}")
            avg_d[k] = torch.stack(vs, 0).mean(0)  # simply average!
        return avg_d
