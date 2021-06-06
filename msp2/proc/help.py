#

# some helper classes

from collections import Counter
from typing import Dict, List, Callable
import math
from msp2.utils import Constants, zwarn, JsonSerializable, zlog, Conf

# One record of result; use a special RES_KEY to indicate the result field for comparison
class ResultRecord(JsonSerializable):
    RES_KEY = "zres"

    def __init__(self, results: Dict=None, description=None, score: float=None):
        self.results = results if results is not None else {}
        self.description = description
        if score is not None:
            if ResultRecord.RES_KEY in results:
                zwarn(f"RES_KEY already exists, rewrite it: {results[ResultRecord.RES_KEY]} -> {score}")
            results[ResultRecord.RES_KEY] = score

    def __getitem__(self, item):
        return self.results[item]

    @property
    def score(self):
        return self.results.get(ResultRecord.RES_KEY, Constants.REAL_PRAC_MIN)  # the larger the better

    # return the one float score, better eval get larger scores
    def __float__(self):
        return float(self.score)

    def __repr__(self):
        return f"Result({self.score:.4f}): {self.results if self.description is None else self.description}"

    # printing purpose
    def to_str(self, **kwargs):
        return str(self)

    # singleton NIL
    _NIL = None
    @classmethod
    def get_nil(cls):
        if cls._NIL is None:
            cls._NIL = cls()
        return cls._NIL

# Record of the training process
class TrainingProgressRecord(JsonSerializable):
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
        self.train_records: List[ResultRecord] = []
        self.dev_records: List[ResultRecord] = []
        # all with dev results
        # track best point
        self.overall_best_record = ResultRecord.get_nil()  # overall best, regardless of 'record_best' or not
        self.overall_best_point = -1
        self.best_record = ResultRecord.get_nil()  # recorded best
        self.best_point = -1  # recorded best point
        # deal with anneal with all bests
        self.bad_counter = 0  # current bad counter
        self.bad_points = []  # all bad points
        self.anneal_points = []  # all anneal points

    def from_json(self, data: Dict):
        super().from_json(data)
        self.train_records = [ResultRecord.cls_from_json(z) for z in self.train_records]
        self.dev_records = [ResultRecord.cls_from_json(z) for z in self.dev_records]
        self.iidx_counter = Counter(self.iidx_counter)
        self.uidx_counter = Counter(self.uidx_counter)

    def to_json(self) -> Dict:
        d = super().to_json()
        d["train_records"] = [z.to_json() for z in self.train_records]
        d["dev_records"] = [z.to_json() for z in self.dev_records]
        return d

    def current_suffix(self, brief_uidx=True):
        # 4 digits should be enough
        if brief_uidx:
            sname = f".c{self.cidx:03d}-e{self.eidx}-u{self.uidx//1000}k"
        else:
            sname = f".c{self.cidx:03d}-e{self.eidx}-u{self.uidx}"
        return sname

    def info_best(self):
        if self.best_point < 0:
            return [-1, "None", "-inf"]
        else:
            return [self.best_point, self.chp_names[self.best_point], str(self.best_record)]

    def info_overall_best(self):
        if self.overall_best_point < 0:
            return [-1, "None", "-inf"]
        else:
            return [self.overall_best_point, self.chp_names[self.overall_best_point], str(self.overall_best_record)]

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
    def update_checkpoint(self, train_result: ResultRecord, dev_result: ResultRecord,
                          no_bad=False, record_best=True, patience=Constants.INT_PRAC_MAX):
        sname = self.current_suffix()
        train_result = ResultRecord.get_nil() if train_result is None else train_result
        dev_result = ResultRecord.get_nil() if dev_result is None else dev_result
        # ----
        if_overall_best = if_best = if_anneal = False
        # --
        # for overall best
        if float(dev_result) > float(self.overall_best_record):
            self.overall_best_record = dev_result
            self.overall_best_point = self.cidx
            if_overall_best = True
        # --
        if float(dev_result) > float(self.best_record):
            if record_best:
                self.bad_counter = 0  # clear bad counter!
                self.best_record = dev_result
                self.best_point = self.cidx
                if_best = True
        else:
            # [cur-name, best-score-idx, best-score-name, best-score]
            cur_info = [sname, self.best_point, self.chp_names[self.best_point], str(self.best_record)] \
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
        self.train_records.append(ResultRecord.get_nil() if train_result is None else train_result)
        self.dev_records.append(ResultRecord.get_nil() if dev_result is None else dev_result)
        self.cidx += 1
        return if_overall_best, if_best, if_anneal

# =====
# scheduled values

class SVConf(Conf):
    def __init__(self):
        self.val = 0.  # basic value
        # how to schedule the value
        self.which_idx = "eidx"  # count steps on which: aidx, eidx, iidx, uidx
        self.mode = "none"  # none, linear, exp, isigm, ...
        # -----
        # transform on idx
        self.before_bias_none = False  # before bias, act as None mode!!
        self.idx_bias = 0
        self.idx_scale = 1.0
        self.idx_int = False  # ensure transformed idx is int?
        # other parameters
        self.min_val = Constants.REAL_PRAC_MIN
        self.max_val = Constants.REAL_PRAC_MAX
        self.b = 0.
        self.k = 1.0
        self.m = 1.0  # specific one

# todo(note): SV should be stateless, that is, its value can be decided by obj at one step!
class ScheduledValue:
    def __init__(self, name: str, sv_conf: SVConf, special_ff: Callable = None):
        self.name = name
        self.sv_conf = sv_conf
        self.val = sv_conf.val
        self.cur_val = None
        # -----
        mode = sv_conf.mode
        k = sv_conf.k
        b = sv_conf.b
        m = sv_conf.m
        # --
        if special_ff is not None:
            assert mode == "none", "Confusing setting for schedule function!"
        # --
        self.changeable = True
        if mode == "linear":
            self._ff = lambda idx: b+k*m*idx
        elif mode == "poly":
            self._ff = lambda idx: b+k*(idx**m)
        elif mode == "root":  # slightly different semantics!
            self._ff = lambda idx: k * ((b**m + idx * (1-b**m)) ** (1/m))
        elif mode == "exp":
            self._ff = lambda idx: b+k*(math.pow(m, idx))
        elif mode == "isigm":
            self._ff = lambda idx: b+k*(m/(m+math.exp(idx/m)))
        elif mode == "div":
            self._ff = lambda idx: b+k*(1/(1.+m*idx))
        elif mode == "none":
            if special_ff is not None:
                self._ff = lambda idx: b+k*special_ff(idx)  # self-defined schedule: lambda idx: return ...
            else:
                self._ff = lambda idx: 1.0
                self.changeable = False  # no need to add as scheduled value
        else:
            raise NotImplementedError(mode)
        # init setting
        self._set(0)
        zlog(f"Init scheduled value {self.name} as {self.cur_val} (changeable={self.changeable}).")

    @property
    def value(self):
        return self.cur_val

    def __float__(self):
        return float(self.cur_val)

    def __int__(self):
        return int(self.cur_val)

    def transform_idx(self, idx: int):
        _conf = self.sv_conf
        x = max(0, idx - _conf.idx_bias)
        v = x/_conf.idx_scale
        if _conf.idx_int:
            v = int(v)
        return v

    def __repr__(self):
        return "SV-%s=%s" % (self.name, self.cur_val)

    def _set(self, the_idx: int):
        _conf = self.sv_conf
        # --
        if _conf.before_bias_none and the_idx < _conf.idx_bias:  # special mode!!
            vv = self.val
        else:
            new_idx = self.transform_idx(the_idx)
            old_val = self.cur_val
            vv = self.val * self._ff(new_idx)
        # todo(note): self.val as the basis, multiplied by the factor
        self.cur_val = max(self.sv_conf.min_val, vv)
        self.cur_val = min(self.sv_conf.max_val, self.cur_val)
        return old_val, self.cur_val

    # adjust at checkpoint
    def adjust_at_ckp(self, sname: str, obj: object, extra_info: str = ""):
        the_idx = getattr(obj, self.sv_conf.which_idx)
        old_val, new_val = self._set(the_idx)
        if self.cur_val != old_val:
            zlog(f"Change scheduled value {self.name}({extra_info}) at {sname}: {old_val} => {self.cur_val}.")
        else:
            zlog(f"Keep scheduled value {self.name}({extra_info}) at {sname} as {self.cur_val}.")

# =====
# special scheduled values: lrate
