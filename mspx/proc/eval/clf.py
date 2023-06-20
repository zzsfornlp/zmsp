#

# evaler for simple classification

__all__ = [
    "ClfEvalConf", "ClfEvaler", "ClfEvalResult",
]

from typing import List, Union
from mspx.data.inst import Doc, Sent, yield_pairs, yield_sent_pairs, get_label_gs
from mspx.utils import AccEvalEntry, ZResult
from .base import *
from .helper import *

@EvalConf.rd('clf')
class ClfEvalConf(EvalConf):
    def __init__(self):
        super().__init__()
        self.trg_f = '_info:label'
        self.on_sent = True  # yield sents
        self.do_regr = False

    def make_trg_f(self):
        return get_label_gs(self.trg_f)[0]  # getter!

@ClfEvalConf.conf_rd()
class ClfEvaler(Evaluator):
    def __init__(self, conf: ClfEvalConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ClfEvalConf = self.conf
        self.trg_f = conf.make_trg_f()
        self.yield_f = yield_sent_pairs if conf.on_sent else yield_pairs
        self.curr_er = ClfEvalResult(conf)
        # --

    def get_er(self): return self.curr_er
    def reset_er(self): self.curr_er = ClfEvalResult(self.conf)

    def eval(self, pred_insts: List[Union[Doc, Sent]], gold_insts: List[Union[Doc, Sent]]):
        res = ClfEvalResult(self.conf)
        for one_p, one_g in self.yield_f(pred_insts, gold_insts):
            one_res = self._eval_one(one_p, one_g)
            res += one_res
        # save to the overall one!
        self.curr_er += res
        return res

    def _eval_one(self, pred_inst, gold_inst):
        conf: ClfEvalConf = self.conf
        _do_regr = conf.do_regr
        # --
        lab_p, lab_g = self.trg_f(pred_inst), self.trg_f(gold_inst)
        mp = MatchedPair(pred_inst, gold_inst,
                         {'lab': (lab_p - lab_g)**2 if _do_regr else float(lab_p==lab_g)})  # acc or mse
        res = ClfEvalResult(conf, [mp])
        return res

class ClfEvalResult(EvalRecord):
    def __init__(self, conf: ClfEvalConf, item_pairs=None):
        self.conf = conf
        self.item_pairs = list(item_pairs) if item_pairs is not None else []
        self.acc = AccEvalEntry()
        for mp in self.item_pairs:
            s = mp.get_matched_score('lab')
            self.acc.record(s)
        # --

    def __iadd__(self, other: 'ClfEvalResult'):
        # add them
        self.item_pairs.extend(other.item_pairs)
        self.acc.combine(other.acc)
        # --
        return self

    def __add__(self, other: 'ClfEvalResult'):
        ret = self.copy()
        ret += other
        return ret

    def copy(self):
        return ClfEvalResult(self.conf, self.item_pairs)

    def get_res(self):
        from copy import deepcopy
        res = ZResult({'acc': deepcopy(self.acc)}, res=float(self.acc), des=self.get_str(brief=True))
        return res

    def get_str(self, brief: bool):
        if brief:
            # one-line brief result (only ACC reported)
            return f"{float(self.acc)}"
        else:
            return f"acc: {self.acc}"
