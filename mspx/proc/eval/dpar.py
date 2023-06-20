#

# eval for pos and dpar

__all__ = [
    "DparEvalConf", "DparEvaler", "DparEvalResult",
]

from typing import List, Union
from mspx.data.inst import Doc, Sent, yield_sent_pairs
from mspx.utils import AccEvalEntry, ZResult
from .base import *
from .helper import *

# =====

@EvalConf.rd('dpar')
class DparEvalConf(EvalConf):
    def __init__(self):
        super().__init__()
        # --
        self.exclude_punct = False  # whether exclued punct?
        self.exclude_punct_set = ['PUNCT', 'PU', '.', '``', "''", ':', ',']  # UD, CTB, PTB
        self.deplab_l1 = False  # only eval L1?
        # mix for final result
        self.weight_pos = 0.
        self.weight_uas = 0.
        self.weight_las = 1.

@DparEvalConf.conf_rd()
class DparEvaler(Evaluator):
    def __init__(self, conf: DparEvalConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: DparEvalConf = self.conf
        self.curr_er = DparEvalResult(conf)
        # --
        self.exclude_punct_set = set(conf.exclude_punct_set)

    def get_er(self): return self.curr_er
    def reset_er(self): self.curr_er = DparEvalResult(self.conf)

    def eval(self, pred_insts: List[Union[Doc, Sent]], gold_insts: List[Union[Doc, Sent]]):
        res = DparEvalResult(self.conf)
        for one_p, one_g in yield_sent_pairs(pred_insts, gold_insts):
            one_res = self._eval_one(one_p, one_g)
            res += one_res
        # save to the overall one!
        self.curr_er += res
        return res

    def _eval_one(self, pred_inst: Sent, gold_inst: Sent):
        conf: DparEvalConf = self.conf
        # --
        pred_tokens = pred_inst.get_tokens()
        gold_tokens = gold_inst.get_tokens()
        assert len(pred_tokens) == len(gold_tokens)
        if conf.exclude_punct:
            res = DparEvalResult(conf, [MatchedPair(a,b) for a,b in zip(pred_tokens, gold_tokens)
                                        if a.upos not in self.exclude_punct_set])
        else:
            res = DparEvalResult(conf, [MatchedPair(a,b) for a,b in zip(pred_tokens, gold_tokens)])
        return res

# --
# record
class DparEvalResult(EvalRecord):
    def __init__(self, conf: DparEvalConf, token_pairs=None):
        self.conf = conf
        # make new lists
        self.token_pairs = list(token_pairs) if token_pairs is not None else []
        # eval
        self.current_results = {_k: AccEvalEntry() for _k in ["pos", "uas", "las"]}
        for mp in self.token_pairs:
            tp, tg = mp.pred, mp.gold
            pos_corr, uas_corr = int(tp.upos==tg.upos), int(tp.head_idx==tg.head_idx)
            if conf.deplab_l1:
                las_corr = uas_corr * (int(str(tp.deplab).split(":")[0]==str(tg.deplab).split(":")[0]))
            else:
                las_corr = uas_corr * (int(tp.deplab==tg.deplab))
            self.current_results["pos"].record(pos_corr)
            self.current_results["uas"].record(uas_corr)
            self.current_results["las"].record(las_corr)
        # --

    def __iadd__(self, other: 'DparEvalResult'):
        # add them
        self.token_pairs.extend(other.token_pairs)
        # add results
        for _key in ["pos", "uas", "las"]:
            self.current_results[_key].combine(other.current_results[_key])
        # --
        return self

    def __add__(self, other: 'DparEvalResult'):
        ret = self.copy()
        ret += other
        return ret

    def copy(self):
        return DparEvalResult(self.conf, self.token_pairs)

    def _get_final_res(self):
        conf: DparEvalConf = self.conf
        final_weights = [conf.weight_pos, conf.weight_uas, conf.weight_las]
        res = sum([a*self.current_results[b].res for a,b in zip(final_weights, ["pos", "uas", "las"])])
        return res

    def get_res(self):
        from copy import deepcopy
        res = ZResult(deepcopy(self.current_results), res=self._get_final_res(), des=self.get_str(brief=True))
        return res

    def get_str(self, brief: bool):
        if brief:
            # one-line brief result (only ACC reported)
            return "/".join([f"{self.current_results[n].res:.4f}" for n in ["pos", "uas", "las"]]) \
                   + f"|[{self._get_final_res():.4f}]"
        else:
            rets = []
            for _key in ["pos", "uas", "las"]:
                rets.append(f"{_key}: {self.current_results[_key]}")
            rets.append(f"final: {self._get_final_res():.4f}")
            return "\n".join(rets)
