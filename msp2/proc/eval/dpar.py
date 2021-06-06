#

# eval for pos and dpar

__all__ = [
    "DparEvalConf", "DparEvaler", "DparEvalResult",
]

from typing import List, Union
from msp2.data.inst import Doc, Sent, yield_sent_pairs
from msp2.utils import AccEvalEntry
from .base import *
from .helper import *

# =====

class DparEvalConf(EvalConf):
    def __init__(self):
        super().__init__()
        # --
        self.exclude_punct = False  # whether exclued punct?
        self.deplab_l1 = True  # only eval L1?

@Evaluator.reg_decorator("dpar", conf=DparEvalConf)
class DparEvaler(Evaluator):
    def __init__(self, conf: DparEvalConf):
        super().__init__(conf)
        conf: DparEvalConf = self.conf
        # --
        self.current_result = DparEvalResult.zero(conf)

    def get_current_result(self): return self.current_result

    def reset(self): self.current_result = DparEvalResult.zero(self.conf)

    def eval(self, gold_insts: List[Union[Doc, Sent]], pred_insts: List[Union[Doc, Sent]]):
        res = DparEvalResult.zero(self.conf)
        for one_g, one_p in yield_sent_pairs(gold_insts, pred_insts):
            one_res = self._eval_one(one_g, one_p)
            res += one_res
        # save to the overall one!
        self.current_result += res
        return res

    def _eval_one(self, gold_inst: Sent, pred_inst: Sent):
        conf: DparEvalConf = self.conf
        # assert gold_inst.id == pred_inst.id, "Err: SentID mismatch!"
        # assert gold_inst.seq_word.vals == pred_inst.seq_word.vals, "Err: sent text mismatch!"
        # --
        gold_tokens = gold_inst.get_tokens()
        pred_tokens = pred_inst.get_tokens()
        assert len(gold_tokens) == len(pred_tokens)
        if conf.exclude_punct:
            res = DparEvalResult(conf, [(a,b) for a,b in zip(gold_tokens, pred_tokens) if a.upos != "PUNCT"])
        else:
            res = DparEvalResult(conf, [(a,b) for a,b in zip(gold_tokens, pred_tokens)])
        return res

# --
# result
class DparEvalResult(EvalResult):
    def __init__(self, conf: DparEvalConf, token_pairs):
        self.conf = conf
        # make new lists
        self.token_pairs = list(token_pairs)
        # eval
        self.current_results = {_k: AccEvalEntry() for _k in ["pos", "uas", "las"]}
        for tg, tp in self.token_pairs:
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
        ret = DparEvalResult(self.conf, self.token_pairs)
        ret += other
        return ret

    @classmethod
    def zero(cls, conf: DparEvalConf):
        return DparEvalResult(conf, [])

    def get_result(self) -> float:
        return self.current_results["las"].res

    def get_brief_str(self) -> str:
        # one-line brief result (only ACC reported)
        return "/".join([f"{self.current_results[n].res:.4f}" for n in ["pos", "uas", "las"]])

    def get_detailed_str(self) -> str:
        rets = []
        for _key in ["pos", "uas", "las"]:
            rets.append(f"{_key}: {self.current_results[_key]}")
        return "\n".join(rets)

    def get_summary(self) -> dict:
        ret = {}
        for _key in ["pos", "uas", "las"]:
            ret[_key] = self.current_results[_key].details
        return ret

# --
# PYTHONPATH=../src/ python3 -m msp2.cli.evaluate dpar gold.input_path:pb/conll05/dev.conll.ud.json pred.input_path:pb/conll05/dev.conll.ud2.json print_details:1
