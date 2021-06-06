#

# base Evaluator
# -- the core of evaluation may be simply matching between gold and pred

__all__ = [
    "EvalConf", "Evaluator", "EvalResult",
]

from typing import List
from msp2.utils import Conf, Registrable

# --
class EvalConf(Conf):
    def __init__(self):
        pass

# --
class Evaluator(Registrable):
    def __init__(self, conf: EvalConf):
        self.conf = conf

    def eval(self, gold_insts: List, pred_insts: List): raise NotImplementedError()
    def get_current_result(self): raise NotImplementedError()  # accumulated (current) result

# --
class EvalResult:
    def get_result(self) -> float: raise NotImplementedError()
    def get_brief_str(self) -> str: raise NotImplementedError()
    def get_detailed_str(self) -> str: raise NotImplementedError()
    def get_summary(self) -> dict: raise NotImplementedError()
    def __float__(self): return float(self.get_result())
    def __repr__(self): return f"{self.__class__.__name__}: {self.get_brief_str()}"
