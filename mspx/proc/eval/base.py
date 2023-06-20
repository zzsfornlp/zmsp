#

# base Evaluator
# -- the core of evaluation may be simply matching between gold and pred

__all__ = [
    "EvalConf", "Evaluator", "EvalRecord",
]

from typing import Iterable
from mspx.utils import Conf, Registrable, Configurable, Serializable, ZResult

# --
@Registrable.rd('EV')
class EvalConf(Conf):
    @classmethod
    def get_base_conf_type(cls): return EvalConf
    @classmethod
    def get_base_node_type(cls): return Evaluator

@Registrable.rd('_EV')
class Evaluator(Configurable):
    def __init__(self, conf: EvalConf, **kwargs):
        super().__init__(conf, **kwargs)

    def eval(self, pred_insts: Iterable, gold_insts: Iterable): raise NotImplementedError()
    def get_er(self): raise NotImplementedError()  # accumulated (current) EvalRecord
    def reset_er(self): raise NotImplementedError()

# --
# current stored results: stat + matched_pairs
class EvalRecord:
    def get_res(self) -> ZResult: raise NotImplementedError()
    def get_str(self, brief: bool) -> str: raise NotImplementedError()

    @property
    def result(self): return self.get_res().result
    def __repr__(self): return f"{self.__class__.__name__}: {self.get_str(brief=True)}"
