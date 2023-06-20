#

# analyzer for UD's upos and udep

__all__ = [
    "AnalyzerDparConf", "AnalyzerDpar", "ATaskDpar",
]

from itertools import chain
from typing import List
from mspx.proc.eval import DparEvalConf, DparEvaler
from mspx.utils import zlog
from .analyzer import *

# --
@AnalyzerConf.rd('dpar')
class AnalyzerDparConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        self.econf = DparEvalConf()

@AnalyzerDparConf.conf_rd()
class AnalyzerDpar(Analyzer):
    def __init__(self, conf: AnalyzerDparConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AnalyzerDparConf = self.conf
        # --
        s_lists, t_lists = self.read_data()
        self.set_var("sl", s_lists, explanation="init")  # sent pair
        self.set_var("tl", t_lists, explanation="init")  # token pair
        # --

    def read_data(self):
        all_files, all_insts, all_sents, all_tokens = self.read_basic_data()
        _pred_files = all_files[1:]
        for one_pidx, one_sents in enumerate(all_sents[1:]):
            eres = self.eval.eval(one_sents, all_sents[0])
            res0 = eres.get_res()
            zlog(f"#=====\nEval with {all_files[0]} vs. {_pred_files[one_pidx]}: res = {res0}\n{eres.get_str(False)}")
        # --
        s_lists = [MatchedList(z) for z in zip(*all_sents)]
        t_lists = [MatchedList(z) for z in zip(*all_tokens)]
        return s_lists, t_lists

    @classmethod
    def get_ann_type(cls): return ATaskDpar

    def do_break_eval2(self, insts_target: str, pcode: str, gcode: str,
                       corr_code="d.pred.head_idx==d.gold.head_idx and d.pred.deplab==d.gold.deplab", pint=0, **kwargs):
        super().do_break_eval2(insts_target, pcode, gcode, corr_code=corr_code, pint=pint, **kwargs)

class ATaskDpar(ATask):
    def obj_info(self, obj, **kwargs) -> str:
        from mspx.proc.eval import MatchedPair
        if isinstance(obj, MatchedPair):
            obj = [obj.gold, obj.pred]
        if isinstance(obj, (list, tuple)):
            return self.printer.str_deptree([z.tree_dep for z in obj], **kwargs)  # note: specific one!
        return super().obj_info(obj, **kwargs)

# example run
"""
PYTHONPATH=../src/ python3 -mpdb -m mspx.cli.analyze dpar gold:? preds:?
x = ann_new sl
"""
