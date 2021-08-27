#

# analyzer for UD's upos and udep

from itertools import chain
from typing import List
from msp2.data.inst import Sent, ArgLink, Frame, Token, Mention, \
    yield_sent_pairs, yield_sents, MyPrettyPrinter, set_ee_heads
from msp2.data.rw import ReaderGetterConf
from msp2.proc.eval import DparEvalConf, DparEvaler
from msp2.utils import OtherHelper, zlog, ConfEntryChoices, wrap_color
from .analyzer import AnalyzerConf, Analyzer, AnnotationTask
from .ana_frame import MatchedList

# --
class UDAnalyzerConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        # files
        self.main = ReaderGetterConf()  # as gold files
        self.gold = ""
        # eval with preds
        self.extra = ReaderGetterConf()
        self.preds = []  # list of preds
        self.econf = DparEvalConf().direct_update(deplab_l1=True)  # by default only check l1

@Analyzer.reg_decorator('ud', conf=UDAnalyzerConf)
class UDAnalyzer(Analyzer):
    def __init__(self, conf: UDAnalyzerConf):
        super().__init__(conf)
        conf: UDAnalyzerConf = self.conf
        # --
        # read main files
        main_insts = list(conf.main.get_reader(input_path=conf.gold))
        self.set_var("main", main_insts, explanation="init")
        # eval
        self.evaler = DparEvaler(conf.econf)
        # --
        all_sents = [list(yield_sents(main_insts))]
        all_toks = [[t for s in yield_sents(main_insts) for t in s.tokens]]
        for one_pidx, one_pred in enumerate(conf.preds):
            one_insts = list(conf.extra.get_reader(input_path=one_pred))  # get all of them
            one_sents = list(yield_sents(one_insts))
            assert len(one_sents) == len(all_sents[0])
            # eval
            eres = self.evaler.eval(main_insts, one_insts)
            zlog(f"#=====\nEval with {conf.main} vs. {one_pred}: res = {eres}\n{eres.get_detailed_str()}")
            # --
            all_sents.append(one_sents)
            all_toks.append([t for s in one_sents for t in s.tokens])
        # --
        s_lists = [MatchedList(z) for z in zip(*all_sents)]
        self.set_var("sl", s_lists, explanation="init")  # sent pair
        s_toks = [MatchedList(z) for z in zip(*all_toks)]
        self.set_var("tl", s_toks, explanation="init")  # token pair
        # --

    @classmethod
    def get_ann_type(cls): return UDAnnotationTask

    def do_break_eval2(self, insts_target: str, pcode: str, gcode: str,
                       corr_code="d.pred.head_idx==d.gold.head_idx and d.pred.deplab==d.gold.deplab", pint=0, **kwargs):
        super().do_break_eval2(insts_target, pcode, gcode, corr_code=corr_code, pint=pint, **kwargs)

class UDAnnotationTask(AnnotationTask):
    def __init__(self, objs: List):
        super().__init__(objs)

    def obj_info(self, obj, **kwargs) -> str:
        # --
        if isinstance(obj, Sent):
            obj = MatchedList([obj])
        assert isinstance(obj, MatchedList)
        # --
        def _get_upos_tags(_s):
            return _s.seq_upos.vals if _s.seq_upos is not None else (["_"]*len(_s))
        # --
        # print the errors
        import pandas as pd
        gold_sent = obj.gold
        gold_upos_tags = _get_upos_tags(gold_sent)
        gold_udep_heads = gold_sent.tree_dep.seq_head.vals
        gold_udep_labels = [z.split(":")[0] for z in gold_sent.tree_dep.seq_label.vals]
        all_cols = [gold_upos_tags, gold_udep_heads, gold_udep_labels]
        for pred_sent in obj.preds:
            assert len(pred_sent) == len(gold_sent)
            pred_upos_tags = [wrap_color(t1, bcolor=('red' if t1!=t2 else 'black')) for t1,t2 in zip(_get_upos_tags(pred_sent), gold_upos_tags)]
            pred_udep_heads = [wrap_color(str(t1), bcolor=('red' if t1!=t2 else 'black')) for t1,t2 in zip(pred_sent.tree_dep.seq_head.vals, gold_udep_heads)]
            pred_udep_labels = [wrap_color(t1, bcolor=('red' if t1!=t2 else 'black')) for t1,t2 in zip([z.split(":")[0] for z in pred_sent.tree_dep.seq_label.vals], gold_udep_labels)]
            all_cols.append(["||"] * len(pred_sent))
            all_cols.extend([pred_upos_tags, pred_udep_heads, pred_udep_labels])
        # --
        all_cols.append(gold_sent.seq_word.vals)
        data = [[all_cols[j][i] for j in range(len(all_cols))] for i in range(len(gold_sent))]  # .T
        d = pd.DataFrame(data, index=list(range(1, 1+len(gold_sent))))
        return d.to_string()

# example run
"""
PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.analyze ud gold:? preds:?
x = ann_new sl
"""
