#

# analyzer for Frame

__all__ = [
    "AnalyzerFrameConf", "AnalyzerFrame", "ATaskFrame",
]

from itertools import chain
import pandas as pd
import pprint
from typing import List
from mspx.data.inst import ArgLink, MyPrettyPrinter
from mspx.proc.eval import *
from mspx.utils import zlog, ConfEntryCallback
from .analyzer import *

# --
@AnalyzerConf.rd('frame')
class AnalyzerFrameConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        self.econf: FrameEvalConf = ConfEntryCallback((lambda s: self.callback_entry(s, T=EvalConf)), default_s='frame')

@AnalyzerFrameConf.conf_rd()
class AnalyzerFrame(Analyzer):
    def __init__(self, conf: AnalyzerFrameConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AnalyzerFrameConf = self.conf
        # --
        i_lists, s_lists, t_lists, f_lists, a_lists = self.read_data()
        self.set_var("il", i_lists, explanation="init")  # inst pair
        self.set_var("sl", s_lists, explanation="init")  # sent pair
        self.set_var("tl", t_lists, explanation="init")  # token pair
        self.set_var("fl", f_lists, explanation="init")  # frame pair
        self.set_var("al", a_lists, explanation="init")  # arg pair
        # --

    @staticmethod
    def read_frame_data(all_files, all_insts, evaler):
        _safe_get = (lambda x: [] if x is None else x)
        # --
        _gold_frame_maps = {id(f): [i, f, [], []] for i, f in enumerate(
            (f for inst in all_insts[0] for f in _safe_get(evaler._get_frames(inst))))}
        _other_frames = []  # no-match ones
        _gold_arg_maps = {id(a): [i, a, [], []] for i, a in enumerate(
            (aa for inst in all_insts[0] for f in evaler._get_frames(inst) for aa in _safe_get(evaler._get_args(f))))}
        _other_args = []
        # --
        # all_res = {}
        _pred_files = all_files[1:]
        for one_pidx, one_insts in enumerate(all_insts[1:]):
            eres = evaler.eval(one_insts, all_insts[0])
            res0 = eres.get_res()
            zlog(f"#=====\nEval with {all_files[0]} vs. {_pred_files[one_pidx]}: res = {res0}\n{eres.get_str(False)}")
            # all_res[_pred_files[one_pidx]] = res0
            # --
            for cur_pairs, cur_map, other_ones in \
                    ([eres.full_pairs, _gold_frame_maps, _other_frames], [eres.arg_pairs, _gold_arg_maps, _other_args]):
                last_hit_gold_idx = 0  # only for sorting order!
                for _pair in cur_pairs:
                    if _pair.gold is None or id(_pair.gold) not in cur_map:  # no hit or no hit for join-c case!
                        # todo(+W): it could be better if we can align preds
                        _preds, _res = [None] * len(_pred_files), [None] * len(_pred_files)
                        _preds[one_pidx], _res[one_pidx] = _pair.pred, _pair
                        other_ones.append([last_hit_gold_idx, None, _preds, _res])
                    else:
                        _res = cur_map[id(_pair.gold)]  # should be there!!
                        last_hit_gold_idx = _res[0]
                        _res[-2].append(_pair.pred)
                        _res[-1].append(_pair)
        # --
        f_lists, a_lists = [], []
        for cur_lists, cur_map, other_ones in \
                ([f_lists, _gold_frame_maps, _other_frames], [a_lists, _gold_arg_maps, _other_args]):
            _tmp_ones = sorted(other_ones + list(cur_map.values()), key=lambda x: x[0])  # sort by gold-hit idx
            for _one in _tmp_ones:
                one_l = MatchedList([_one[1]] + _one[2])  # gold, *preds
                one_l.mps = [None] + _one[3]  # evals matchpairs
                cur_lists.append(one_l)
        return f_lists, a_lists

    def read_data(self):
        all_files, all_insts, all_sents, all_tokens = self.read_basic_data()
        i_lists = [MatchedList(z) for z in zip(*all_insts)]
        s_lists = [MatchedList(z) for z in zip(*all_sents)]
        t_lists = [MatchedList(z) for z in zip(*all_tokens)]
        f_lists, a_lists = self.read_frame_data(all_files, all_insts, self.eval)
        return i_lists, s_lists, t_lists, f_lists, a_lists

    # --
    # some helpers
    def get_syntax_paths(self, tree, x: int, y: int):
        tree_labs = [z.split(":")[0] for z in tree.seq_label.vals]
        spine0, spine1 = tree.get_path(x, y)
        return (tuple([tree_labs[z] for z in spine0]), tuple([tree_labs[z] for z in spine1]))

    def sdist(self, arg: ArgLink, _abs=True, _min=0, _max=10, _buckets=(1,2,4,7,10)):  # surface distance
        z = arg.main.mention.shead_widx - arg.arg.mention.shead_widx
        if _abs:
            z = abs(z)
        z = min(_max, max(z, _min))
        if _buckets is not None:
            for ii, vv in enumerate(_buckets):
                if z <= vv: return ii
            return len(_buckets)
        else:
            return z

    def tdist(self, arg: ArgLink, _min=0, _max=4, _sum=True):  # tree distance
        tree = arg.main.sent.tree_dep
        labs0, labs1 = self.get_syntax_paths(tree, arg.main.mention.shead_widx, arg.arg.mention.shead_widx)
        a, b = len(labs0), len(labs1)
        if _sum:
            z = a+b
            z = min(_max, max(z, _min))
            return z
        else:
            a = min(_max, max(a, _min))
            b = min(_max, max(b, _min))
            return a,b

    def tpath(self, arg: ArgLink, external_tree=None):
        tree = arg.main.sent.tree_dep if external_tree is None else external_tree
        labs0, labs1 = self.get_syntax_paths(tree, arg.main.mention.shead_widx, arg.arg.mention.shead_widx)
        return labs0, labs1
    # --

    def get_cons_table(self, key=None):
        from mspx.tasks.zrel.cons_table import CONS_SET
        if key is None:
            return CONS_SET
        else:
            return CONS_SET[key]
        # --
        # fg al "(d[0].main.label, d[0].label, d[0].arg.label) not in self.get_cons_table('evt')" "(d[0].main.label, d[0].label, d[0].arg.label)"
        # fg al "(d[0].main.label, d[0].label, d[0].arg.label) not in self.get_cons_table('rel')" "(d[0].label, d[0].main.label, d[0].arg.label)"
        # --

    @classmethod
    def get_ann_type(cls): return ATaskFrame

class ATaskFrame(ATask):
    def __init__(self, objs: List, conf):
        super().__init__(objs, conf)
        self.printer = MyPrettyPrinter(sent_frame_cates=([conf.econf.frame_cate] if conf.econf.frame_cate else []))
        # --

# example run
"""
PYTHONPATH=../src/ python3 -m pdb -m mspx.cli.analyze frame gold:? preds:?
# example commands
# - see target extractions
# -- extra ones
zp = filter fl "d.gold is None"
# -- missing ones
zg = filter fl "d.pred is None"
# -- specific ones
zz = filter fl "d.gold is None and d.pred.mention.text=='South'"
zz = filter fl "d.pred is None and d.gold.mention.text=='South'"
x=ann_new zz
aj
# -- general stats
group fl "(d.gold.mention.wlen, )"
# --
# some specific ones
eval "print(self.cur_ann_task.cur_obj.gold.info)"
args = eval "list(a for f in vs.fl for a in f.gold.args)"
group args "min(d.mention.wlen, 30)," -- sum_key:id
group args "(lambda z: sum(z.label==d2.label for d2 in z.par.args))(d),"
eval "(lambda ff,zs: print('\n'.join([ff(z, sent_evt=1) for z in zs])))(self.cur_ann_task.obj_info, self.cur_ann_task.objs[:4])"
# --
# examples of breakdowns
break_eval fl "d.pred.label" "d.gold.label" -- sort_key:-3
break_eval al "d.pred.label" "d.gold.label" -- sort_key:-3
break_eval2 al "d.pred.arg.sent.sid-d.pred.main.sent.sid" "d.gold.arg.sent.sid-d.gold.main.sent.sid" "d.pred.label == d.gold.label" -- pint:0
break_eval2 al "''.join(d.pred.get_spath())" "''.join(d.gold.get_spath())" "d.pred.label == d.gold.label" -- pint:0 sort_key:-3
break_eval2 al "min(5,len(d.pred.get_spath()))" "min(5,len(d.gold.get_spath()))" "d.pred.label == d.gold.label" -- pint:0 sort_key:-3
# --
group fl '(lambda x: sum(len(x.mention.overlap_tokens(z.mention))>0 for z in x.sent.get_frames()))(d[0])'
group al "''.join(d.gold.get_spath())"
"""
