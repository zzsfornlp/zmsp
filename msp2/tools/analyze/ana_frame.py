#

# analyzer for Frame

from itertools import chain
import pandas as pd
from typing import List
from msp2.data.inst import Sent, ArgLink, Frame, Token, Mention, \
    yield_sent_pairs, yield_sents, yield_frames, MyPrettyPrinter, set_ee_heads
from msp2.data.rw import ReaderGetterConf
from msp2.proc.eval import *
from msp2.proc import ResultRecord
from msp2.utils import OtherHelper, zlog, ConfEntryChoices, AccEvalEntry, F1EvalEntry, zglob
from .analyzer import AnalyzerConf, Analyzer, AnnotationTask

# --
class FrameAnalyzerConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        # files
        self.take_ratio = 1.0
        self.main = ReaderGetterConf()  # as gold files
        self.gold = ""
        self.align_gold_sent = False
        self.fake_gold_frames = False  # put syntax frames to gold
        # eval with preds
        self.extra = ReaderGetterConf()
        self.preds = []  # list of preds
        self.econf: FrameEvalConf = ConfEntryChoices(
            {'frame': FrameEvalConf(), 'fn': MyFNEvalConf(), 'pb': MyPBEvalConf()}, 'frame')
        # self.econf = FrameEvalConf()
        # others
        self.gold_set_ee_heads = False  # try to auto assign heads
        self.pred_set_ee_heads = False  # try to auto assign heads

# not only match for gold/pred, but also for possible multiple preds
class MatchedList(list):
    # some shortcuts
    @property
    def gold(self):
        return self[0]

    @property
    def pred(self):
        return self[1]

    @property
    def preds(self):
        return self[1:]

    def has_g(self):
        return self.gold is not None

    def has_p(self, style='all'):
        _ff = {'all': all, 'any': any}[style]
        return _ff(p is not None for p in self.preds)

    def has_gp(self, style='all'):
        _ff = {'all': all, 'any': any}[style]
        return _ff(p is not None for p in self)

@Analyzer.reg_decorator('frame', conf=FrameAnalyzerConf)
class FrameAnalyzer(Analyzer):
    def __init__(self, conf: FrameAnalyzerConf):
        super().__init__(conf)
        conf: FrameAnalyzerConf = self.conf
        # --
        # read main files
        main_insts = self.take_first_samples(list(conf.main.get_reader(input_path=conf.gold)))
        if conf.gold_set_ee_heads:  # auto heads
            set_ee_heads(main_insts)
        if conf.fake_gold_frames:
            self.fake_syntax_frames(main_insts)
        self.set_var("main", main_insts, explanation="init")
        # --
        eval_conf = conf.econf
        if isinstance(eval_conf, MyFNEvalConf):  # todo(+N): ugly!
            evaler = MyFNEvaler(eval_conf)
        elif isinstance(eval_conf, MyPBEvalConf):
            evaler = MyPBEvaler(eval_conf)
        else:
            evaler = FrameEvaler(eval_conf)
        # --
        self.evaler = evaler
        # --
        i_lists, s_lists, t_lists, f_lists, a_lists = self.read_preds(main_insts)
        self.set_var("il", i_lists, explanation="init")  # inst pair
        self.set_var("sl", s_lists, explanation="init")  # sent pair
        self.set_var("tl", t_lists, explanation="init")  # token pair
        self.set_var("fl", f_lists, explanation="init")  # frame pair
        self.set_var("al", a_lists, explanation="init")  # arg pair

    # --
    # some helpers
    def get_syntax_paths(self, tree, x: int, y: int):
        tree_labs = [z.split(":")[0] for z in tree.seq_label.vals]
        spine0, spine1 = tree.get_path(x, y)
        return (tuple([tree_labs[z] for z in spine0]), tuple([tree_labs[z] for z in spine1]))

    def take_first_samples(self, insts):
        _s = self.conf.take_ratio
        if _s <= 1.0:
            _s = int(len(insts)*_s+0.99)
        else:
            _s = int(_s)
        return insts[:_s]

    # --
    def fake_syntax_frames(self, insts):
        for sent in yield_sents(insts):
            # first delete original ones!
            sent.delete_frames('evt')
            sent.delete_frames('ef')
            # --
            tree = sent.tree_dep
            all_fake_efs = [sent.make_entity_filler(widx=ii, wlen=1, type="UNK") for ii in range(len(sent))]
            for ii in range(len(sent)):
                fake_evt = sent.make_event(widx=ii, wlen=1, type=sent.seq_upos.vals[ii])
                for jj, ef in enumerate(all_fake_efs):
                    role = self.get_syntax_paths(tree, ii, jj)
                    fake_evt.add_arg(ef, role=role)
            # --

    # --
    def read_preds(self, main_insts):
        conf: FrameAnalyzerConf = self.conf
        _safe_get = (lambda x: [] if x is None else x)
        # --
        # read preds: [1+len(pred), *]
        all_insts = [main_insts]
        all_sents = [list(yield_sents(main_insts))]
        all_tokens = [list(t for s in all_sents[-1] for t in s.tokens)]
        # --
        _gold_frame_maps = {id(f): [i, f, [], []] for i, f in
                            enumerate(list(f for s in all_sents[-1] for f in _safe_get(self.evaler._get_frames(s))))}
        _other_frames = []  # no-match ones
        _gold_arg_maps = {id(a): [i, a, [], []] for i, a in
                          enumerate(list(aa for s in all_sents[-1] for f in
                                         self.evaler._get_frames(s) for aa in _safe_get(self.evaler._get_args(f))))}
        _other_args = []
        # --
        all_res = {}
        _pred_files = sum([zglob(z, sort=True) for z in conf.preds], [])
        for one_pidx, one_pred in enumerate(_pred_files):
            one_insts = self.take_first_samples(list(conf.extra.get_reader(input_path=one_pred)))  # get all of them
            if conf.pred_set_ee_heads:  # auto heads
                set_ee_heads(one_insts)
            all_insts.append(one_insts)
            # --
            all_sents.append(list(yield_sents(one_insts)))
            if conf.align_gold_sent:
                for spred, sgold in zip(all_sents[-1], all_sents[0]):
                    spred.gsent = sgold
                    try:
                        sgold.psents.append(spred)
                    except:
                        sgold.psents = [spred]
            all_tokens.append(list(t for s in all_sents[-1] for t in s.tokens))
            # eval
            eres = self.evaler.eval(main_insts, one_insts)
            res0 = ResultRecord(results=eres.get_summary(), description=eres.get_brief_str(), score=float(eres.get_result()))
            zlog(f"#=====\nEval with {conf.main} vs. {one_pred}: res = {eres}\n{eres.get_detailed_str()}")
            all_res[one_pred] = res0
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
        zlog(f"zzzztestfinal: {ResultRecord(results=all_res, score=0.)}")
        # --
        # construct the lists
        i_lists = [MatchedList(z) for z in zip(*all_insts)]
        s_lists = [MatchedList(z) for z in zip(*all_sents)]
        t_lists = [MatchedList(z) for z in zip(*all_tokens)]
        f_lists, a_lists = [], []
        for cur_lists, cur_map, other_ones in \
                ([f_lists, _gold_frame_maps, _other_frames], [a_lists, _gold_arg_maps, _other_args]):
            _tmp_ones = sorted(other_ones + list(cur_map.values()), key=lambda x: x[0])  # sort by gold-hit idx
            for _one in _tmp_ones:
                one_l = MatchedList([_one[1]] + _one[2])  # gold, *preds
                one_l.mps = [None] + _one[3]  # evals matchpairs
                cur_lists.append(one_l)
        # --
        return i_lists, s_lists, t_lists, f_lists, a_lists

    # helpers for breakdown
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

    @classmethod
    def get_ann_type(cls): return FrameAnnotationTask

class FrameAnnotationTask(AnnotationTask):
    def __init__(self, objs: List):
        super().__init__(objs)

    def obj_info(self, obj, **kwargs) -> str:
        # --
        def _obj_info(_obj):
            if _obj is None:
                return "[None]"
            if isinstance(_obj, Sent):
                return MyPrettyPrinter.str_sent(_obj, **kwargs)
            elif isinstance(_obj, Frame):
                return MyPrettyPrinter.str_frame(_obj, **kwargs)
            elif isinstance(_obj, ArgLink):
                return MyPrettyPrinter.str_alink(_obj, **kwargs)
            elif isinstance(_obj, Token):
                return MyPrettyPrinter.str_token(_obj, **kwargs)
            elif isinstance(_obj, Mention):
                return MyPrettyPrinter.str_mention(_obj, **kwargs)
            else:
                return None
        # --
        if isinstance(obj, MatchedPair):
            s1, s2 = _obj_info(obj.gold), _obj_info(obj.pred)
            return f"#GOLD: {s1}\n#PRED: {s2}"
        elif isinstance(obj, (list, tuple)):
            ss = [f"#{i}: {_obj_info(z)}" for i,z in enumerate(obj)]
            return "\n".join(ss)
        else:
            return _obj_info(obj)

    def do_ap(self, sent_evt=1, **kwargs):
        super().do_ap(sent_evt=sent_evt, **kwargs)

# example run
"""
PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.analyze frame gold:? preds:?
PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.analyze frame gold:./fn15_fulltext.dev.json preds:./fn15_out.dev.json
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
# -- special one: lemma mismatch
zz = filter fl "' '.join([t.lemma for t in d.gold.mention.get_tokens()]) != d.gold.info['luName'].split('.')[0]"
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
break_eval al "d.pred.label" "d.gold.label" "d.pred.label == d.gold.label and d.pred.main.label == d.gold.main.label" -3
# --
break_eval al "d.preds[0].label" "d.gold.label" "d.preds[0].label == d.gold.label" -- sort_key:-3
aa=filter al "d.gold is not None and d.gold.label=='A0' and not (d.preds[0] is not None and d.preds[0].label=='A0') and (d.preds[1] is not None and d.preds[1].label=='A0')"
# --
break_eval tl "d.pred.deplab" "d.gold.deplab" "d.pred.deplab == d.gold.deplab and d.pred.head_idx == d.gold.head_idx" -3
# --
break_eval2 al "d.pred.arg.sent.sid-d.pred.main.sent.sid" "d.gold.arg.sent.sid-d.gold.main.sent.sid" "d.pred.label == d.gold.label" -- pint:0
"""
