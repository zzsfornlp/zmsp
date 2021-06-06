#

# analyze srl results

from typing import List, Tuple
import pandas as pd
from msp2.data.inst import Sent, ArgLink, Frame, Token, Mention, HeadFinder, MyPrettyPrinter
from msp2.tools.analyze import FrameAnalyzerConf, FrameAnalyzer, Analyzer, FrameAnnotationTask
from msp2.utils import zlog, F1EvalEntry, ZObject

# --
class SRLAnalyzerConf(FrameAnalyzerConf):
    def __init__(self):
        super().__init__()
        # --
        self.pre_load = ""
        self.err_map = "simple"
        self.load_vocab = ""

@Analyzer.reg_decorator('mysrl', conf=SRLAnalyzerConf)
class SRLAnalyzer(FrameAnalyzer):
    def __init__(self, conf: SRLAnalyzerConf):
        super().__init__(conf)
        conf: SRLAnalyzerConf = self.conf
        self.err_map = ErrDetail.ERR_MAPS[conf.err_map]
        # --
        if conf.pre_load:
            self.do_load(conf.pre_load)
        else:
            # further analyze the arguments
            num_pred = len(conf.preds)
            f_lists = self.get_var("fl")  # get frame matches
            f_all_correct_list = []
            f_some_wrong_list = []
            for fl in f_lists:
                gold_frame = fl.gold
                pred_frames = fl.preds
                assert len(pred_frames) == num_pred
                # get them all
                self._process_args(gold_frame)
                err_infos = []
                for pf in pred_frames:
                    self._process_args(pf)  # sort args
                    einfo = ErrInfo.create(gold_frame, pf)
                    err_infos.append(einfo)
                fl.err_infos = err_infos
                # --
                if all(e.fully_correct() for e in err_infos):
                    f_all_correct_list.append(fl)
                else:
                    f_some_wrong_list.append(fl)
            self.set_var("fl1", f_all_correct_list, explanation="init")  # eval pair
            self.set_var("fl0", f_some_wrong_list, explanation="init")  # eval pair
            zlog(f"All frames = {len(f_lists)}, all_corr = {len(f_all_correct_list)}({len(f_all_correct_list)/max(1, len(f_lists))})")
            # --
            # breakdowns for all
            for pi in range(num_pred):
                one_err_infos = [e for fl in f_lists for e in fl.err_infos[pi].rps]
                self.set_var(f"eip{pi}", one_err_infos)
                # self.do_group(f"eip{pi}", "d.get_signature('etype', 'etype2', emap=self.err_map)")
                self.do_group(f"eip{pi}", "d.get_signature('etype', emap=self.err_map)")
                self.do_group(f"eip{pi}", "d.get_signature('etype2')")
                # group eip0 "d.get_signature('etype2')"
                # fg eip0 "d.get_signature('explain')!='_'" "d.get_signature('explain')"
            # --
            # get ps objects
            # self.set_var("dps100", self._get_dpath_objects(f_lists, 100))
        # --
        # load vocab
        if conf.load_vocab:
            from msp2.utils import default_pickle_serializer
            self.vocabs, _ = default_pickle_serializer.from_file(conf.load_vocab)
        # --

    # --
    def _process_args(self, f: Frame):
        if f is not None:
            f.args.sort(key=lambda a: a.mention.get_span() + (a.role, ))
            f.args = [a for a in f.args if a.role not in ["V", "C-V"]]  # note: ignore these!!
    # --

    @classmethod
    def get_ann_type(cls): return SRLAnnotationTask

    # special for pairwise comparing
    def get_pair_epattern(self, fl, i0=0, i1=1):
        einfo0 = fl.err_infos[i0]
        einfo1 = fl.err_infos[i1]
        # --
        n0, n1 = einfo0.get_num_errors(), einfo1.get_num_errors()
        if n0 == n1:
            if n0 == 0:
                return ("equal_good", 0)
            else:
                return ("equal_bad", n0)
        elif n0 < n1:  # i0 better
            return ("better0", n0-n1)
        else:
            return ("better1", n0-n1)
        # --

    # get difficulty of a frame
    def get_difficulty(self, frame: Frame, criterion: str):
        # --
        # helpers
        def _sd(f0, f1):  # surface distance
            _s0, _l0 = f0.mention.get_span()
            _s1, _l1 = f1.mention.get_span()
            _e0, _e1 = _s0+_l0, _s1+_l1
            if _s1 >= _e0:
                return _s1 - _e0 + 1
            elif _e1 <= _s0:
                return _s0 - _e1 + 1
            else:  # overlap!
                return 0
        def _td(f0, f1):  # heads' tree distance
            _h0, _h1 = f0.mention.shead_widx, f1.mention.shead_widx
            a, b = frame.sent.tree_dep.get_path(_h0, _h1)
            return len(a)+len(b)
        # --
        if criterion == "slen":  # sentence length
            ret = len(frame.sent)
        elif criterion == "narg":  # number of args
            ret = len(frame.args)
        elif criterion == "arange":  # arg's range
            _left, _tmp_len = frame.mention.get_span()
            _right = _left + _tmp_len
            for a in frame.args:
                _sa, _la = a.mention.get_span()
                _left = min(_left, _sa)
                _right = max(_right, _sa+_la)
            ret = _right - _left
        elif criterion == "sum_sd":  # sum of surface distance to args
            return sum(_sd(frame, a) for a in frame.args)
        elif criterion == "avg_sd":  # average surface distance to args
            dists = [_sd(frame, a) for a in frame.args]
            return sum(dists)/len(dists) if len(dists)>0 else 0
        elif criterion == "sum_td":  # sum of (dep)-tree distance to args
            return sum(_td(frame, a) for a in frame.args)
        elif criterion == "avg_td":  # average (dep)-tree distance to args
            dists = [_td(frame, a) for a in frame.args]
            return sum(dists)/len(dists) if len(dists)>0 else 0
        else:
            raise NotImplementedError()
        return ret

    # breakdown & eval!
    # --
    # examples:
    # self.breakdown_and_eval('slen', [10,20,30,40,50,100])
    # self.breakdown_and_eval('narg', [1,2,3,4,100])
    # self.breakdown_and_eval('arange', [5,10,15,20,25,30,1000])
    # self.breakdown_and_eval('sum_sd', [2,4,6,8,10,1000])
    # self.breakdown_and_eval('avg_sd', [1,2,3,4,5,100])
    # self.breakdown_and_eval('sum_td', [1,2,3,4,5,100])
    # self.breakdown_and_eval('avg_td', )
    # --
    def breakdown_and_eval(self, dif_cri: str, bins: List[int]=None):
        NSYS = len(self.conf.preds)
        # --
        all_fl = self.get_var("fl")  # get all of them
        all_values = [self.get_difficulty(one_item.gold, dif_cri) for one_item in all_fl]
        if bins is None:  # auto range
            bins = list(range(int(min(all_values)), int(max(all_values) + 1)))
        # grouping
        buckets = [[] for _ in bins]  # <= each one
        values = [[] for _ in bins]  # all values
        for one_item, one_v in zip(all_fl, all_values):
            for ii, bb in enumerate(bins):
                if one_v <= bb:  # add to the first bin that <=
                    buckets[ii].append(one_item)
                    values[ii].append(one_v)
                    break
        # --
        # evaluating
        # --
        def _eval(_fl, _sidx):
            if len(_fl) == 0:
                return 0.
            _entry = F1EvalEntry()
            for _one in _fl:
                M, P, R = _one.err_infos[_sidx].get_mpr()
                _entry.record_p(M, P)
                _entry.record_r(M, R)
            return _entry.res
        # --
        _data = {"bins": bins, "counts": [len(z) for z in buckets], "values": [(sum(z)/len(z) if len(z)>0 else 0.) for z in values]}
        for sys_idx in range(NSYS):
            _data[f"f1_{sys_idx}"] = [_eval(z, sys_idx) for z in buckets]
            # minus sys0
            if sys_idx>0:
                _data[f"D{sys_idx}"] = [a-b for a,b in zip(_data[f"f1_0"], _data[f"f1_{sys_idx}"])]
        _df = pd.DataFrame(_data)
        return _df

    # --
    # for checking of dep-paths to predicate
    def _get_dpaths(self, f, max_depth: int):
        slen = len(f.sent)
        widx = f.mention.shead_widx
        one_paths = []
        for ii in range(slen):
            _paths = f.sent.tree_dep.get_path(ii, widx)  # P0, P1
            _plens = (min(max_depth, len(_paths[0])), min(max_depth, len(_paths[1])))  # L0, L1
            one_paths.append(_plens)
        return one_paths

    def _get_dpath_objects(self, fl, max_depth: int):
        ret = []
        for f in fl:
            if any(z is None for z in f):
                continue  # filter out the unmatched ones!
            # --
            # first get has args from gold!!
            gold_slen = len(f.gold.sent)
            gold_widx = f.gold.mention.shead_widx
            gold_has_args = [False] * gold_slen
            for arg in f.gold.args:
                arg_widx, arg_wlen = arg.mention.get_span()
                gold_has_args[arg_widx:arg_widx+arg_wlen] = [True] * arg_wlen
            # then get each one's dep-path
            all_paths = []
            for one_frame in f:
                assert len(one_frame.sent)==gold_slen and one_frame.mention.shead_widx==gold_widx
                one_paths = self._get_dpaths(one_frame, max_depth)
                all_paths.append(one_paths)
            # --
            for ii in range(gold_slen):
                ret.append(ZObject(has_arg=gold_has_args[ii], ps=[z[ii] for z in all_paths]))
        return ret
    # --

# --
# err info for one pair of frame
class ErrDetail:
    # --
    # some maps
    ERR_MAPS = {
        "nope": {},
        "simple": {"span_": "span", "span+": "label", "merge": "attach", "split": "attach", "miss~": "others", "over~": "others"},
    }
    # --

    def __init__(self, etype: str, etype2):
        self.etype = etype  # auto etype
        self.etype2 = etype2
        self.explain = "_"  # manual explain

    def correct(self):
        return self.etype == "corr!"

    def __repr__(self):
        return str(self.etype)

    def set_explain(self, explain: str):
        if explain != self.etype:
            assert not self.correct()  # no explain for correct one!
            self.explain = explain

    def get_explain(self):
        return self.explain

    def has_explain(self):
        return self.correct() or self.explain != "_"

    def get_signature(self, prop: str, prop2=None, emap=None, emap2=None):
        if emap is None:
            emap = {}
        if emap2 is None:
            emap2 = {}
        v = getattr(self, prop)
        ret = ("Y" if self.correct() else "N", emap.get(v, v), )
        if prop2 is not None:
            v2 = getattr(self, prop2)
            ret += (emap2.get(v2, v2), )
        return ret

class ErrInfo:
    def __init__(self, gold: Frame, pred: Frame, rs: List, ps: List):
        self.gold = gold
        self.pred = pred
        # err types
        self.rs = rs  # for Recall
        self.ps = ps  # for Precision
        # self.groups = groups  # List[(#gold, #pred)]  # most regular (1-0, 0-1, 1-1, 1-many, many-1)

    @property
    def rps(self):
        return self.rs + self.ps

    def get_num_errors(self):  # number of total errors
        return sum(int(not z.correct()) for z in self.rs) + sum(int(not z.correct()) for z in self.ps)

    def get_mpr(self):
        M = sum(z.correct() for z in self.ps)
        assert M == sum(z.correct() for z in self.rs)
        return M, len(self.ps), len(self.rs)  # num_match, num_p, num_r

    def get_f1(self):
        from msp2.utils import MathHelper
        _div = MathHelper.safe_div
        # --
        P = _div(sum(z.correct() for z in self.ps), len(self.ps))
        R = _div(sum(z.correct() for z in self.rs), len(self.rs))
        F = _div(2*P*R, P+R)
        return F

    def fully_correct(self):
        return all(z.correct() for z in self.rs) and all(z.correct() for z in self.ps)

    def has_full_explains(self):
        return all(z.has_explain() for z in self.rs) and all(z.has_explain() for z in self.ps)

    def set_explains(self, ss: str):
        a, b = ss.split("|||")
        e1s = [z for z in a.split() if z!=""]
        e2s = [z for z in b.split() if z!=""]
        assert len(e1s) == len(self.rs) and len(e2s) == len(self.ps)
        for e, z in zip(e1s, self.rs):
            z.set_explain(e)
        for e, z in zip(e2s, self.ps):
            z.set_explain(e)
        # --

    def get_explains(self):
        return [z.get_explain() for z in self.rs] + [z.get_explain() for z in self.ps]

    def __repr__(self):
        r0 = f"{' '.join([str(z) for z in self.rs])} ||| {' '.join([str(z) for z in self.ps])}"  # etype
        r1 = f"{' '.join([z.etype2 for z in self.rs])} ||| {' '.join([z.etype2 for z in self.ps])}"  # etype
        r2 = f"{[z.get_explain() for z in self.rs]} {[z.get_explain() for z in self.ps]}"
        return " ||| ".join([r0,r1,r2])

    @staticmethod
    def _iou(span1, span2):
        start1, len1 = span1
        start2, len2 = span2
        overlap = min(start1 + len1, start2 + len2) - max(start1, start2)
        overlap = max(0, overlap)  # overlapped tokens
        score = overlap / (len1 + len2 - overlap)  # using Jaccard Index
        return score

    @staticmethod
    def create(gold: Frame, pred: Frame):
        rs, ps, group_counts = [], [], []
        if gold is None and pred is None:
            pass
        elif gold is None:  # over-predicting frame
            ps = [ErrDetail("fover", "FErr")] * len(pred.args)
        elif pred is None:  # over-predicting frame
            rs = [ErrDetail("fmiss", "FErr")] * len(gold.args)
        else:
            frame_is_correct = (gold.type == pred.type)
            # --
            hf = HeadFinder("NOUN")
            sent = gold.sent
            dep_heads = sent.tree_dep.seq_head.vals
            # --
            g_args, p_args = gold.args, pred.args  # assume already sorted by position!
            g_matches, p_matches = [-1] * len(g_args), [-1] * len(p_args)
            # find matches by head word
            g_positions, p_positions = [], []  # (widx, wlen, hidx)
            for _args, _posis in zip([g_args, p_args], [g_positions, p_positions]):
                for _a in _args:
                    _widx, _wlen = _a.mention.get_span()
                    _hidx_set = set(hf.get_mindepth_list(sent, _widx, _wlen))  # prediction can have a set!
                    _posis.append((_widx, _wlen, _hidx_set))
            # simply greedy
            matched_pairs = []
            for gidx in range(len(g_args)):
                best_pidx, best_score = None, -1
                for pidx in range(len(p_args)):
                    if p_matches[pidx] < 0:  # not matched before
                        if len(g_positions[gidx][-1].intersection(p_positions[pidx][-1])):  # head match
                            cur_score = ErrInfo._iou(g_positions[gidx][:2], p_positions[pidx][:2])  # iou of full spans
                            if g_args[gidx].role == p_args[pidx].role:
                                cur_score += 1.
                            if cur_score >= best_score:  # prefer the right one!
                                best_pidx = pidx
                                best_score = cur_score
                if best_pidx is not None:
                    matched_pairs.append((gidx, best_pidx))
                    g_matches[gidx] = best_pidx
                    p_matches[best_pidx] = gidx
            # --
            def _get_label_err2(_a0, _a1=None):
                if _a1 is None:
                    _a1 = _a0
                if (_a0.startswith("AM") or _a0.startswith("ARGM")) and (_a1.startswith("AM") or _a1.startswith("ARGM")):
                    return "AM"  # both ARGM
                else:  # guess maybe frame-specific error!
                    return "AC" if frame_is_correct else "FR"
            # --
            # then go for each one!
            rs, ps = [], []
            for cur_args, cur_trgs, cur_e0 in zip([g_args, p_args], [rs, ps], ["miss~", "over~"]):
                for aa in cur_args:
                    cur_trgs.append((cur_e0, _get_label_err2(aa.role)))
            # --
            # matched ones
            for gidx, pidx in matched_pairs:
                r1, r2 = g_args[gidx].role, p_args[pidx].role
                role_match = (r1 == r2)
                if g_positions[gidx] == p_positions[pidx]:
                    _etype = ("corr!", "_") if role_match else ("label", _get_label_err2(r1, r2))
                else:
                    _etype = ("span_", "SYN") if role_match else ("span+", _get_label_err2(r1, r2))
                rs[gidx] = _etype
                ps[pidx] = _etype
            # --
            # split
            for gidx in range(len(g_args)):
                if g_matches[gidx] >= 0: continue  # only for unmatched ones!
                ghidx_set = g_positions[gidx][-1]
                for gidx2, pidx2 in matched_pairs:  # towards a matched pair
                    p_widx, p_wlen, p_hidx_set = p_positions[pidx2]
                    hit = True
                    for ghidx in ghidx_set:
                        if not(ghidx>=p_widx and ghidx<p_widx+p_wlen and
                               any(dep_heads[ghidx]==dep_heads[p_hidx] for p_hidx in p_hidx_set)):  # sib
                            hit = False
                            break
                    if hit:
                        _etype = ("split", "SYN")
                        rs[gidx] = _etype
            # --
            # merge
            for pidx in range(len(p_args)):
                if p_matches[pidx] >= 0: continue  # only for unmatched ones!
                phidx_set = p_positions[pidx][-1]
                for gidx2, pidx2 in matched_pairs:  # towards a matched pair
                    g_widx, g_wlen, _ = g_positions[gidx2]
                    p_widx, p_wlen, _ = p_positions[pidx2]
                    hit = True
                    for phidx in phidx_set:  # inside gold and be descendants
                        if not (phidx>=g_widx and phidx<g_widx+g_wlen and
                                dep_heads[phidx]-1>=p_widx and dep_heads[phidx]-1<p_widx+p_wlen):  # offset by 1
                            hit = False
                            break
                    if hit:
                        _etype = ("merge", "SYN")
                        ps[pidx] = _etype
            # --
            rs = [ErrDetail(z1, z2) for z1,z2 in rs]
            ps = [ErrDetail(z1, z2) for z1,z2 in ps]
        return ErrInfo(gold, pred, rs, ps)

# --
class SRLAnnotationTask(FrameAnnotationTask):
    def __init__(self, fl: List):
        super().__init__(fl)

    def obj_info(self, obj, **kwargs) -> str:
        # --
        def _obj_info(_obj):
            return MyPrettyPrinter.str_frame(_obj, **kwargs)
        # --
        ss = [f"#GOLD: {_obj_info(obj.gold)}"]
        for pi, p in enumerate(obj.preds):
            ss.append(f"#SYS{pi}: {_obj_info(p)}")
        ss.append("#--")
        err_infos = getattr(obj, "err_infos", None)
        if err_infos is not None:
            for pi, e in enumerate(err_infos):
                ss.append(f"#ERR{pi}[{e.has_full_explains()}]: {e}")
        # --
        return "\n".join(ss)

    # add explain
    def do_ae(self, pi: int, ss: str):
        pi = int(pi)
        einfo = self.cur_obj.err_infos[pi]
        old_explains = einfo.get_explains()
        einfo.set_explains(ss)
        new_explains = einfo.get_explains()
        zlog(f"Change #{pi} from {old_explains} to {new_explains}")

# --
if __name__ == '__main__':
    import sys
    from msp2.cli.analyze import main
    main('mysrl', *sys.argv[1:])

# --
"""
# etypes
-- corr!(~): fully correct
-- span_(span): only span error, head matched
-- label(~): span correct but label error
-- span+(label): span error (but head matched), plus label error
-- merge(attach): a pred span can be merged to another matched span
-- split(attach): a matched span can be split one part out to be a good span
-- miss~(others): miss gold span
-- over~(others): over-prdict pred span
# maybe further etypes?
-- syn: requires syn structure (mostly about phrase-attachments)
-- am: subtle AM roles differ (including over and miss)
-- sem: related with core roles
-- trg: wrong targets (including over and miss)
# require what?: what knowledge will be required if needing to make it correct?
-- syn/simple(sim): mostly simple attachment problem
-- syn/complex(com): more complex patterns or constructions
-- sem/frame(fr): predicate/frame meaning
-- sem/arg_core(ac): core ARG problems
-- sem/arg_mod(am): subtle ARGM problems
-- sem/lexicon(lex): rare/unusual words or meanings (on args)
-- -/others(oo): ambiguous/close cases (like coref), or possible annotation problems
"""

# --
# examples
"""
PYTHONPATH=../src/ python3 -m pdb -m msp2.scripts.srl_pb.ana.ana_srl econf:pb gold:../pb/conll05/dev.conll.ud.json preds:../zfinalsv1/run_pb05_b0_head_r1/_zoutG.dev.conll.ud.json
# see groups; note: mostly normal ones: 1-0,0-1,1-1,1-N,N-1
# see etypes
zz = join fl "d.err_infos[0].rs + d.err_infos[0].ps"
fg zz "d.etype!='corr!'" "d.etype"
# --
# annotate the details
PYTHONPATH=../../src/ python3 -m pdb -m msp2.scripts.srl_pb.ana.ana_srl econf:pb gold:c5/gold.json preds:c5/headb0.json,c5/seqb0.json,c5/headb1.json,c5/seqb1.json
PYTHONPATH=../../src/ python3 -m pdb -m msp2.scripts.srl_pb.ana.ana_srl econf:pb gold:r1/gold.json preds:r1/headb0.json,r1/seqb0.json,r1/headb1.json,r1/seqb1.json
zz = sort fl0 "np.random.random()"
x = ann_new zz
# preds: pre_load:??
ae 0 "corr! ||| corr! am"
# --
# filter ann
z2 = filter zz "all(z.has_full_explains() for z in d.err_infos)"
# =====
# compare two
PYTHONPATH=../../src/ python3 -m pdb ana_srl.py econf:pb gold_set_ee_heads:1 gold:../../pb/conll05/dev.conll.ud.json preds:_zrun_tune1125_bnl2/run_tune1125_bnl2_1/zout.json.dev0,_zrun_tune1125_bnl2/run_tune1125_bnl2_5/zout.json.dev0
group fl "self.get_pair_epattern(d)"
group fl "(len(d[0].sent)//10 if d[0] is not None else -1, self.get_pair_epattern(d)[0],)"
group fl "d.err_infos[0].get_num_errors() - d.err_infos[1].get_num_errors()" -- sum_key:id
group fl "int(10*(d.err_infos[0].get_f1() - d.err_infos[1].get_f1()))" -- sum_key:id
# --
corr fl "d.err_infos[0].get_num_errors() - d.err_infos[1].get_num_errors()" "self.vocabs['evt'].get_else_unk(d[0].label)"
corr fl "d.err_infos[0].get_num_errors() - d.err_infos[1].get_num_errors()" "d[0].sent.tree_dep.depths[d[0].mention.shead_widx]"
corr fl "d.err_infos[0].get_num_errors() - d.err_infos[1].get_num_errors()" "len(d[0].sent.tree_dep.chs_lists[1+d[0].mention.shead_widx])"
"""
