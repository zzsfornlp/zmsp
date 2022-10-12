#

# frame evaluator (eval-range is 'Sent')

__all__ = [
    "FrameEvalConf", "FrameEvaler", "FrameEvalResult",
    "MyFNEvalConf", "MyFNEvaler", "MyPBEvalConf", "MyPBEvaler",
]

from typing import List, Union, Callable
import os
from msp2.utils import DivNumber, AccEvalEntry, F1EvalEntry, zlog, default_json_serializer, zwarn
from msp2.data.inst import Doc, Sent, Mention, Frame, ArgLink, yield_sent_pairs, set_ee_heads
from msp2.data.resources.frames import FramePresetHelper
from .base import *
from .helper import *

# =====
# todo(+N): one hard situation to handle is: same_mention+same_type -> multiple frames with diff args

class FrameEvalConf(EvalConf):
    def __init__(self):
        super().__init__()
        # how to get frames
        self.frame_getter = "evt"  # events/entity_fillers/... or eval(this)
        self.skip_gold_empty = False  # skip no gold frame sentences
        # final result
        self.weight_frame = 1.  # avg weight for frame
        self.weight_arg = 1.  # avg weight for arg (after matching frame)
        self.weight_full = 0.  # avg weight for full-frame-exact-match
        # span mode
        self.span_mode_frame = "span"  # span, head, shead
        self.span_mode_arg = "span"  # ...
        # special arg mode
        self.match_arg_with_frame = True  # need the frames to be matched first!
        self.match_arg_with_frame_type = True  # need the frames to match type!
        # matching thresh (need to be >=this!)
        self.posi_thresh_frame = 1.0
        self.posi_thresh_arg = 1.0
        self.match_lab_alpha = 0.25  # label's weight for matching
        # modification to labels?
        self.labf_frame = ""
        self.labf_arg = ""  # or require the triple: "(lambda x: (x.label, x.main.label, x.arg.label))"
        # --
        # special mode!!
        self.fpair_filter = "None"
        self.apair_filter = "None"
        # --
        # breakdowns (function of item to label, note: ignore if return None!)
        self.bd_frame = ""
        self.bd_arg = ""
        # for printing breakdowns
        self.bd_frame_lines = 0  # how many lines to print for breakdown (0 means nope)
        self.bd_arg_lines = 0
        # --

# --
# pre-compiled frame-getter
def _get_evts(s: Sent): return s.events
def _get_efs(s: Sent): return s.entity_fillers

_FRAME_GETTERS = {
    "evt": _get_evts, "event": _get_evts,
    "ef": _get_efs, "entity_fillers": _get_efs,
}
# --

@Evaluator.reg_decorator("frame", conf=FrameEvalConf)
class FrameEvaler(Evaluator):
    def __init__(self, conf: FrameEvalConf):
        super().__init__(conf)
        conf: FrameEvalConf = self.conf
        # --
        # functions
        self.frame_getter = _FRAME_GETTERS.get(conf.frame_getter)
        if self.frame_getter is None:  # shortcut or eval!
            self.frame_getter = eval(conf.frame_getter)
        self.span_getter_frame = Mention.create_span_getter(conf.span_mode_frame)
        self.span_getter_arg = Mention.create_span_getter(conf.span_mode_arg)
        # special
        self.fpair_filter_f = eval(conf.fpair_filter)
        self.apair_filter_f = eval(conf.apair_filter)
        # breakdown analysis
        self.bd_frame = FramePresetHelper(conf.bd_frame) if conf.bd_frame else None
        self.bd_arg = FramePresetHelper(conf.bd_arg) if conf.bd_arg else None
        # --
        self.current_result = FrameEvalResult.zero(conf)
        # lab function
        self.default_labf = (lambda x: x.label.lower())
        self.labf_frame = self.labf_arg = self.default_labf
        if conf.labf_frame:
            self.labf_frame = eval(conf.labf_frame)
        if conf.labf_arg:
            self.labf_arg = eval(conf.labf_arg)
        # --


    def get_current_result(self): return self.result

    def reset(self): self.current_result = FrameEvalResult.zero(self.conf)

    def eval(self, gold_insts: List[Union[Doc, Sent]], pred_insts: List[Union[Doc, Sent]]):
        res = FrameEvalResult.zero(self.conf)
        for one_g, one_p in yield_sent_pairs(gold_insts, pred_insts):
            one_res = self._eval_one(one_g, one_p)
            res += one_res
        # save to the overall one!
        self.current_result += res
        return res

    def _eval_one(self, gold_inst: Sent, pred_inst: Sent):
        conf: FrameEvalConf = self.conf
        assert gold_inst.id == pred_inst.id, "Err: SentID mismatch!"
        # assert gold_inst.seq_word.vals == pred_inst.seq_word.vals, "Err: sent text mismatch!"
        assert len(gold_inst) == len(pred_inst)
        gold_frames, pred_frames = self._get_frames(gold_inst), self._get_frames(pred_inst)
        # --
        if conf.skip_gold_empty and len(gold_frames) == 0:
            return FrameEvalResult.zero(conf)
        # --
        # first match frames
        frame_pairs = self._match_items(gold_frames, pred_frames, self.span_getter_frame, conf.posi_thresh_frame,
                                        labf=self.labf_frame)
        if self.fpair_filter_f is not None:  # special filtering
            frame_pairs = [p for p in frame_pairs if self.fpair_filter_f(p)]
        # then match args & full
        arg_pairs = []
        full_pairs = []
        for fpair in frame_pairs:
            if fpair.is_matched():
                # _lab_arr = float(fpair.gold.label.lower()==fpair.pred.label.lower()) if conf.match_arg_with_frame_type else None
                _lab_arr = float(self.labf_frame(fpair.gold)==self.labf_frame(fpair.pred)) if conf.match_arg_with_frame_type else None
                one_arg_pairs = self._match_items(
                    self._get_args(fpair.gold), self._get_args(fpair.pred), self.span_getter_arg, conf.posi_thresh_arg,
                    lab_arr=_lab_arr, labf=self.labf_arg)  # whether need to count frame type??
            elif fpair.gold is not None:
                one_arg_pairs = [MatchedPair(a, None) for a in self._get_args(fpair.gold)]
            else:
                one_arg_pairs = [MatchedPair(None, a) for a in self._get_args(fpair.pred)]
            arg_pairs.extend(one_arg_pairs)
            if self.apair_filter_f is not None:  # special filtering
                arg_pairs = [p for p in arg_pairs if self.apair_filter_f(p)]
            # copy and calculate (reduce all by prod) for full match!
            fpair2 = fpair.copy()
            fpair2.arg_pairs = one_arg_pairs  # directly set here!
            for _key in fpair2.get_mached_score_keys():  # set the scores as multiplied by all
                # todo(note): also ignore frame type if set the option!
                _score = fpair2.get_matched_score(_key) if conf.match_arg_with_frame_type else fpair2.get_matched_score('posi')
                for _ap in one_arg_pairs:
                    _score *= _ap.get_matched_score(_key)
                fpair2.set_matched_score(_key, _score)
            full_pairs.append(fpair2)
        # or match all args (without first match frame)
        if not conf.match_arg_with_frame:
            # special mode, require frame type matching!
            gold_all_args, pred_all_args = [a for f in gold_frames for a in self._get_args(f)], \
                                           [a for f in pred_frames for a in self._get_args(f)]
            # must match type in this mode!!
            ftype_masks = ItemMatcher.score_items(
                [self.labf_frame(a.main) for a in gold_all_args], [self.labf_frame(a.main) for a in pred_all_args])
            # replace instead
            arg_pairs = self._match_items(
                gold_all_args, pred_all_args, self.span_getter_arg, conf.posi_thresh_arg,
                posi_arr=ftype_masks, labf=self.labf_arg)  # note: especially make this the base one: posi_arr!!
            if self.apair_filter_f is not None:  # special filtering
                arg_pairs = [p for p in arg_pairs if self.apair_filter_f(p)]
        # get results
        self._process_frame_pairs(frame_pairs)
        self._process_arg_pairs(arg_pairs)
        self._process_full_pairs(full_pairs)
        res = FrameEvalResult(conf, frame_pairs, arg_pairs, full_pairs)
        return res

    # matches
    def _match_items(self, items1: List, items2: List, span_getter: Callable, posi_thresh: float,
                     posi_arr=None, lab_arr=None, labf=None):
        conf: FrameEvalConf = self.conf
        lab_alpha = conf.match_lab_alpha
        if labf is None:
            labf = self.default_labf
        # --
        mentions1, mentions2 = [z.mention for z in items1], [z.mention for z in items2]
        scores_posi_arr = ItemMatcher.score_mentions(mentions1, mentions2, span_getter, posi_thresh)  # [size1, size2]
        if posi_arr is not None:  # extra mask
            scores_posi_arr *= posi_arr
        labs1, labs2 = [labf(z) for z in items1], [labf(z) for z in items2]
        scores_lab_arr = ItemMatcher.score_items(labs1, labs2)  # [size1, size2]
        scores_lab_arr *= scores_posi_arr  # multiply by posi
        if lab_arr is not None:
            scores_lab_arr *= lab_arr
        # combine scores for matching purpose
        scores_arr = ((1-lab_alpha) * scores_posi_arr + lab_alpha * scores_lab_arr)  # [size1, size2]
        # todo(+W): we do not want to solve the full assignment problem, but use a simple greedy method
        matched_pairs, unmatched1, unmatched2 = ItemMatcher.match_simple(scores_arr)
        rets = []
        # --
        # note: especially counting as 1. if >0
        scores_posi_arr = (scores_posi_arr>0).astype(float)
        scores_lab_arr = (scores_lab_arr>0).astype(float)
        # --
        for a,b in matched_pairs:
            one_scores = {"posi": scores_posi_arr[a,b], "lab": scores_lab_arr[a,b]}
            pair = MatchedPair(items1[a], items2[b], matched_scores=one_scores)
            rets.append(pair)
        rets.extend([MatchedPair(items1[a], None) for a in unmatched1])
        rets.extend([MatchedPair(None, items2[b]) for b in unmatched2])
        return rets

    # =====
    # for special budget settings
    def _get_frames(self, s: Sent):
        ret = self._do_get_frames(s)
        if self.bd_frame is not None:
            ret = [z for z in ret if self.bd_frame.c(z.label)]
        return ret

    def _get_args(self, frame: Frame):
        ret = self._do_get_args(frame)
        if self.bd_arg is not None:
            ret = [z for z in ret if self.bd_arg.c(z.label)]
        return ret

    # to be overridden
    def _do_get_frames(self, s: Sent): return self.frame_getter(s)
    def _do_get_args(self, frame: Frame): return frame.args
    def _process_frame_pairs(self, pairs: List[MatchedPair]): pass
    def _process_arg_pairs(self, pairs: List[MatchedPair]): pass
    def _process_full_pairs(self, pairs: List[MatchedPair]): pass

    # helper
    def _fake_discontinuous_ef(self, all_mentions: List):
        # todo(+N): for simplicity, special support for it!!
        # simply make the fake ones but no adding!!
        start_widx, end_widx = all_mentions[0].widx, all_mentions[0].wridx
        for m in all_mentions[1:]:
            if m.sent is all_mentions[0].sent:  # expand
                start_widx = min(start_widx, m.widx)
                end_widx = max(end_widx, m.wridx)
        fake_ef = Frame.create(mention=Mention.create(all_mentions[0].sent, start_widx, end_widx - start_widx))
        # --
        excluded_idxes = set(range(start_widx, end_widx))
        for m in all_mentions:
            for i in range(m.widx, m.wridx):
                if i in excluded_idxes:  # there can be repeated mentions
                    excluded_idxes.remove(i)
        fake_ef.mention.excluded_idxes = excluded_idxes  # todo(+N): ugly fix!!
        return fake_ef

class FrameEvalResult(EvalResult):
    def __init__(self, conf: FrameEvalConf, frame_pairs: List[MatchedPair], arg_pairs: List[MatchedPair], full_pairs: List[MatchedPair]):
        self.conf = conf
        # make new lists!
        self.frame_pairs = list(frame_pairs)
        self.arg_pairs = list(arg_pairs)
        self.full_pairs = list(full_pairs)
        # calculate some results
        self.current_results = {}  # key -> (frame, arg, full)
        for _key in ["posi", "lab"]:
            self.current_results[_key] = (
                self._sum_pairs(_key, self.frame_pairs), self._sum_pairs(_key, self.arg_pairs),
                self._sum_pairs(_key, self.full_pairs),
            )
        # cached final results
        self.cached_final_results = None

    # helper
    def _sum_pairs(self, key: str, pairs: List[MatchedPair]):
        res = F1EvalEntry()
        for p in pairs:
            one_g, one_p = p.get_gp_results(key)
            res.combine_dn(one_p, one_g)
        return res

    def __iadd__(self, other: 'FrameEvalResult'):
        # add ones!
        self.frame_pairs.extend(other.frame_pairs)
        self.arg_pairs.extend(other.arg_pairs)
        self.full_pairs.extend(other.full_pairs)
        # add results
        for _key in ["posi", "lab"]:
            for self_gp, other_gp in zip(self.current_results[_key], other.current_results[_key]):
                self_gp.combine(other_gp)
        # clear cache
        self.cached_final_results = None
        # --
        return self

    def __add__(self, other: 'FrameEvalResult'):
        ret = FrameEvalResult(self.conf, self.frame_pairs, self.arg_pairs, self.full_pairs)
        ret += other
        return ret

    @classmethod
    def zero(cls, conf: FrameEvalConf):
        return FrameEvalResult(conf, [], [], [])

    # =====
    # final calculations

    @property
    def final_results(self):
        if self.cached_final_results is None:
            conf: FrameEvalConf = self.conf
            self.cached_final_results = {}
            final_weights = [conf.weight_frame, conf.weight_arg, conf.weight_full]
            for _key in ["posi", "lab"]:
                _entry = F1EvalEntry()
                for one_entry, one_weight in zip(self.current_results[_key], final_weights):
                    _entry.combine(one_entry, scale=one_weight)
                self.cached_final_results[_key] = _entry
        return self.cached_final_results

    def get_result(self) -> float:
        return self.final_results["lab"].res

    def get_brief_str(self) -> str:
        # one-line brief result (only F1 reported)
        rets = []
        final_results = self.final_results
        for _key in ["posi", "lab"]:
            f1s = [e.res for e in self.current_results[_key]]
            f1s.append(final_results[_key].res)
            rets.append(f"{_key}: " + "/".join([f"{v:.4f}" for v in f1s]))
        return " ||| ".join(rets)

    def get_bd_str(self, pairs, lines: int, tag: str):
        df = MatchedPair.breakdown_eval(pairs)
        r_macro, r_macro2, r_micro = MatchedPair.df2avg(df)
        summary = {f'ZBreak-{tag}': {'macro': r_macro, 'macro2': r_macro2, 'micro': r_micro}}
        center_line = "" if len(df) <= lines else f" ... (Truncate {lines}/{len(df)})\n"
        ret = f"# -- Breakdown-{tag}\n{df[:lines].to_string()}\n{center_line}{summary}\n# --"
        return ret

    def get_detailed_str(self) -> str:
        # more detailed results
        rets = []
        final_results = self.final_results
        for _key in ["posi", "lab"]:
            for _e, _n in zip(list(self.current_results[_key])+[final_results[_key]], ['frame', 'arg', 'full', 'final']):
                rets.append(f"{_key}-{_n}: {_e}")
        # --
        # get breakdown results?
        if self.conf.bd_frame_lines > 0:
            rets.append(self.get_bd_str(self.frame_pairs, self.conf.bd_frame_lines, 'Frame'))
        if self.conf.bd_arg_lines > 0:
            rets.append(self.get_bd_str(self.arg_pairs, self.conf.bd_arg_lines, 'Arg'))
        # --
        return "\n".join(rets)

    def get_summary(self) -> dict:
        ret = {}
        final_results = self.final_results
        for _key in ["posi", "lab"]:
            for _e, _n in zip(list(self.current_results[_key])+[final_results[_key]], ['frame', 'arg', 'full', 'final']):
                ret[f"{_key}-{_n}"] = _e.details
        return ret

# =====
# special one for FN
# todo(+N): there are some slight diffs (0.sth%) to the semafor-script: possibly from repeated-span frame/args;
#  => therefore, for reporting, finally need to use the semafor-script!!

class MyFNEvalConf(FrameEvalConf):
    def __init__(self):
        super().__init__()
        # --
        self.frame_file = ""  # where to find frame-file?
        # special weights for core and noncore roles!
        self.weight_core_role = 1.0
        self.weight_noncore_role = 0.5
        self.only_rank1_role = True  # only count rank1 roles!
        self.combine_discontinuous = True  # make a large arg
        self.remove_repeat_frames = True  # remove same-mention frames
        # overwrite the super one!!
        self.skip_gold_empty = True  # by default True

    @classmethod
    def _get_type_hints(cls):
        return {'frame_file': 'zglob1'}

@Evaluator.reg_decorator("fn", conf=MyFNEvalConf)
class MyFNEvaler(FrameEvaler):
    def __init__(self, conf: MyFNEvalConf):
        super().__init__(conf)
        # --
        conf: MyFNEvalConf = self.conf
        # load frame files
        if os.path.exists(conf.frame_file):
            self.coreness_map = MyFNEvaler._read_coreness_from_file(conf.frame_file)
        else:
            self.coreness_map = MyFNEvaler._read_coreness_from_nltk(conf.frame_file)
        # --
        # print info
        num_arg, num_core = 0, 0
        for v in self.coreness_map.values():
            num_arg += len(v)
            num_core += len([1 for v2 in v.values() if v2=="Core"])
        zlog(f"Read coreness from {conf.frame_file}: {len(self.coreness_map)} frames, {num_arg} args, {num_core} core-args.")
        # --

    def _do_get_frames(self, s: Sent):
        conf: MyFNEvalConf = self.conf
        # --
        ret_frames = self.frame_getter(s)
        if conf.remove_repeat_frames:
            frames_map = {}  # (sid, widx, wlen) -> []
            for f in ret_frames:
                _k = (f.mention.sent.sid, f.mention.widx, f.mention.wlen)
                if _k in frames_map:
                    frames_map[_k].append(f)
                else:
                    frames_map[_k] = [f]
            # todo(+N): simply get the last one!!
            # ret_frames = [flist[0] for flist in frames_map.values()]
            ret_frames = [flist[-1] for flist in frames_map.values()]
        return ret_frames

    def _do_get_args(self, frame: Frame):
        conf: MyFNEvalConf = self.conf
        # --
        ret_args = frame.args
        if conf.only_rank1_role:  # ignore other ranks!
            ret_args = [a for a in ret_args if a.info.get("rank", 1) == 1]
        # note: fake special arglink and mention for discontinuous ones
        if conf.combine_discontinuous:
            role_maps = {}
            for a in ret_args:
                if a.role not in role_maps:
                    role_maps[a.role] = [a]
                else:
                    role_maps[a.role].append(a)
            # --
            ret_args = []
            for role, alist in role_maps.items():
                if len(alist) == 1:
                    ret_args.append(alist[0])
                elif role == "Path":  # each piece of a Path FE is treated separately
                    ret_args.extend(alist)
                else:
                    fake_ef = self._fake_discontinuous_ef([a.arg.mention for a in alist])
                    # fake ArgLink only for eval, not actually linked to anyone!!
                    fake_arg = ArgLink.create(frame, fake_ef, role=role, score=sum(a.score for a in alist))
                    ret_args.append(fake_arg)
        return ret_args

    def _process_arg_pairs(self, pairs: List[MatchedPair]):
        weight_core_role, weight_noncore_role = self.conf.weight_core_role, self.conf.weight_noncore_role
        cmap = self.coreness_map
        # --
        def _get_weight(_arg: ArgLink):
            return weight_core_role if cmap.get(_arg.main.type, {}).get(_arg.role, "") == "Core" else weight_noncore_role
        # --
        for p in pairs:
            gold_weight, pred_weight = None, None
            if p.gold is not None:
                gold_weight = _get_weight(p.gold)
            if p.pred is not None:
                pred_weight = _get_weight(p.pred)
            p.set_weights(gold_weight, pred_weight)

    @staticmethod
    def _read_coreness_from_file(file: str):
        frame_map = default_json_serializer.from_file(file)
        cmap = {}  # FrameName -> {RoleName -> CoreType}
        for f, v in frame_map.items():
            assert f not in cmap, f"Err: repeated frame {f}"
            new_map = {}
            for fe in v["FE"]:
                role, core_type = fe["name"], fe["coreType"]
                # assert role not in new_map, f"Err: repeated frame-role {f}:{role}"
                if role in new_map:  # skip this one!
                    zwarn(f"repeated frame-role {f}:{role}")
                else:
                    new_map[role] = core_type
            cmap[f] = new_map
        return cmap

    @staticmethod
    def _read_coreness_from_nltk(which_fn="fn15"):
        which_fn = {"fn15": 15, '': 15, 'fn17': 17}.get(which_fn, None)
        if which_fn is None:
            zwarn("Cannot read coreness, simply let it be EMPTY!!")
            return {}
        if which_fn == 15:
            from nltk.corpus import framenet15 as nltk_fn
        else:
            from nltk.corpus import framenet as nltk_fn
        # --
        cmap = {}  # FrameName -> {RoleName -> CoreType}
        for frame in nltk_fn.frames():
            cmap[frame.name] = {k:v.coreType for k,v in frame.FE.items()}
        return cmap

# =====
# special one for PB

class MyPBEvalConf(FrameEvalConf):
    def __init__(self):
        super().__init__()
        # --
        # overwrite the super one!
        self.weight_frame = 0.
        self.match_arg_with_frame_type = False  # by default, no need for frame type
        self.no_join_c = False  # disable joining c-* together as in conll05 eval!

@Evaluator.reg_decorator("pb", conf=MyPBEvalConf)
class MyPBEvaler(FrameEvaler):
    def __init__(self, conf: MyPBEvalConf):
        super().__init__(conf)
        # --

    def _do_get_frames(self, s: Sent):
        # remove repeated ones and keep wlen==1
        hit_frames = [None] * len(s)
        for f in self.frame_getter(s):
            # todo(+N): here only output len==1
            if f.mention.wlen != 1: continue
            one_widx = f.mention.widx
            hit_frames[one_widx] = f
        return [f for f in hit_frames if f is not None]

    def _do_get_args(self, frame: Frame):
        _no_join_c = self.conf.no_join_c
        # note: here we don't make it a seq, but do combine A? & C-A?
        last_role_maps = {}
        arg_groups = []
        for a in sorted(frame.args, key=lambda x: x.mention.get_span()):
            role = a.role
            if role in ["V", "C-V"]:
                continue  # ignore "V" args!!
            if not _no_join_c and role.startswith("C-"):
                actual_role = role[2:]
                if actual_role not in last_role_maps:
                    last_role_maps[actual_role] = [a]
                else:
                    last_role_maps[actual_role].append(a)
            else:
                if role in last_role_maps:  # save previous one
                    arg_groups.append((role, last_role_maps[role]))
                # make new one!!
                last_role_maps[role] = [a]
        arg_groups.extend([(k,v) for k,v in last_role_maps.items()])
        # --
        ret_args = []
        for role, alist in arg_groups:
            if len(alist) == 1:
                ret_args.append(alist[0])
            else:
                fake_ef = self._fake_discontinuous_ef([a.arg.mention for a in alist])
                # fake ArgLink only for eval, not actually linked to anyone!!
                fake_arg = ArgLink.create(frame, fake_ef, role=role, score=sum(a.score for a in alist))
                ret_args.append(fake_arg)
        return ret_args

# --
# PYTHONPATH=../src/ python3 -m msp2.cli.evaluate fn gold.input_path:fn15_fulltext.dev.json pred.input_path:fn15_out.dev.json print_details:1
# for dd in run_pb12c_mb0b*; do for domain in nw bc bn mz pt tc wb; do PYTHONPATH=../src/ python3 -m msp2.cli.evaluate pb gold.input_path:../pb/conll12c/en.${domain}.test.conll.ud.json pred.input_path:${dd}/_zoutG.en.${domain}.test.conll.ud.json print_details:1 "fpair_filter:lambda x: x.gold.type not in ['be.03','become.03','do.01','have.01']" |& tee ${dd}/_logGE2.en.${domain}.test.conll.ud.json; done done
