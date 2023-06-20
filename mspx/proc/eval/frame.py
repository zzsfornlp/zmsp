#

# frame evaluator (sentence-based!)
# -- useful for various ie tasks

__all__ = [
    "FrameEvalConf", "FrameEvaler", "FrameEvalRecord",
    "MyFNEvalConf", "MyFNEvaler", "MyPBEvalConf", "MyPBEvaler",
]

import os
from typing import List, Union, Callable
from mspx.data.inst import Doc, Sent, Mention, Frame, ArgLink, yield_sent_pairs
from mspx.utils import F1EvalEntry, ZResult, zwarn, default_json_serializer, zglob1, zlog
from .base import *
from .helper import *

# --

@EvalConf.rd('frame')
class FrameEvalConf(EvalConf):
    def __init__(self):
        super().__init__()
        # how to get frames
        self.frame_cate = ['evt', 'ef']  # frame category(s)
        self.skip_gold_empty = False  # skip no gold frame sentences
        self.skip_pred_empty = False  # skip no pred frame sentences
        self.only_pred_mark = []  # special marks for considered sents
        # final result mix
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
        # labelings
        self.labf_frame = ""
        self.labf_arg = ""  # or require the triple: "(lambda x: (x.label, x.main.label, x.arg.label))"
        # --
        # item/pair filters
        self.filter_frame = ""  # (lambda Frame: ...)
        self.filter_arg = ""  # (lambda Arg: ...)
        self.filter_fpair = ""  # (lambda MP[Frame]: ...)
        self.filter_apair = ""  # (lambda MP[Arg]: ...)
        self.ignore_labels = ["_NIL_", "_UNK_"]
        # --
        # for printing breakdowns
        self.bd_frame_lines = 0  # how many lines to print for breakdown (0 means nope)
        self.bd_arg_lines = 0

    @staticmethod
    def get_conf_options(**kwargs):
        ret = {'plain': FrameEvalConf(), 'ee': FrameEvalConf().direct_update(match_arg_with_frame=False),
               'fn': MyFNEvaler(), 'pb': MyPBEvaler()}
        if kwargs:
            ret = {k: v.direct_update(**kwargs) for k,v in ret.items()}
        return ret

@FrameEvalConf.conf_rd()
class FrameEvaler(Evaluator):
    def __init__(self, conf: FrameEvalConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: FrameEvalConf = self.conf
        self.curr_er = FrameEvalRecord(conf)
        # --
        self.span_getter_frame = Mention.create_span_getter(conf.span_mode_frame)
        self.span_getter_arg = Mention.create_span_getter(conf.span_mode_arg)
        # pair filter
        _default_filter = (lambda x: True)  # by default all pass!
        self.filter_frame = self.filter_arg = self.filter_fpair = self.filter_apair = _default_filter
        for _k in ['filter_frame', 'filter_arg', 'filter_fpair', 'filter_apair']:
            if getattr(conf, _k):
                setattr(self, _k, eval(getattr(conf, _k)))
        # lab function
        self.ignore_labels = set(conf.ignore_labels)
        self.default_labf = (lambda x: x.label.lower())
        self.labf_frame = self.labf_arg = self.default_labf
        if conf.labf_frame:
            self.labf_frame = eval(conf.labf_frame)
        if conf.labf_arg:
            self.labf_arg = eval(conf.labf_arg)
        # --

    def get_er(self): return self.curr_er
    def reset_er(self): self.curr_er = FrameEvalRecord(self.conf)

    def eval(self, pred_insts: List[Union[Doc, Sent]], gold_insts: List[Union[Doc, Sent]]):
        res = FrameEvalRecord(self.conf)
        for one_p, one_g in yield_sent_pairs(pred_insts, gold_insts):
            one_res = self._eval_one(one_p, one_g)
            res += one_res
        # save to the overall one!
        self.curr_er += res
        return res

    def _eval_one(self, pred_inst: Sent, gold_inst: Sent):
        conf: FrameEvalConf = self.conf
        if len(pred_inst) != len(gold_inst):
            zwarn(f"Length mismatch: {len(pred_inst)} vs {len(gold_inst)}")
        pred_frames, gold_frames = self._get_frames(pred_inst), self._get_frames(gold_inst)
        # --
        if conf.skip_gold_empty and len(gold_frames) == 0:
            return FrameEvalRecord(conf)
        if conf.skip_pred_empty and len(pred_frames) == 0:
            return FrameEvalRecord(conf)
        if conf.only_pred_mark:
            if not any(pred_inst.info.get(k) for k in conf.only_pred_mark):
                return FrameEvalRecord(conf)
        # --
        _labf_frame, _labf_arg = self.labf_frame, self.labf_arg
        # first match frames
        frame_pairs = self._match_items(
            pred_frames, gold_frames, self.span_getter_frame, conf.posi_thresh_frame, labf=_labf_frame)
        frame_pairs = [p for p in frame_pairs if self.filter_fpair(p)]
        # then match args & full
        arg_pairs = []
        full_pairs = []
        for fpair in frame_pairs:
            # arg pairs
            has_pred, has_gold = fpair.pred is not None, fpair.gold is not None
            if has_pred and has_gold:
                _pred_args, _gold_args = self._get_args(fpair.pred), self._get_args(fpair.gold)
                if conf.match_arg_with_frame_type and _labf_frame(fpair.pred) != _labf_frame(fpair.gold):
                    # note: no match!
                    one_arg_pairs = [MatchedPair(a, None) for a in _pred_args] + [MatchedPair(None, b) for b in _gold_args]
                else:
                    one_arg_pairs = self._match_items(
                        _pred_args, _gold_args, self.span_getter_arg, conf.posi_thresh_arg, labf=self.labf_arg)
            elif has_pred:
                one_arg_pairs = [MatchedPair(a, None) for a in self._get_args(fpair.pred)]
            else:
                assert has_gold
                one_arg_pairs = [MatchedPair(None, b) for b in self._get_args(fpair.gold)]
            one_arg_pairs = [p for p in one_arg_pairs if self.filter_apair(p)]
            arg_pairs.extend(one_arg_pairs)
            # full pairs
            fpair2 = fpair.copy()
            fpair2.arg_pairs = one_arg_pairs  # directly set here!
            for _key in fpair2.get_mached_score_keys():  # set the scores as multiplied by all
                # note: also ignore frame type if set the option!
                _score = fpair2.get_matched_score(_key) if conf.match_arg_with_frame_type else fpair2.get_matched_score('posi')
                for _ap in one_arg_pairs:
                    _score *= _ap.get_matched_score(_key)
                fpair2.set_matched_score(_key, _score)
            full_pairs.append(fpair2)
        # special mode: match all args (without first match frame)
        if not conf.match_arg_with_frame:
            # special mode, require frame type matching!
            all_pred_args, all_gold_args = [a for f in pred_frames for a in self._get_args(f)], \
                                           [a for f in gold_frames for a in self._get_args(f)]
            # must match type in this mode!!
            ftype_masks = ItemMatcher.score_items(
                [_labf_frame(a.main) for a in all_pred_args], [_labf_frame(a.main) for a in all_gold_args])
            # replace instead
            arg_pairs = self._match_items(
                all_pred_args, all_gold_args, self.span_getter_arg, conf.posi_thresh_arg,
                posi_arr=ftype_masks, labf=self.labf_arg)  # note: especially make this the base one: posi_arr!!
            arg_pairs = [p for p in arg_pairs if self.filter_apair(p)]
        # get results
        res = FrameEvalRecord(conf, frame_pairs, arg_pairs, full_pairs)
        return res

    # matches
    def _match_items(self, items1: List, items2: List, span_getter: Callable, posi_thresh: float,
                     posi_arr=None, lab_arr=None, labf=None):
        conf: FrameEvalConf = self.conf
        lab_alpha = conf.match_lab_alpha
        if labf is None:
            labf = self.default_labf
        # todo(+N): also consider args' matches?
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
        # todo(+N): we do not want to solve the full assignment problem, but use a simple greedy method
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

    # helper
    def _fake_discontinuous_ef(self, all_mentions: List, label: str = None):
        # todo(+N): for simplicity, special support for it!!
        # simply make the fake ones but no adding!!
        start_widx, end_widx = all_mentions[0].widx, all_mentions[0].wridx
        for m in all_mentions[1:]:
            if m.sent is all_mentions[0].sent:  # expand
                start_widx = min(start_widx, m.widx)
                end_widx = max(end_widx, m.wridx)
        fake_ef = Frame(mention=Mention(start_widx, end_widx - start_widx, label=label, par=all_mentions[0].sent))
        # --
        excluded_idxes = set(range(start_widx, end_widx))
        for m in all_mentions:
            for i in range(m.widx, m.wridx):
                if i in excluded_idxes:  # there can be repeated mentions
                    excluded_idxes.remove(i)
        fake_ef.mention.excluded_idxes = excluded_idxes  # todo(+N): ugly fix!!
        return fake_ef

    def _get_frames(self, s: Sent):
        return [f for f in s.yield_frames(cates=self.conf.frame_cate)
                if self.filter_frame(f) and f.label not in self.ignore_labels]

    def _get_args(self, frame: Frame):
        return [a for a in frame.args if self.filter_arg(a) and a.label not in self.ignore_labels]

# --
# record
class FrameEvalRecord(EvalRecord):
    def __init__(self, conf: FrameEvalConf, frame_pairs=None, arg_pairs=None, full_pairs=None):
        self.conf = conf
        # make new lists!
        self.frame_pairs = list(frame_pairs) if frame_pairs is not None else []
        self.arg_pairs = list(arg_pairs) if arg_pairs is not None else []
        self.full_pairs = list(full_pairs) if full_pairs is not None else []
        # calculate some results
        self.current_results = {}  # key -> (frame, arg, full)
        for _k1 in ["posi", "lab"]:
            for _k2, pairs in zip(['frame', 'arg', 'full'], [self.frame_pairs, self.arg_pairs, self.full_pairs]):
                self.current_results[f"{_k1}-{_k2}"] = self._sum_pairs(_k1, pairs)
        # --

    def _sum_pairs(self, key: str, pairs: List[MatchedPair]):
        res = F1EvalEntry()
        for p in pairs:
            one_p, one_g = p.get_pg_results(key)
            res.combine_dn(one_p, one_g)
        return res

    def __iadd__(self, other: 'FrameEvalRecord'):
        # add ones!
        self.frame_pairs.extend(other.frame_pairs)
        self.arg_pairs.extend(other.arg_pairs)
        self.full_pairs.extend(other.full_pairs)
        # add results
        for _key in self.current_results.keys():
            self_entry, other_entry = self.current_results[_key], other.current_results[_key]
            self_entry.combine(other_entry)
        # --
        return self

    def __add__(self, other: 'FrameEvalRecord'):
        ret = self.copy()
        ret += other
        return ret

    def copy(self):
        return FrameEvalRecord(self.conf, self.frame_pairs, self.arg_pairs, self.full_pairs)

    def _get_final_res(self, key: str):
        conf: FrameEvalConf = self.conf
        self.cached_final_results = {}
        final_weights = [conf.weight_frame, conf.weight_arg, conf.weight_full]
        _entry = F1EvalEntry()
        for _k2, _w in zip(['frame', 'arg', 'full'], final_weights):
            one_entry = self.current_results[f"{key}-{_k2}"]
            _entry.combine(one_entry, scale=_w)
        return _entry

    def _get_res_raw(self):
        from copy import deepcopy
        _cr = deepcopy(self.current_results)
        ret = {}
        for _key in ["posi", "lab"]:  # reorder things!
            for _k2 in ['frame', 'arg', 'full']:
                ret[f"{_key}-{_k2}"] = _cr[f"{_key}-{_k2}"]
            ret[f"{_key}-final"] = self._get_final_res(_key)
        return ret

    def get_res(self):
        _cr = self._get_res_raw()
        res = ZResult(_cr, res=_cr["lab-final"].res, des=self.get_str(brief=True))
        return res

    def get_bd_str(self, pairs, lines: int, tag: str):
        df = MatchedPair.get_breakdown(pairs)[0]
        r_macro, r_macro2, r_micro = MatchedPair.df2avg(df)
        summary = {f'ZBreak-{tag}': {'macro': r_macro, 'macro2': r_macro2, 'micro': r_micro}}
        center_line = "" if len(df) <= lines else f" ... (Truncate {lines}/{len(df)})\n"
        ret = f"# -- Breakdown-{tag}\n{df[:lines].to_string()}\n{center_line}{summary}\n# --"
        return ret

    def get_str(self, brief: bool):
        res = self._get_res_raw()
        if brief:
            # one-line brief result (only F1 reported)
            rets = []
            for _key in ["posi", "lab"]:
                f1s = [v.res for k,v in res.items() if k.startswith(_key)]
                rets.append(f"{_key}: " + "/".join([f"{v:.4f}" for v in f1s]))
            return " ||| ".join(rets)
        else:
            # more detailed results
            rets = []
            for _k, _v in res.items():
                rets.append(f"{_k}: {_v}")
            # --
            # get breakdown results?
            if self.conf.bd_frame_lines > 0:
                rets.append(self.get_bd_str(self.frame_pairs, self.conf.bd_frame_lines, 'Frame'))
            if self.conf.bd_arg_lines > 0:
                rets.append(self.get_bd_str(self.arg_pairs, self.conf.bd_arg_lines, 'Arg'))
            # --
            return "\n".join(rets)

# --
# more specific ones

# =====
# special one for FN
# todo(+N): there are some slight diffs (0.sth%) to the semafor-script: possibly from repeated-span frame/args;
#  => therefore, for reporting, finally need to use the semafor-script!!

@EvalConf.rd('frame_fn')
class MyFNEvalConf(FrameEvalConf):
    def __init__(self):
        super().__init__()
        # --
        self.fn_coreness_file = ""  # where to find coreness-file?
        self.fn_version = 'fn15'  # fn15/fn17
        # special weights for core and noncore roles!
        self.weight_core_role = 1.0
        self.weight_noncore_role = 0.5
        self.only_rank1_role = True  # only count rank1 roles!
        self.combine_discontinuous = True  # make a large arg
        self.remove_repeat_frames = True  # remove same-mention frames
        # overwrite the super one!!
        self.skip_gold_empty = True  # by default True

@MyFNEvalConf.conf_rd()
class MyFNEvaler(FrameEvaler):
    def __init__(self, conf: MyFNEvalConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        conf: MyFNEvalConf = self.conf
        # load frame files
        _cfile = zglob1(conf.fn_coreness_file)
        if os.path.exists(_cfile):
            self.coreness_map = default_json_serializer.from_file(_cfile)
            _cextra = f"R:{_cfile}"
        else:
            self.coreness_map = MyFNEvaler._read_coreness_from_nltk(conf.fn_version)
            if _cfile:
                default_json_serializer.to_file(self.coreness_map, _cfile)
                _cextra = f"S:{_cfile}"
            else:
                _cextra = "N"
        self._print_coreness_info(_cextra)
        # --

    def _print_coreness_info(self, extra=''):
        num_arg, num_core = 0, 0
        for v in self.coreness_map.values():
            num_arg += len(v)
            num_core += len([1 for v2 in v.values() if v2=="Core"])
        zlog(f"Info fn-coreness ({extra}): {len(self.coreness_map)} frames, {num_arg} args, {num_core} core-args.")

    def _get_frames(self, s: Sent):
        conf: MyFNEvalConf = self.conf
        # --
        ret_frames = super()._get_frames(s)
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

    def _get_args(self, frame: Frame):
        conf: MyFNEvalConf = self.conf
        # --
        ret_args = super()._get_args(frame)
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
                    fake_arg = ArgLink(fake_ef, label=role, score=sum(a.score for a in alist), par=frame)
                    ret_args.append(fake_arg)
        return ret_args

    def _eval_one(self, pred_inst: Sent, gold_inst: Sent):
        ret = super()._eval_one(pred_inst, gold_inst)
        # --
        # post-processing
        weight_core_role, weight_noncore_role = self.conf.weight_core_role, self.conf.weight_noncore_role
        cmap = self.coreness_map
        # --
        def _get_weight(_arg: ArgLink):
            return weight_core_role if cmap.get(_arg.main.type, {}).get(_arg.role, "") == "Core" else weight_noncore_role
        # --
        for p in ret.arg_pairs:
            gold_weight, pred_weight = None, None
            if p.gold is not None:
                gold_weight = _get_weight(p.gold)
            if p.pred is not None:
                pred_weight = _get_weight(p.pred)
            p.set_weights(gold_weight, pred_weight)
        return ret.copy()  # note: re-calculate with new weights

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

@EvalConf.rd('frame_pb')
class MyPBEvalConf(FrameEvalConf):
    def __init__(self):
        super().__init__()
        # --
        self.no_join_c = False  # disable joining c-* together as in conll05 eval!
        # overwrite the super one!
        self.weight_frame = 0.
        self.match_arg_with_frame_type = False  # by default, no need for frame type

@MyPBEvalConf.conf_rd()
class MyPBEvaler(FrameEvaler):
    def __init__(self, conf: MyPBEvalConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --

    def _get_frames(self, s: Sent):
        # remove repeated ones and keep wlen==1
        hit_frames = [None] * len(s)
        for f in super()._get_frames(s):
            # todo(+N): here only output len==1
            if f.mention.wlen != 1:
                zwarn(f"Meet strange pb-frame (wlen>1): {f}")
                continue
            one_widx = f.mention.widx
            hit_frames[one_widx] = f
        return [f for f in hit_frames if f is not None]

    def _get_args(self, frame: Frame):
        _no_join_c = self.conf.no_join_c
        # note: here we don't make it a seq, but do combine A? & C-A?
        last_role_maps = {}
        arg_groups = []
        for a in sorted(super()._get_args(frame), key=lambda x: x.mention.get_span()):
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
                fake_arg = ArgLink(fake_ef, label=role, score=sum(a.score for a in alist), par=frame)
                ret_args.append(fake_arg)
        return ret_args

# --
# b mspx/proc/eval/frame:
