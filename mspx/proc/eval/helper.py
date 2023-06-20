#

# some helpers for eval

__all__ = [
    "ItemMatcher", "MatchedPair",
]

from typing import List, Callable, Dict, Union
from collections import defaultdict, Counter
import numpy as np
from mspx.utils import Constants, DivNumber, F1EvalEntry, ZObject
from mspx.data.inst import Mention

class ItemMatcher:
    # =====
    # score matching pairs

    @staticmethod
    def score_mention_pair(m1: Mention, m2: Mention, span_getter: Callable) -> float:
        if m1.sent.sid != m2.sent.sid:  # first need sent match
            return 0.
        if span_getter is None:  # default one
            span_getter = lambda x: x.get_span()
        if hasattr(m1, "excluded_idxes") or hasattr(m2, "excluded_idxes"):  # todo(+N): ugly fix!!
            start1, len1 = span_getter(m1)
            start2, len2 = span_getter(m2)
            empty_set = set()
            s1 = set(range(start1, start1 + len1)).difference(getattr(m1, "excluded_idxes", empty_set))
            s2 = set(range(start2, start2 + len2)).difference(getattr(m2, "excluded_idxes", empty_set))
            score = len(s1.intersection(s2)) / len(s1.union(s2))
            return score
        else:
            # [start, len)
            start1, len1 = span_getter(m1)
            start2, len2 = span_getter(m2)
            overlap = min(start1 + len1, start2 + len2) - max(start1, start2)
            overlap = max(0, overlap)  # overlapped tokens
            score = overlap / (len1 + len2 - overlap)  # using Jaccard Index
            return score

    # get the matching score matrix for mentions
    @staticmethod
    def score_mentions(list1: List[Mention], list2: List[Mention], span_getter: Callable, min_thresh=0.):
        _f = ItemMatcher.score_mention_pair
        if span_getter is None:  # default one
            span_getter = lambda x: x.get_span()
        _ff = lambda x,y: _f(x,y,span_getter)
        return ItemMatcher.score_items(list1, list2, min_thresh=min_thresh, match_f=_ff)

    # general one: get matching matrix
    @staticmethod
    def score_items(list1: List, list2: List, min_thresh=0., match_f: Callable = (lambda x,y: float(x==y))):
        size1, size2 = len(list1), len(list2)
        scores = [match_f(m1, m2) for m1 in list1 for m2 in list2]
        scores_arr = np.asarray(scores).reshape([size1, size2])  # [s1, s2]
        if min_thresh > 0.:  # only apply if >0.
            scores_arr *= (scores_arr >= min_thresh)  # apply thresh!
        return scores_arr

    # =====
    # match with scores

    # simple one by two-side greedy picking: O(N^2)
    @staticmethod
    def match_simple(match_score_arr: np.ndarray, min_thresh=1e-5):
        size1, size2 = match_score_arr.shape
        # --
        # empty checking
        if size2 == 0:
            return [], list(range(size1)), []
        # --
        # first follow each dim1's max scores and calculate dim2's max ones
        max_s2: List[int] = np.argmax(match_score_arr, -1).tolist()  # [size1]
        best_idxes, best_scores = [None] * size2, [0.] * size2  # [size2]
        for idx_s1, idx_s2 in enumerate(max_s2):
            _score = match_score_arr[idx_s1, idx_s2]
            if _score>=min_thresh and _score>best_scores[idx_s2]:
                best_idxes[idx_s2] = idx_s1
                best_scores[idx_s2] = _score
        # put the confirmed ones!
        best12, best21 = [None]*size1, best_idxes  # [size1], [size2]
        updated_match_score_arr = match_score_arr.copy()
        _NEG = Constants.REAL_PRAC_MIN
        for idx_s2, idx_s1 in enumerate(best_idxes):
            if idx_s1 is None:
                continue
            updated_match_score_arr[:, idx_s2] += _NEG  # exclude it!
            assert best12[idx_s1] is None
            best12[idx_s1] = idx_s2
        # greedily resolve the rest
        for idx_s1, idx_s2 in enumerate(best12):
            if idx_s2 is not None:  # already matched
                continue
            max_s2: int = updated_match_score_arr[idx_s1].argmax().item()
            if updated_match_score_arr[idx_s1, max_s2] >= min_thresh:  # find extra one!
                best12[idx_s1] = max_s2
                assert best21[max_s2] is None
                best21[max_s2] = idx_s1
                updated_match_score_arr[:, max_s2] += _NEG
        # return
        matched_pairs = [(a,b) for a,b in enumerate(best12) if b is not None]
        unmatched1, unmatched2 = [a for a,b in enumerate(best12) if b is None], [a for a,b in enumerate(best21) if b is None]
        return matched_pairs, unmatched1, unmatched2

# --
class MatchedPair:
    def __init__(self, pred, gold, matched_scores: Dict = None):
        assert not (pred is None and gold is None), "Err: cannot put both None as pair!"
        self.pred, self.gold = pred, gold
        self.matched_scores = defaultdict(float)  # [0.,1.] by default zero
        if matched_scores is not None:
            self.matched_scores.update(matched_scores)
        # weights (by default both as 1.)
        self.pred_weight, self.gold_weight = 1., 1.

    def set_weights(self, pred_weight: float = None, gold_weight: float = None):
        if pred_weight is not None:
            self.pred_weight = pred_weight
        if gold_weight is not None:
            self.gold_weight = gold_weight

    def is_matched(self):
        return self.pred is not None and self.gold is not None

    def copy(self):
        ret = MatchedPair(self.pred, self.gold, matched_scores=self.matched_scores.copy())
        ret.set_weights(self.pred_weight, self.gold_weight)
        return ret

    def get_mached_score_keys(self):
        return self.matched_scores.keys()

    def get_matched_score(self, key: str):
        return self.matched_scores.get(key, 0.)

    def set_matched_score(self, key: str, value: float):
        self.matched_scores[key] = value

    # get results for p&g
    def get_pg_results(self, key: str):
        _matched_score = self.get_matched_score(key)
        # extra check for non-match ones
        res_p = DivNumber(0., 0.) if (self.pred is None) else DivNumber(_matched_score*self.pred_weight, self.pred_weight)
        res_g = DivNumber(0., 0.) if (self.gold is None) else DivNumber(_matched_score*self.gold_weight, self.gold_weight)
        return res_p, res_g

    # --
    # help function to do breakdowns (sort=-3 means gold-count)
    @staticmethod
    def get_breakdown(pairs: List['MatchedPair'], pcode=(lambda x: x.label), gcode=(lambda y: y.label),
                      corr_code=(lambda x,y: x.label==y.label), sort_key=-3, do_macro=False):
        import pandas as pd
        # --
        # get functions
        functions = [pcode, gcode, corr_code]
        for ii in range(len(functions)):
            if isinstance(functions[ii], str):
                functions[ii] = eval(functions[ii])
        _fp, _fg, _fc = functions
        # --
        res = defaultdict(F1EvalEntry)
        for pp in pairs:
            corr = 0
            if pp.pred is not None and pp.gold is not None:
                corr = int(_fc(pp.pred, pp.gold))
            if pp.pred is not None:
                key_p = _fp(pp.pred)
                res[key_p].record_p(corr)
            if pp.gold is not None:
                key_g = _fg(pp.gold)
                res[key_g].record_r(corr)
        # --
        # final
        details = [(k,)+v.details for k,v in res.items()]
        details = sorted(details, key=(lambda x: x[sort_key]), reverse=True)  # by default, sort by gold count
        if do_macro:
            macro_line = ['_Macro'] + [0] * 7
            macro_line[3] = np.mean([z[3] for z in details])
            macro_line[6] = np.mean([z[6] for z in details])
            macro_line[7] = np.mean([z[7] for z in details])
            details.append(macro_line)
            res["_Macro"] = ZObject(res=macro_line[-1], prf=[macro_line[z] for z in [3,6,7]], details=macro_line[1:])
        df = pd.DataFrame(details, columns=['T', 'Pc', 'Pa', 'P', 'Rc', 'Ra', 'R', 'F1'])
        return df, res

    @staticmethod
    def df2avg(df, ra_thr: int = 1):
        # get macro/micro average
        macro_res = [df[z].mean() for z in ['P', 'R', 'F1']]
        df2 = df[df['Ra'] >= ra_thr]  # only appearing ones
        macro2_res = [df2[z].mean() for z in ['P', 'R', 'F1']]
        micro_entry = F1EvalEntry()
        micro_entry.record_p(df['Pc'].sum(), df['Pa'].sum())
        micro_entry.record_r(df['Rc'].sum(), df['Ra'].sum())
        micro_res = list(micro_entry.prf)
        # --
        ret = macro_res, macro2_res, micro_res
        return [[round(z2, 4) for z2 in z] for z in ret]  # keep 4 digits

    # allow an additional mcode(matched-code!)
    @staticmethod
    def get_confusion(pairs: List['MatchedPair'], pcode=(lambda x: x.label), gcode=(lambda y: y.label),
                      mcode=(lambda x,y: True), sort_key=-3, include_nil=True):
        import pandas as pd
        _NIL = '_NIL'
        # --
        # get functions
        functions = [pcode, gcode, mcode]
        for ii in range(len(functions)):
            if isinstance(functions[ii], str):
                functions[ii] = eval(functions[ii])
        _fp, _fg, _fm = functions
        df0 = MatchedPair.get_breakdown(pairs, _fp, _fg, (lambda x,y: _fm(x,y) and _fp(x)==_fg(y)), sort_key)
        # --
        stat = defaultdict(Counter)
        for pp in pairs:
            matched0 = True  # matched or not?
            if pp.pred is None or pp.gold is None:
                if not include_nil:
                    continue  # skip not matched!
            else:
                matched0 = mcode(pp.pred, pp.gold)
                if not matched0 and not include_nil:
                    continue  # skip not matched!
            label_p, label_g = _NIL if pp.pred is None else _fp(pp.pred), _NIL if pp.gold is None else _fg(pp.gold)
            if not matched0:
                stat[_NIL][label_g] += 1
                stat[label_p][_NIL] += 1
            else:
                stat[label_p][label_g] += 1
        # return
        names = ([_NIL] if include_nil else []) + df0['T'].tolist()
        arr = np.asarray([[stat[p][g] for g in names] for p in names])
        df = pd.DataFrame(arr, index=names, columns=names)
        return df, stat
