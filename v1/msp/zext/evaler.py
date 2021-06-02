#

from typing import Dict
from collections import Counter

from msp.utils import zlog, Helper

class DivResult:
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        if b == 0.:
            assert a == 0., "Illegal non-zero/zero"
            self.res = 0.
        else:
            self.res = a/b

    def __repr__(self):
        return f"[{self.a}/{self.b}={self.res:.4f}]"

    def __float__(self):
        return self.res

class F1Result:
    def __init__(self, p, r):
        self.p = p
        self.r = r
        p, r = float(p), float(r)
        self.f1 = float(DivResult(2*p*r, p+r))

    def __repr__(self):
        return f"(p={self.p}, r={self.r}, f={self.f1:.4f})"

    def __float__(self):
        return self.f1

class LabelF1Evaler:
    def __init__(self, name, ignore_none=False):
        # key -> List[labels]
        self.name = name
        self.golds = {}
        self.preds = {}
        self.labels = set()
        self.ignore_none = ignore_none

    # =====
    # adding ones

    def _add_group(self, d: Dict, key, label):
        if key not in d:
            d[key] = [label]
        else:
            d[key].append(label)
        self.labels.add(label)

    def add_gold(self, key, label):
        if key is None and self.ignore_none:
            return
        self._add_group(self.golds, key, label)

    def add_pred(self, key, label):
        if key is None and self.ignore_none:
            return
        self._add_group(self.preds, key, label)

    # =====
    # results

    # this can be precision(base=pred) or recall(base=gold)
    def _calc_result(self, base_d: Dict, query_d: Dict):
        counts = {k:0 for k in self.labels}
        corrs = {k:0 for k in self.labels}
        ucorrs = 0
        for one_k, one_labels in base_d.items():
            other_label_list = query_d.get(one_k, [])
            other_label_maps = Counter(other_label_list)  # label -> count
            ucorrs += min(len(one_labels), len(other_label_list))
            for one_lab in one_labels:
                counts[one_lab] += 1
                remaining_budget = other_label_maps.get(one_lab, 0)
                if remaining_budget > 0:
                    corrs[one_lab] += 1
                    other_label_maps[one_lab] = remaining_budget - 1
        results = {k: DivResult(corrs[k], counts[k]) for k in self.labels}
        all_result_u = DivResult(ucorrs, sum(counts.values()))
        all_result_l = DivResult(sum(corrs.values()), sum(counts.values()))
        return all_result_u, all_result_l, results

    # evaluate results
    def eval(self, quiet=True, breakdown=False):
        all_pre_u, all_pre_l, label_pres = self._calc_result(self.preds, self.golds)
        all_rec_u, all_rec_l, label_recs = self._calc_result(self.golds, self.preds)
        all_f_u = F1Result(all_pre_u, all_rec_u)
        all_f_l = F1Result(all_pre_l, all_rec_l)
        label_fs = {k: F1Result(label_pres[k], label_recs[k]) for k in self.labels} if breakdown else {}
        if not quiet:
            zlog(f"Overall f1 score for {self.name}: unlabeled {all_f_u}; labeled {all_f_l}")
            zlog("Breakdowns: \n" + Helper.printd_str(label_fs))
        return all_f_u, all_f_l, label_fs

    # analysis
    def analyze(self):
        pass

# b msp/zext/evaler:87
