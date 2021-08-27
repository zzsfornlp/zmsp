#

# some simple post-processing

__all__ = [
    "PostProcessorConf", "PostProcessor",
]

from typing import List
from collections import defaultdict
from msp2.nn import BK
from msp2.nn.layers import *
import re

# --

class PostProcessorConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        # label_budget and non_overlapping
        self.label_budget_preload = ""  # preload what
        self.label_budget = 100  # max budget for each role
        self.non_overlapping = False  # check overlapping
        # label mapping
        self.shorten_arg = False  # ARG->A
        self.strategy_c = 'keep'  # for C-A*, keep/delete/strip
        self.strategy_r = 'keep'  # for R-A*, keep/delete/strip
        # --

@node_reg(PostProcessorConf)
class PostProcessor(BasicNode):
    def __init__(self, conf: PostProcessorConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PostProcessorConf = self.conf
        # --
        self._del_f = lambda x: x.delete_self()  # note: currently only for arglink!
        # --
        from msp2.data.resources import get_frames_label_budgets
        self.label_budget_preload = None  # by default, nope!
        if conf.label_budget_preload:
            self.label_budget_preload = get_frames_label_budgets(conf.label_budget_preload)
        # --

    # helpers
    def prune_by_label_budget(self, sorted_insts: List, label_budget: int, frame_label: str):
        # label budget?
        ref_budgets = None if self.label_budget_preload is None else self.label_budget_preload.get(frame_label)
        cur_budgets = {}
        # --
        survived_insts = []
        for one in sorted_insts:
            one_label = one.label
            one_label_count = cur_budgets.get(one_label, 0)
            # --
            _to_delete = False
            if ref_budgets is not None:  # if we have ref_budgets, just use this!!
                if one_label_count >= ref_budgets.get(one_label, 0):
                    _to_delete = True
            elif one_label_count >= label_budget:
                _to_delete = True
            # --
            if _to_delete:
                self._del_f(one)  # delete this one!!
            else:
                survived_insts.append(one)
                cur_budgets[one_label] = one_label_count + 1
        return survived_insts

    def prune_by_overlapping(self, sorted_insts: List):
        # non overlapping
        cur_hits = {}  # id(sent) -> list[HIT?]
        survived_insts = []
        for one in sorted_insts:
            # check records
            sent = one.mention.sent
            one_hits = cur_hits.get(id(sent))
            if one_hits is None:
                one_hits = [False] * len(sent)
                cur_hits[id(sent)] = one_hits
            # check hits?
            one_widx, one_wlen = one.mention.get_span()  # use full span!
            if any(one_hits[i] for i in range(one_widx, one_widx + one_wlen)):
                self._del_f(one)  # delete this one!!
            else:
                survived_insts.append(one)
                one_hits[one_widx:one_widx + one_wlen] = [True] * one_wlen
        return survived_insts

    def prune_by_prefix(self, insts: List, label_prefix: str):
        survived_insts = []
        for one in insts:
            if one.label.startswith(label_prefix):
                self._del_f(one)  # delete this one!!
            else:
                survived_insts.append(one)
        return survived_insts

    # main one
    def process(self, evt):
        conf: PostProcessorConf = self.conf
        insts = list(evt.args)
        # --
        cur_insts = sorted(insts, key=(lambda x: x.score), reverse=True)  # sort by score
        # change label?
        if conf.shorten_arg:  # change "ARG" to "A"
            for one in cur_insts:
                one.set_label(one.label.replace("ARG", "A"))
        for _stra, _prefix in zip([conf.strategy_c, conf.strategy_r], ["C-", "R-"]):
            if _stra == "keep":  # no change
                continue
            elif _stra == "delete":  # delete those
                cur_insts = self.prune_by_prefix(cur_insts, _prefix)
            elif _stra == "strip":  # remove those prefixes from labels
                for one in cur_insts:
                    if one.label.startswith(_prefix):
                        one.set_label(one.label[len(_prefix):])
        # label budget?
        if conf.label_budget < len(cur_insts):
            cur_insts = self.prune_by_label_budget(cur_insts, conf.label_budget, evt.label)
        # non overlapping?
        if conf.non_overlapping:
            cur_insts = self.prune_by_overlapping(cur_insts)
        # --
        return cur_insts

# --
# b msp2/tasks/zmtl2/zmod/dec/dec_srl/postprocess:??
