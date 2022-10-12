#

# common base extractor module

__all__ = [
    "ZTaskBaseEConf", "ZTaskBaseE", "ZModBaseEConf", "ZModBaseE",
]

from typing import List
from msp2.nn.l3 import *
from msp2.utils import *
from ..pretrained import *
from ...core import ZTaskConf, ZTask, ZModConf, ZMod

class ZTaskBaseEConf(ZTaskConf):
    def __init__(self):
        super().__init__()
        # --

class ZTaskBaseE(ZTask):
    def __init__(self, conf: ZTaskBaseEConf):
        super().__init__(conf)
        conf: ZTaskBaseEConf = self.conf
        # --

class ZModBaseEConf(ZModConf):
    def __init__(self):
        super().__init__()
        # --
        self.bconf = ConfEntryChoices({
            "bmod": ZBmodConf(), "bert": ZBertConf(), "bart": None, "wvec": ZWvecConf(), "shared": None}, "bmod")
        self.shared_bmod_name = ""  # if bert is shared from others, the name of that mod!
        # --

@node_reg(ZModBaseEConf)
class ZModBaseE(ZMod):
    def __init__(self, conf: ZModBaseEConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, **kwargs)
        conf: ZModBaseEConf = self.conf
        # --
        # setup bmod
        if conf.bconf is None:
            _mod = zmodel.get_mod(conf.shared_bmod_name)
            assert isinstance(_mod, ZModBaseE)
            _bmod = _mod.bmod
        else:
            _bmod = conf.bconf.make_node()
        self.bmod = _bmod
        # --

    @property
    def tokenizer(self):
        return self.bmod.tokenizer

    # --
    # common helpers

    # do sub-tokenization and cache
    def prep_sent(self, sent, ret_toks=False):
        sub_toker = self.bmod.sub_toker
        _sub_key = f"_sub_{sub_toker.key}"
        if _sub_key not in sent._cache:
            sub_vals, sub_idxes, sub_info = sub_toker.sub_vals(sent.seq_word.vals)
            cache_idxes = [sub_idxes[a:b] for a,b in zip(sub_info.orig2begin, sub_info.orig2end)]  # List[List]
            sent._cache[_sub_key] = cache_idxes
        cache_idxes = sent._cache[_sub_key]
        if ret_toks:
            return cache_idxes, sent.tokens
        else:
            return cache_idxes

    # extend sent?
    def extend_ctx_sent(self, sent, n: int, ret_toks=False):
        curr = sent
        before_ids, before_toks = [], []
        for ii in range(n):  # add before
            curr = curr.prev_sent
            if curr is None: break
            before_ids = self.prep_sent(curr) + before_ids
            before_toks = curr.tokens + before_toks
        curr = sent
        after_ids, after_toks = [], []
        for ii in range(n):  # add after
            curr = curr.next_sent
            if curr is None: break
            after_ids.extend(self.prep_sent(curr))
            after_toks.extend(curr.tokens)
        if ret_toks:
            return before_ids, after_ids, before_toks, after_toks
        else:
            return before_ids, after_ids

    # truncate them if too long, slightly prefer previous context (by before_ratio)
    def truncate_subseq(self, sub_lens, truncate_center: int, max_len: int, before_ratio=0.65, silent=False):
        before_budget = int(max_len * before_ratio)
        after_budget = max_len - before_budget
        t_start = truncate_center
        while t_start > 0 and before_budget - sub_lens[t_start-1] >= 0:
            before_budget -= sub_lens[t_start-1]
            t_start -= 1
        after_budget = after_budget + before_budget - sub_lens[truncate_center]
        t_end = truncate_center + 1
        while t_end < len(sub_lens) and after_budget - sub_lens[t_end] >= 0:
            after_budget -= sub_lens[t_end]
            t_end += 1
        if not silent:
            zwarn(f"Truncate seq, full_len={sum(sub_lens)}->{sum(sub_lens[t_start:t_end])},"
                  f" [{len(sub_lens)}][{t_start}:{truncate_center}:{t_end}]")
        return t_start, t_end

# --
# b msp2/tasks/zmtl3/mod/extract/base:??
