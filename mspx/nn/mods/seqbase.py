#

# basic seq ZTask/Zmod

__all__ = [
    'ZTaskSbConf', 'ZTaskSb', 'ZModSbConf', 'ZModSb',
]

import numpy as np
from mspx.data.inst import DataPadder
from mspx.utils import ConfEntryChoices, zwarn, zlog
from mspx.nn import BK, NnConf, NnLayer, CombinerConf
from ..layers2 import *
from .helper import *
from .mod import *

# --
# basis of seq mods, this may also serve as a standalone encoder!

@ZTaskConf.rd('sb')
class ZTaskSbConf(ZTaskConf):
    def __init__(self):
        super().__init__()
        self.mod = ZModSbConf()

@ZTaskSbConf.conf_rd()
class ZTaskSb(ZTask):
    pass

@ZModConf.rd('sb')
class ZModSbConf(ZModConf):
    def __init__(self, default_bmod=''):
        super().__init__()
        # --
        self.bconf = ConfEntryChoices({
            "bmod1": ZBmod1Conf(), "bmod2": ZBmod2Conf(), "shared": "shared"}, default_bmod)
        self.shared_bmod_name = ""  # if bert is shared from others, the name of that mod!
        # input & output
        self.input_name = ""  # use other's output as input for bmod
        self.bout = BoutConf()  # final output
        # output with gnn
        from mspx.nn.layers import GnnConf
        self.out_gnn_name = ''  # att from which module?
        self.gnn_conf = GnnConf()
        # --
        self.max_seq_len = 128  # maximum seq length
        self.ctx_nsent = 0  # how many more before & after?
        self.remain_toks = False  # need to keep tok information?
        self.do_sub_split = True  # do sub splits? (word level or inputs already subtokenzied)

@ZModSbConf.conf_rd()
class ZModSb(ZMod):
    def __init__(self, conf: ZModSbConf, ztask: ZTask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModSbConf = self.conf
        # --
        # setup bmod
        _bmod: ZBmodBaseMod = None
        if conf.bconf is not None:
            if conf.bconf == 'shared':
                _mod = zmodel.get_mod(conf.shared_bmod_name)
                assert isinstance(_mod, ZMod)
                _bmod = _mod.bmod
            else:
                _bmod = conf.bconf.make_node()
        self.bmod = _bmod
        if _bmod is not None:
            self.bout = conf.bout.make_node(bert_dim=_bmod.get_mdim(), att_num=_bmod.get_head_num())
        else:
            self.bout = None
        if conf.out_gnn_name:
            self.gnn = conf.gnn_conf.make_node(dim=_bmod.get_mdim())
        else:
            self.gnn = None
        # --
        self.cache_version = 0  # current cache version!
        # --

    def update_cache_version(self):
        self.cache_version += 1
        zlog(f"Update cache_version to {self.cache_version}!!")

    def form_cache_key(self, extra=''):
        return f"C{self.cache_version}_{self.name}_{extra}"

    def prepare_sent(self, sent, remain_toks: bool):
        _toker = self.bmod.toker
        if self.conf.do_sub_split:
            _sf = sent.seq_word.get_sf(sub_toker=_toker)
            sub_vals, sub_idxes, sub_info = _sf.vals, _sf.idxes, _sf.ma_info
            if remain_toks:  # keep token info
                _idxes = [sub_idxes[a:b] for a, b in zip(sub_info.o2n_start, sub_info.o2n_end)]  # List[List]
                _tokens = sent.tokens
            else:  # no need to keep token information!
                _idxes = sub_idxes
                _tokens = None
        else:
            _idxes = _toker.convert_tokens_to_ids(sent.seq_word.vals)
            _tokens = sent.tokens if remain_toks else None
        return _idxes, _tokens

    # extend sent?
    def extend_ctx_sent(self, sent, n: int, remain_toks: bool):
        curr = sent
        before_ids, before_toks = [], ([] if remain_toks else None)
        for ii in range(n):  # add before
            curr = curr.prev_sent
            if curr is None: break
            a, b = self.prep_sent(curr, remain_toks)
            before_ids = a + before_ids
            if remain_toks:
                before_toks = b + before_toks
        curr = sent
        after_ids, after_toks = [], ([] if remain_toks else None)
        for ii in range(n):  # add after
            curr = curr.next_sent
            if curr is None: break
            a, b = self.prepare_sent(curr, remain_toks)
            after_ids.extend(a)
            if remain_toks:
                after_toks.extend(b)
        return before_ids, after_ids, before_toks, after_toks

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

    def prepare_one_item(self, item, remain_toks: bool):
        conf: ZModSbConf = self.conf
        # --
        sent = item.sent
        seq_idxes, seq_toks = self.prepare_sent(sent, remain_toks)
        truncate_center = 0
        # extend context?
        if conf.ctx_nsent > 0:
            before_ids, after_ids, before_toks, after_toks = self.extend_ctx_sent(sent, conf.ctx_nsent, remain_toks)
            _before_len, _after_len = len(before_ids), len(after_ids)
            truncate_center += _before_len  # note: remember to add this!
            seq_idxes = before_ids + seq_idxes + after_ids
            if remain_toks:
                seq_toks = before_toks + seq_toks + after_toks
        # check max_len & truncate
        if remain_toks:
            sub_lens = [len(z) for z in seq_idxes]
            if sum(sub_lens) > conf.max_seq_len:  # truncate things!
                t_start, t_end = self.truncate_subseq(sub_lens, truncate_center, conf.max_seq_len, silent=True)
                seq_idxes, seq_toks = seq_idxes[t_start:t_end], seq_toks[t_start:t_end]
        else:  # note: simply truncate directly
            if len(seq_idxes) > conf.max_seq_len:
                t_start = max(0, truncate_center - int(conf.max_seq_len * 0.65))
                t_end = min(t_start + conf.max_seq_len, len(seq_idxes))
                seq_idxes = seq_idxes[t_start:t_end]
        # construct seq
        _toker = self.bmod.toker
        if remain_toks:
            ret = [[_toker.cls_token_id]] + seq_idxes + [[_toker.sep_token_id]], [None] + seq_toks + [None]
        else:
            ret = [_toker.cls_token_id] + seq_idxes + [_toker.sep_token_id], None
        return ret

    def prepare_ibatch(self, ibatch, remain_toks: bool, no_cache=False):
        _key = self.form_cache_key()
        _toker = self.bmod.toker
        _mask_id, _pad_id = _toker.mask_token_id, _toker.pad_token_id
        # --
        all_caches = []
        for item in ibatch.items:
            _cache = item.cache.get(_key)
            if _cache is None or no_cache:
                _cache = self.prepare_one_item(item, remain_toks)
                if not no_cache:
                    item.cache[_key] = _cache
            all_caches.append(_cache)
        # --
        if remain_toks:
            arr_ids = [sum(z[0], []) for z in all_caches]
            arr_sublens = [[len(z2) for z2 in z[0]] for z in all_caches]
            arr_toks = [z[1] for z in all_caches]
            t_ids, _ = DataPadder.batch_2d(arr_ids, _pad_id, ret_tensor=True)  # [bs, len1]
            t_sublens, _ = DataPadder.batch_2d(arr_sublens, 0, ret_tensor=True)  # [bs, len0]
            arr_toks, _ = DataPadder.batch_2d(arr_toks, None, dtype=object)  # [bs, len0]
        else:
            arr_ids = [z[0] for z in all_caches]
            t_ids, _ = DataPadder.batch_2d(arr_ids, _pad_id, ret_tensor=True)  # [bs, len1]
            t_sublens = arr_toks = None
        t_mask = (t_ids != _pad_id).to(BK.DEFAULT_FLOAT)
        return t_ids, t_mask, t_sublens, arr_toks

    # --
    def do_prep(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModSbConf = self.conf
        _input_name, _remain_toks = conf.input_name, conf.remain_toks
        if _input_name:  # no need to prepare inputs
            _input_name0 = _input_name.split(":")[0]  # note: at the mod's level!
            t_ids, t_mask, t_sublens, arr_toks = rc.get_cache((_input_name0, 'input'))
        else:
            t_ids, t_mask, t_sublens, arr_toks = self.prepare_ibatch(rc.ibatch, _remain_toks)
        rc.set_cache((self.name, 'input'), (t_ids, t_mask, t_sublens, arr_toks))

    def _do_forward(self, rc: ZRunCache, input_f=None):
        conf: ZModSbConf = self.conf
        _input_name, _remain_toks = conf.input_name, conf.remain_toks
        # --
        t_ids, t_mask, t_sublens, arr_toks = rc.get_cache((self.name, 'input'))
        if _input_name:  # use other inputs
            _ihid = rc.get_cache(_input_name)  # specified in the name, for eg: "enc0:bout:hidden_states:-1"
            if input_f is not None:
                _ihid = input_f(_ihid)
            bout = self.bmod.forward_enc(None, t_mask=t_mask, t_ihid=_ihid)
        else:
            if input_f is None:  # plain
                bout = self.bmod.forward_enc(t_ids, t_mask=t_mask)
            else:  # add to input emb!
                _ihid = input_f(self.bmod.forward_emb(t_ids, forw_full=True))
                bout = self.bmod.forward_enc(None, t_mask=t_mask, t_ihid=_ihid)
        # --
        # outputs
        rc.set_cache((self.name, 'bout'), bout)
        ret_enc = self.bout(bout, rc, (t_sublens if _remain_toks else None))
        rc.set_cache((self.name, 'enc'), ret_enc)
        # --
        final_ret = ret_enc['ET' if _remain_toks else 'E']
        final_ret = self.pass_gnn(final_ret, rc, arr_toks)
        return final_ret

    def pass_gnn(self, ret, rc, arr_toks, bidxes=None):
        conf = self.conf
        if conf.out_gnn_name:
            t_att = rc.get_cache((conf.out_gnn_name, 't_att'))
            if bidxes is not None:
                t_att = t_att[bidxes]
                arr_toks = arr_toks[bidxes.cpu().numpy()]
            vfunc = np.vectorize((lambda x: float(x is not None)))
            arr_tmask = vfunc(arr_toks).astype(np.float32)
            arr_tmask[:, 0] = 1.  # AROOT!
            t_tmask = BK.input_real(arr_tmask)
            ret = self.gnn(ret, t_att, t_tmask)  # pass through gnn
        return ret

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        self._do_forward(rc)
        return (None, {})

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        self._do_forward(rc)
        return {}

# --
# b mspx/nn/mods/seqbase:??
