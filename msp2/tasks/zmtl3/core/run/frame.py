#

# frame specific ones
# mainly in training mode!!
# note: we do truncating later in models and ignore seq-lengths here!

import numpy as np
from collections import Counter
import pandas as pd
from msp2.utils import Conf, Constants, Random, zlog, zwarn, zglob1z
from msp2.data.inst import yield_frames, yield_sents, set_ee_heads
from .common import *

# --

class FrameBatcherConf(Conf):
    def __init__(self):
        self.mix_neg_sents = False  # mix neg sentences (the decoder need to be able to handle this!)
        self.type_sample_replace = False  # replace sampling for type?
        self.type_sample_alpha = 0.  # sample types according to (freq**alpha)
        self.n_type = [8,0,0,0]  # how many type to sample each batch: [plain_n0, special_n, contrast_n, plain_n], by default add 0s
        self.k_shot = 5  # how many insts to sample per type
        self.remove_fcount = 0  # remove frames that are less than this count (simply removed)
        self.ignore_fcount = 2  # ignore frames that are less than this count (no yielding)
        self.ignore_types = ['be.03','become.03','do.01','have.01']  # ignore types: for example, pb's aux verbs
        self.filter_onto = ""
        self.filter_onto_do_amap = False  # do a_map for args
        self.msent_n = 0  # extra ones for those have msent args
        # --
        # special arranging
        self.lemma_vsize = 10000  # when building lemma-based sim, the size of lemma vocab
        self.lemma_thresh = 0.1  # larger than this!
        self.score_tau = 0.1  # exp(margin_score/this)
        self.score_avgE = 5.  # add this to denominator to down-weight small groups
        self.score_alpha = 1.  # final: score ** alpha
        # --
        # special sampling
        self.sample_seed = 12345
        self.sample_num = -1  # valid if >0
        # --

    def get_batcher(self):
        return FrameBatcher(self)

class FrameBatcher:
    def __init__(self, conf: FrameBatcherConf):
        self.conf = conf
        self.type_budgets = (conf.n_type + [0]*4)[:4]  # [plain_n0, special_n, contrast_n, plain_n]
        # --
        # current status
        self.all_types, self.all_items = None, None
        self.rearrange_info = None
        # --
        self.ignore_types = set(conf.ignore_types)
        if conf.filter_onto:
            # todo(+N): not good design ...
            from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto
            _path = zglob1z(conf.filter_onto)
            self.filter_onto = zonto.Onto.load_onto(_path)
        else:
            self.filter_onto = None
        # --

    def yield_batches(self, stream, loop: bool):
        conf = self.conf
        _gen = Random.get_generator('stream')
        assert loop, "This Batcher is specifically utilized for looping mode!"
        # --
        _fonto = self.filter_onto
        _items = {}  # type -> List
        cc = Counter()
        for inst in stream:
            cc['orig_inst'] += 1
            for evt in yield_frames(inst):
                cc['orig_evt'] += 1
                cc['orig_arg'] += len(evt.args)
                _type = evt.type
                if _type not in _items:
                    cc['orig_etype'] += 1
                    _items[_type] = []
                _items[_type].append(DataItem(evt))  # wrap it here!
            if conf.mix_neg_sents:
                for sent in yield_sents(inst):
                    if len(sent.events) == 0:  # otherwise these sents will not appear!
                        _type = '_NIL'
                        if _type not in _items:
                            _items[_type] = []
                        _items[_type].append(DataItem(sent))  # wrap it here!
        zlog(f"Data stat (orig) for this frame batcher: {cc}")
        all_types = []
        all_items = []
        remove_counts, ignore_counts, ignoreT_counts, filter_counts = [], [], [], []
        for k in sorted(_items.keys(), key=(lambda x: (-len(_items[x]), x))):  # sort by freq & name
            _counts = len(_items[k])
            if len(_items[k]) < conf.remove_fcount:  # simply remove
                for one in _items[k]:
                    if one.frame is not None:  # delete this!!
                        one.frame.sent.delete_frame(one.frame, 'evt')
                remove_counts.append(_counts)
            elif len(_items[k]) < conf.ignore_fcount:
                ignore_counts.append(_counts)
            elif k in self.ignore_types:
                ignoreT_counts.append(_counts)
            elif _fonto is not None and _fonto.find_frame(k) is None:
                filter_counts.append(_counts)
            else:  # finally adding!
                all_types.append(k)
                all_items.append(_items[k].copy())  # copy the list
                if conf.filter_onto_do_amap:  # re-mapping arg names (for convenience)!
                    _amap = _fonto.find_frame(k).info.get('a_map', {})
                    for _one_item in _items[k]:
                        for _one_arg in _one_item.frame.args:
                            _one_arg.set_label(_amap.get(_one_arg.label, _one_arg.label))
        if any(len(z)>0 for z in [remove_counts, ignore_counts, ignoreT_counts, filter_counts]):
            zlog(f"Remove frame whose counts are less than {conf.remove_fcount}: {len(remove_counts)}/{sum(remove_counts)}")
            zlog(f"Ignore frame whose counts are less than {conf.ignore_fcount}: {len(ignore_counts)}/{sum(ignore_counts)}")
            zlog(f"Ignore frame by type {self.ignore_types}: {len(ignoreT_counts)}/{sum(ignoreT_counts)}")
            zlog(f"Filter frame by {_fonto}: {len(filter_counts)}/{sum(filter_counts)}")
        # --
        # sample?
        if conf.sample_num > 0:
            rng = Random.get_np_generator(conf.sample_seed)  # make sure to create the same data!
            before_all_items = [id(z) for zs in all_items for z in zs]
            rng.shuffle(before_all_items)
            after_all_items = set(before_all_items[:conf.sample_num])
            for ii in range(len(all_items)):  # filtering!
                all_items[ii] = [z for z in all_items[ii] if id(z) in after_all_items]
            # remove zero ones!
            _tmp_all_types, _tmp_all_items = [], []
            for a, b in zip(all_types, all_items):
                if len(b) > 0:
                    _tmp_all_types.append(a)
                    _tmp_all_items.append(b)
            zlog(f"Sample-evts: {len(before_all_items)} -> {len(after_all_items)}, T= {len(all_types)}->{len(_tmp_all_types)}")
            all_types, all_items = _tmp_all_types, _tmp_all_items
        # --
        assert self.all_types is None, "Reusing the same batcher is not allowed!"
        self.all_types, self.all_items = all_types, all_items
        self.rearrange()
        # for special msent ones
        all_msent_items = []
        _msent_n = conf.msent_n
        if _msent_n > 0:
            for vs in all_items:
                for v in vs:
                    _dists = [abs(arg.arg.sent.sid-arg.main.sent.sid) for arg in v.frame.args]
                    if any(d<=2 and d>0 for d in _dists):  # todo(+1): currently assume a window of 5
                        all_msent_items.append(v)
            _msent_n = min(_msent_n, len(all_msent_items))
            zlog(f"Prepare msent_items: {len(all_msent_items)}/{sum(len(z) for z in all_items)}; msent_n={_msent_n}")
        # --
        _n_p0, _n_s, _n_c, _n_p = self.type_budgets
        _k_shot = conf.k_shot
        _repl = conf.type_sample_replace  # allow replace sampling?
        while True:
            # --
            # sample types
            arr_prob, arr_sim, (all_sscore_arrs, sscore_avgarr, sscore_max1arr) = self.rearrange_info
            all_idxes = np.asarray([], dtype=np.int64)
            # plain0: simply according to arr_prob
            if _n_p0 > 0:
                idxes_p0 = _gen.choice(len(arr_prob), _n_p0, replace=_repl, p=arr_prob)  # [N]
                all_idxes = np.concatenate([all_idxes, idxes_p0])
            # special: according to max_sscore
            # todo(+N): sample type first to avoid separating?
            if _n_s > 0:
                idxes_s = self.do_sample(_n_s, sscore_max1arr, all_idxes, _repl, _gen)
                all_idxes = np.concatenate([all_idxes, idxes_s])
            # contrast: pick those similar to previous selected
            if _n_c > 0:
                _mm = (arr_sim[:, all_idxes] > conf.lemma_thresh).sum(-1).astype(np.float32)
                idxes_c = self.do_sample(_n_c, arr_prob*_mm, all_idxes, _repl, _gen)
                all_idxes = np.concatenate([all_idxes, idxes_c])
            # plain: again according to arr_prob
            if _n_p > 0:
                idxes_p = self.do_sample(_n_p, arr_prob, all_idxes, _repl, _gen)
                all_idxes = np.concatenate([all_idxes, idxes_p])
            # --
            # sample items for each
            cur_items = []
            for ii in all_idxes:
                # sample type?
                _pt = sscore_avgarr[ii]  # [1+N]
                _ti = _gen.choice(len(_pt), p=(_pt/_pt.sum()))  # which type?
                # then items
                _pi = all_sscore_arrs[ii][:,_ti]
                _snum = min(_k_shot, (_pi>0).sum())
                _this_items = all_items[ii]
                i_idxes = _gen.choice(len(_pi), _snum, replace=False, p=(_pi/_pi.sum()))  # [K]
                c_items = [_this_items[z] for z in i_idxes]  # [K]
                cur_items.extend(c_items)
            # extra msent ones
            if _msent_n>0:
                mi_idxes = _gen.choice(len(all_msent_items), _msent_n, replace=False)
                cur_items.extend([all_msent_items[z] for z in mi_idxes])
            # --
            # breakpoint()
            yield cur_items
            # --
        # --

    def do_sample(self, num: int, _p, prev_idxes, _repl, _gen):
        if (not _repl) and len(prev_idxes) > 0:
            _p = _p.copy()
            _p[prev_idxes] = 0.
        num = int(min(num, (_p>0).sum()))
        if num <= 0:  # nothing to sample
            return np.asarray([], dtype=np.int64)
        _p = _p / _p.sum()
        ret = _gen.choice(len(_p), num, replace=_repl, p=_p)
        return ret

    def rearrange(self):
        if self.all_types is None:
            return  # not yet initialized!
        # --
        all_types, all_items = self.all_types, self.all_items
        if self.rearrange_info is None:
            arr_prob, arr_sim = None, None
        else:
            arr_prob, arr_sim = self.rearrange_info[:2]
        # --
        # first on overall plain counts & plain probs
        if arr_prob is None:
            arr_prob = self.get_arr_prob(all_types, all_items)
        # also check lemma overlap (build n*n similarity matrix)
        if arr_sim is None:
            arr_sim = self.get_arr_sim(all_types, all_items)
        # then special scores; note: always re-build this since things can be changed!
        sscore_info = self.get_sscore_info(all_types, all_items)
        # --
        self.rearrange_info = (arr_prob, arr_sim, sscore_info)
        return

    def get_arr_prob(self, all_types, all_items):
        conf: FrameBatcherConf = self.conf
        arr_count = np.asarray([len(z) for z in all_items])
        arr_prob = (arr_count ** conf.type_sample_alpha) / (arr_count ** conf.type_sample_alpha).sum()
        df = pd.DataFrame({"type": all_types, "raw_count": arr_count, "raw_probs": arr_count/arr_count.sum(), "probs": arr_prob})
        zlog(f"Prepare batches: {len(all_types)} types, {sum(len(z) for z in all_items)} items, probs=\n{df[:50].to_string()}")
        return arr_prob

    def get_lemma(self, frame):
        if frame is None:
            return None
        _t = frame.type
        if str.isdigit(_t[-2:]):  # pb frames
            return _t.split(".")[0].split("_")[0]
        set_ee_heads(frame.sent)
        _lemma = frame.mention.shead_token.lemma
        return _lemma

    def get_arr_sim(self, all_types, all_items):
        conf: FrameBatcherConf = self.conf
        all_lemmas = Counter()
        for _items in all_items:
            for _item in _items:
                _lemma = self.get_lemma(_item.frame)
                if _lemma is not None:
                    all_lemmas[_lemma.lower()] += 1
        freq_lemmas = all_lemmas.most_common(conf.lemma_vsize)
        zlog(f"Build lemma-voc: {len(freq_lemmas)}({sum(z[1] for z in freq_lemmas)}) "
             f"from {len(all_lemmas)}({sum(all_lemmas.values())})")
        lemma_voc = {zz[0]:ii for ii, zz in enumerate(freq_lemmas)}
        _arr = np.zeros([len(all_items), len(lemma_voc)])  # [Nframe, NLemma]
        for _ii, _items in enumerate(all_items):
            for _item in _items:
                _lemma = self.get_lemma(_item.frame)
                if _lemma is not None:
                    if _lemma.lower() in lemma_voc:
                        _arr[_ii][lemma_voc[_lemma.lower()]] += 1.
        _arr = _arr / (_arr.sum(-1, keepdims=True) + 1e-7)  # normalize as probs
        _arr = _arr / (np.linalg.norm(_arr, axis=-1, keepdims=True) + 1e-7)  # l2 norm
        arr_sim = np.matmul(_arr, _arr.T)  # [Nframe, Nframe]
        _arange = np.arange(len(arr_sim))
        arr_sim[_arange, _arange] = 0.  # no self!
        zlog(f"Build lemma-based sim, {(arr_sim>conf.lemma_thresh).sum()} from {arr_sim.shape}")
        return arr_sim

    def get_sscore_info(self, all_types, all_items):
        conf: FrameBatcherConf = self.conf
        _need_sinfo = self.type_budgets[1] > 0
        # --
        all_sscore_arrs = []  # (*Nf)[Ni, 1+N]
        all_sscore_avgarrs = []  # [Nf, 1+N]
        # first get overall max_len
        _max_len = max(len(_item.frame._cache.get('fextra_margin', []) if _item.frame is not None else [])
                       for _items in all_items for _item in _items)
        _max_len = max(1, _max_len)
        _tmp_m = np.array([1.]+[0.]*(_max_len-1), dtype=np.float32)  # [1+N], note: assume things are all idx0!
        for _ii, _items in enumerate(all_items):
            one_ss = []
            for _item in _items:
                _m = _item.frame._cache.get('fextra_margin', _tmp_m) if _item.frame is not None else _tmp_m
                one_ss.append(_m)
                if _need_sinfo and _m is _tmp_m:
                    zwarn(f"Curr item does not have good scores (simply pad 0.): {_item.frame}")
            one_arr = np.stack(one_ss, axis=0)  # [Ni, 1+N]
            one_arr = np.exp(one_arr / conf.score_tau).clip(min=1.) - 1.  # [Ni, 1+N], clip by 0
            one_arr = one_arr ** conf.score_alpha  # [Ni, 1+N]
            all_sscore_arrs.append(one_arr)
            one_avgarr = one_arr.sum(0) / (conf.score_avgE + (one_arr>0).sum(0))  # [1+N], average over the group!
            all_sscore_avgarrs.append(one_avgarr)
        sscore_avgarr = np.stack(all_sscore_avgarrs, axis=0)  # [Nf, 1+N]
        # [NF], max non-idx0 one or simply 0 if no other dims!
        sscore_max1arr = sscore_avgarr[:,1:].max(-1) if sscore_avgarr.shape[-1]>1 else (sscore_avgarr[:,0]*0.)
        ret = (all_sscore_arrs, sscore_avgarr, sscore_max1arr)
        # breakpoint()
        return ret

# --
# b msp2/tasks/zmtl3/core/run/frame:70
