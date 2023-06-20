#

# assign relations (args)
# note: operate on given frames: can be either gold or predicted

__all__ = [
    "ZTaskRelConf", "ZTaskRel", "ZModRelConf", "ZModRel",
]

from typing import List
import numpy as np
from mspx.data.inst import yield_frames, yield_sents, DataPadder
from mspx.data.vocab import Vocab
from mspx.proc.eval import FrameEvalConf
from mspx.proc.run import SVConf
from mspx.utils import zlog, ConfEntryChoices, zwarn
from mspx.nn import BK, ZTaskConf, ZModConf, ZTaskSbConf, ZTaskSb, ZModSbConf, ZModSb, ZRunCache
from mspx.nn import PairScoreConf, ExtraEmbedConf, extend_idxes, flatten_dims, fulfill_idx_ranges, label_smoothing

# --

@ZTaskConf.rd('rel')
class ZTaskRelConf(ZTaskSbConf):
    def __init__(self):
        super().__init__()
        from .mod2 import ZModRel2Conf
        self.mod = ConfEntryChoices({'': ZModRelConf(), 'v2': ZModRel2Conf()}, '')
        self.eval = FrameEvalConf().direct_update(weight_frame=0., weight_arg=1., bd_arg_lines=50)
        # --
        self.inc_nil_frames = True  # also include nil-frames that have args
        self.lab_nil = "_NIL_"  # special tag for NIL
        self.lab_unk = "_UNK_"  # special tag for UNK (PA)
        self.cateHs = []  # frame cates for Head
        self.cateTs = []  # frame cates for Tail

@ZTaskRelConf.conf_rd()
class ZTaskRel(ZTaskSb):
    def __init__(self, conf: ZTaskRelConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZTaskRelConf = self.conf
        # --
        self.cateHs, self.cateTs = set(conf.cateHs), set(conf.cateTs)
        _old_filterF = self.eval.filter_frame
        self.eval.filter_frame = (lambda f: _old_filterF(f) and f.cate in self.cateHs)  # constrain eval frames
        _old_filterA = self.eval.filter_arg
        self.eval.filter_arg = (lambda a: _old_filterA(a) and a.arg.cate in self.cateTs)  # constrain eval frames
        # --

    # some special processings to handle special marks in AL
    def process_label(self, lab: str):
        sep = '___'
        lab0 = None
        if sep in lab:
            lab0, lab = lab.split(sep, 1)
        if lab.startswith("**"):
            lab = lab[2:]
        if lab0 is not None:
            lab = lab0 + sep + lab
        return lab

    def yield_my_frames(self, insts, type: str):
        _inc_nf = self.conf.inc_nil_frames
        cates = {'H': self.cateHs, 'T': self.cateTs}[type.upper()]
        _ignore_labs = [self.conf.lab_nil, self.conf.lab_unk]  # NIL for all types
        for one in yield_frames(insts, cates=cates):
            _lab = self.process_label(one.label)
            if (_lab not in _ignore_labs) or (_inc_nf and (len(one.args)>0 or len(one.as_args)>0)):
                yield one  # also allow explicit NIL-NIL!

    def build_vocab(self, datasets):
        conf: ZTaskRelConf = self.conf
        _cateHs, _cateTs = self.cateHs, self.cateTs
        _ignore_labs = [conf.lab_nil, conf.lab_unk]
        # --
        vocR, vocH, vocT = [Vocab.build_empty(f"voc{z}_{self.name}") for z in "RHT"]
        for dataset in datasets:
            if dataset.name.startswith('train'):
                for sent in yield_sents(dataset.yield_insts()):
                    for frame in self.yield_my_frames(sent, 'H'):
                        if frame.label in _ignore_labs:
                            continue
                        vocH.feed_one(frame.cate_label)
                        for alink in frame.args:
                            if alink.arg.cate in _cateTs:  # ok!
                                if alink.label in _ignore_labs:
                                    continue
                                vocR.feed_one(alink.label)
                    for frame in self.yield_my_frames(sent, 'T'):
                        if frame.label in _ignore_labs:
                            continue
                        vocT.feed_one(frame.cate_label)
        # --
        ret = (vocR, vocH, vocT)
        for z in ret:
            z.build_sort()
        zlog(f"Finish building for: {ret}")
        return ret
        # --

    def compile_dec_cons(self, name: str, vocR):
        ret = None
        if name:
            if name.startswith("dec:"):  # decoding
                from .cons_dec import ConsDecoder
                ret = ConsDecoder(name.split(":", 1)[-1], self)
            else:  # simple
                from .cons_table import CONS_SET
                _lenR = len(vocR)
                allowed_triples = CONS_SET[name]
                _m = {}  # (head, tail) -> arr
                for h, r, t in allowed_triples:
                    _key = (h, t)
                    _ri = vocR[r]
                    if _key not in _m:
                        _m[_key] = np.array([1.] + [0.] * (_lenR - 1))  # always allow NIL
                    _m[_key][_ri] = 1.
                zlog(f"Compile constraints for {name} with {len(_m)}")
                ret = _m
        return ret

# --

@ZModConf.rd('rel')
class ZModRelConf(ZModSbConf):
    def __init__(self):
        super().__init__('bmod2')
        self.remain_toks = True  # overwrite!
        # special
        self.alink_no_self = True  # no self-link!
        self.allow_cross = False  # allow cross sent relations (as Tails!)
        # --
        self.scorer_rel = PairScoreConf().direct_update(aff_dim=300, use_biaff=True, _rm_names=['isize', 'osize'])
        self.mark_extra = ExtraEmbedConf().direct_update(init_scale=0.5)
        self.emb_strategy = ""  # emb what?
        self.emb_extra = ExtraEmbedConf().direct_update(init_scale=0.5)
        # self.emb_at_input = True  # input or final; note: deprecated
        # --
        self.label_smooth = 0.  # label smoothing
        self.strg_ratio = SVConf.direct_conf(val=0.)  # use soft target for loss?
        self.strg_thresh = 0.  # need strg's max-prob > this
        # self.strg_thresh2 = 0.  # special mode for argmax-marginal + partial
        self.strg_frame_specs = [0., 0.]  # specs for predicted frames used in strg: 0)>this, 1)**this
        self.inf_tau = 1.  # temperature for inference (marginals & argmax)
        self.loss_margin = 0.  # minus score of gold (unary) ones
        self.partial_alpha = 1.  # weight for partial parts
        # --
        self.zrel_debug_print_neg = False
        self.nil_frame_w = 1.  # special (down-)weighting for nil_frame
        self.nil_frame_self = 0.  # how much to mix self-pred for NIL-frame
        # self.neg_rate = -1.  # down-weight neg ones (no effect if <0)
        self.neg_rate = SVConf.direct_conf(val=-1.)  # down-weight neg ones (no effect if <0)
        self.neg_strg_thr = 1.  # for strg, count score-NIL>this as neg!
        # --
        self.pred_use_partial = False  # use partial annotations as constraints!
        self.pred_do_strg = False  # output marginals to strg?
        self.pred_strg_hard = False  # hard strg (argmax) rather than soft (marginal)
        self.pred_do_dec = True  # do actual decoding (and write results)
        self.pred_for_cali = ''  # store scores (logit or logm) for calibration!
        self.pred_m_tau = 1.  # tau for m-inference at pred
        self.pred_dec_cons = ''  # use constraints when decoding?
        self.pred_one_dir = True  # force one-direction prediction for a pair

@ZModRelConf.conf_rd()
class ZModRel(ZModSb):
    def __init__(self, conf: ZModRelConf, ztask: ZTaskRel, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModRelConf = self.conf
        assert conf.remain_toks
        # --
        self.vocR, self.vocH, self.vocT = ztask.vpack
        self.osize = len(self.vocR)
        _ssize = self.bout.dim_out_hid()
        self.scorer_rel = conf.scorer_rel.make_node(isize=_ssize*2, osize=self.osize)  # concat [first,last]
        # --
        _mdim = self.bmod.get_mdim()
        # marks: todo(+N); currently fixed with marking Head!
        # self.flag_marks = [z in conf.mark_strategy for z in 'ht']
        self.e_mark = conf.mark_extra.make_node(mdim=_mdim, osizes=[2])
        # embs
        self.flag_embs = [z in conf.emb_strategy for z in 'ht']
        self.e_emb = conf.emb_extra.make_node(
            mdim=_mdim, osizes=[len(z) for f,z in zip(self.flag_embs, [self.vocH, self.vocT]) if f])
        # --
        self.strg_ratio = self.svs['strg_ratio']
        self.neg_rate = self.svs['neg_rate']
        self.KEY_PA = f"{self.name}_ispart"  # partial info: whether it is partial? note: explicitly mark NIL rels!
        self.LAB_NIL, self.LAB_UNK = ztask.conf.lab_nil, ztask.conf.lab_unk
        self.IDX_PA, self.IDX_NIL = -1, 0  # special vocab indexes
        self.KEY_STRG = f"{self.name}_strg"  # soft target
        self.KEY_CALI = f"{self.name}_cali"  # for calibration!
        self.pred_dec_cons = ztask.compile_dec_cons(conf.pred_dec_cons, self.vocR)
        # --

    def prep_one_trgs(self, item, toks):
        conf: ZModRelConf = self.conf
        ztask: ZTaskRel = self.ztask
        # --
        # first prepare center sent!
        center_sent = item.sent
        center_heads = list(ztask.yield_my_frames(center_sent, 'H'))  # [H]
        center_tails = list(ztask.yield_my_frames(center_sent, 'T'))  # [Tc]
        other_tails = []  # [To]
        hit_sent_id = {id(center_sent)}
        if conf.allow_cross:  # extraly collect others
            for tok in toks:
                if tok is not None:
                    other_sent = tok.sent
                    if id(other_sent) not in hit_sent_id:
                        hit_sent_id.add(other_sent)
                        other_tails.extend(ztask.yield_my_frames(other_sent, 'T'))
        all_tails = center_tails + other_tails  # [Tc + To]
        # --
        # locate frame position and filtering
        final_heads, final_tails = [], []
        final_infoH, final_infoT = [], []  # [posi, wlen, lab]
        posi_map = {id(tok): ii for ii, tok in enumerate(toks) if tok is not None}
        for one_frames, one_trgF, one_trgI, one_voc in \
                zip([center_heads, all_tails], [final_heads, final_tails],
                    [final_infoH, final_infoT], [self.vocH, self.vocT]):
            for one_frame in one_frames:
                tok0 = one_frame.mention.get_tokens()[0]
                _posi = posi_map.get(id(tok0))
                if _posi is None:
                    zwarn(f"Ignoring truncated frame: {one_frame}")
                else:
                    one_trgF.append(one_frame)
                    one_trgI.append([_posi, one_frame.mention.wlen, (one_voc[ztask.process_label(one_frame.cate_label)] if ztask.process_label(one_frame.label)!=self.LAB_NIL else self.IDX_NIL)])  # note: simply reuse names!
        # --
        arr_args = np.full([len(final_heads), len(final_tails)], None, dtype=object)
        arr_labs = np.full_like(arr_args, self.IDX_NIL, dtype=np.int32)  # [H, T]
        tail_map = {id(zz): ii for ii,zz in enumerate(final_tails)}
        for hii, one_head in enumerate(final_heads):
            is_partial = one_head.info.get(self.KEY_PA, False)  # partial for head!
            if is_partial:  # mark partial
                arr_labs[hii] = self.IDX_PA
            # check each arg
            for alink in one_head.args:
                if alink.arg.cate in ztask.cateTs:
                    tii = tail_map.get(id(alink.arg))
                    if tii is None:
                        zwarn(f"Ignoring truncated alink: {alink} ({one_head} -> {alink.arg})")
                        continue
                    if arr_args[hii, tii] is not None:
                        zwarn(f"Multiple rels for the pair: ({one_head}, {alink.arg}): {arr_args[hii, tii]} {alink}")
                    arr_args[hii, tii] = alink
                    _alink_lab = self.LAB_UNK if alink.info.get('is_pred', False) else alink.label  # as UNK if pred
                    _alink_lab = ztask.process_label(_alink_lab)
                    arr_labs[hii, tii] = self.IDX_NIL if (_alink_lab == self.LAB_NIL) else \
                        (self.IDX_PA if (_alink_lab == self.LAB_UNK) else self.vocR[_alink_lab])
        # --
        return final_heads, final_tails, final_infoH, final_infoT, arr_args, arr_labs

    def prep_ibatch_trgs(self, ibatch, arr_toks, prep_soft: bool, no_cache=False):
        _alink_no_self = self.conf.alink_no_self
        # --
        _bsize = len(ibatch.items)
        # first collect them!
        all_inputs = [[] for _ in range(6)]
        _key = self.form_cache_key('T')
        for bidx, item in enumerate(ibatch.items):
            _cache = item.cache.get(_key)
            if _cache is None or no_cache:
                _cache = self.prep_one_trgs(item, arr_toks[bidx])
                if not no_cache:
                    item.cache[_key] = _cache
            # --
            for ii, zz in enumerate(_cache):
                all_inputs[ii].append(zz)
        # --
        # batch
        arr_heads, arr_mH = DataPadder.batch_2d(all_inputs[0], None, ret_mask=True)  # [*, Nh]
        arr_tails, arr_mT = DataPadder.batch_2d(all_inputs[1], None, ret_mask=True)  # [*, Nt]
        t_mH, t_mT = BK.input_real(arr_mH), BK.input_real(arr_mT)
        t_infoH, _ = DataPadder.batch_3d(all_inputs[2], 0)  # [*, Nh, 3]
        t_infoT, _ = DataPadder.batch_3d(all_inputs[3], 0)  # [*, Nh, 3]
        t_infoH, t_infoT = BK.input_idx(t_infoH), BK.input_idx(t_infoT)
        _a_shape = BK.get_shape(t_mH) + [BK.get_shape(t_mT, -1)]
        arr_args, arr_rels = np.full(_a_shape, None, dtype=object), np.full(_a_shape, self.IDX_NIL)
        for bidx in range(_bsize):
            _s1, _s2 = all_inputs[4][bidx].shape
            arr_args[bidx, :_s1, :_s2] = all_inputs[4][bidx]
            arr_rels[bidx, :_s1, :_s2] = all_inputs[5][bidx]
        t_rels = BK.input_idx(arr_rels)
        t_strg = None
        if prep_soft:  # must link and score every pair!
            no_strg = False
            arr_strg = np.full(_a_shape + [len(self.vocR)], 0., dtype=np.float32)
            for bidx in range(_bsize):
                for hii, hitem in enumerate(arr_heads[bidx]):
                    if hitem is None: continue
                    for tii, titem in enumerate(arr_tails[bidx]):
                        if titem is None: continue
                        if _alink_no_self and hitem is titem: continue  # ignore self-link!
                        alink = arr_args[bidx, hii, tii]
                        if alink is None or self.KEY_STRG not in alink.arrs:
                            # note: since we simply use CE finally, all 0 will just lead to zero loss!
                            zwarn(f"Cannot find strg for {hitem} {titem} {alink}")
                            pass
                        else:
                            arr_strg[bidx, hii, tii] = alink.arrs[self.KEY_STRG]
            if not no_strg:
                t_strg = BK.input_real(arr_strg)
        # --
        return arr_heads, arr_tails, t_mH, t_mT, t_infoH, t_infoT, t_rels, t_strg

    def _do_aug_score(self, t_score, t_pmask, t_gold, adding: float, inplace: bool):
        t_cons = t_pmask * (t_gold != self.IDX_PA).to(t_pmask)  # [*, Nh, Nt]
        tmp_adding = (t_cons * adding).unsqueeze(-1)  # [*, L, 1]
        if inplace:
            t_score.scatter_add_(-1, t_gold.clamp(min=0).unsqueeze(-1), tmp_adding)
            return t_score
        else:
            ret = t_score.scatter_add(-1, t_gold.clamp(min=0).unsqueeze(-1), tmp_adding)
            return ret

    def _do_inf(self, t_score, t_pmask, t_gold=None, do_m=True, do_hard_m=False):
        ADDING = 100.  # should be enough!
        _tau = self.conf.inf_tau
        with BK.no_grad_env():
            t_score = t_score / _tau
            if t_gold is not None:  # force it!
                t_score = self._do_aug_score(t_score, t_pmask, t_gold, ADDING, inplace=True)
            # --
            is_zero_shape = BK.is_zero_shape(t_score)
            if (not do_m) or do_hard_m:
                ret = t_score.sum(-1).to(BK.DEFAULT_INT) if is_zero_shape else t_score.max(-1)[1]  # [*, Lh, Lm]
                if do_m:  # [*, Lh, Lm, V]
                    ret = extend_idxes(ret, self.osize)
            else:  # [*, Lh, Lm, V]
                ret = t_score.softmax(-1)
        return ret

    def _my_forward(self, rc, strgR=0., no_cache=False):
        conf: ZModRelConf = self.conf
        # prepare
        t_ids, t_mask, t_sublens, arr_toks = rc.get_cache((self.name, 'input'))  # [*, Ls or Lt]
        arr_heads, arr_tails, t_mH, t_mT, t_infoH, t_infoT, t_rels, t_strg = \
            self.prep_ibatch_trgs(rc.ibatch, arr_toks, prep_soft=(strgR>0.), no_cache=no_cache)  # [*, N?]
        t_pmask = t_mH.unsqueeze(-1) * t_mT.unsqueeze(-2)  # [*, Nh, Nt]
        if conf.alink_no_self:
            if not BK.is_zero_shape(t_pmask):
                t_noself = BK.input_real([[[int(a1 is not b1) for b1 in b] for a1 in a]
                                          for a,b in zip(arr_heads, arr_tails)])
                t_pmask = t_pmask * t_noself  # [*, Nh, Nt]
        _bsize, _lssize = BK.get_shape(t_ids)
        _ltsize = BK.get_shape(arr_toks, -1)
        # check zero
        if BK.is_zero_shape(t_pmask):  # nothing to score
            t_scores = BK.zeros(BK.get_shape(t_pmask) + [self.osize])  # [*, Nh, Nt, V]
        else:
            # explicit forward as in _do_forward
            _input_name = conf.input_name
            if _input_name:  # use other inputs
                t_in0 = rc.get_cache(_input_name)
            else:  # use own embedding
                t_in0 = self.bmod.forward_emb(t_ids, forw_full=True)
            # mapping idxes (subtok & tok)
            t_t2s = (t_sublens.cumsum(-1) - t_sublens).clamp(min=0, max=_lssize-1)  # [*, Lt]
            t_s2t = (BK.arange_idx(_lssize) >= t_sublens.cumsum(-1).unsqueeze(-1)).sum(-2).clamp(max=_ltsize-1)  # [*, Ls]
            # mark head: extend for heads!
            _ti_mH = t_mH.to(BK.DEFAULT_INT)
            t_orig2ext = (flatten_dims(_ti_mH, -2, None).cumsum(-1).view(t_mH.shape) - 1) * _ti_mH  # [*, Nh]
            t_ext2bidx = (BK.arange_idx(_bsize).unsqueeze(-1) * _ti_mH)[_ti_mH>0]  # [**], orig bidx
            t2_in0, t2_mask, t2_t2s, t2_s2t, t2_infoH, t2_infoT = \
                [z[t_ext2bidx] for z in [t_in0, t_mask, t_t2s, t_s2t, t_infoH, t_infoT]]  # [**, ...]
            # --
            _arange_t = BK.arange_idx(BK.get_shape(t_ext2bidx, 0)).unsqueeze(-1)  # [**, 1]
            ts_infoH = t_infoH[_ti_mH>0]  # [**, 3], note: use original one to separate things!
            t2_tmark = fulfill_idx_ranges(ts_infoH[..., 0:1], ts_infoH[..., 1:2], 1, _ltsize)  # [**, Lt]
            t2_smark = t2_tmark[_arange_t, t2_s2t]  # [**, Ls]
            t2_in0 = self.e_mark(t2_in0, [t2_smark])  # [**, Ls, D], add marker!
            # emb
            t2_eidxes = []
            for _flag, _t2_info in zip(self.flag_embs, [t2_infoH, t2_infoT]):
                if _flag:
                    _t2_eidxT = fulfill_idx_ranges(_t2_info[..., 0], _t2_info[..., 1], _t2_info[..., 2], _ltsize)
                    _t2_eidxS = _t2_eidxT[_arange_t, t2_s2t]  # [**, Ls]
                    t2_eidxes.append(_t2_eidxS)
            if len(t2_eidxes) > 0:
                t2_in0 = self.e_emb(t2_in0, t2_eidxes)  # [**, Ls, D], add emb!
            # forward
            bout = self.bmod.forward_enc(None, t_mask=t2_mask, t_ihid=t2_in0)
            t2_hid0 = self.bout(bout, rc=rc, sublen_t=t_sublens[t_ext2bidx], ext_sidx=t_ext2bidx)['ET']  # [**, Lt, D]
            t2_hid0 = self.pass_gnn(t2_hid0, rc, arr_toks, t_ext2bidx)  # pass gnn if needed
            t2_reprHs = [t2_hid0[_arange_t, ts_infoH[..., 0].unsqueeze(-1)],
                         t2_hid0[_arange_t, (ts_infoH[..., :2].sum(-1)-1).clamp(min=0,max=_ltsize-1).unsqueeze(-1)]]  # [**, 1, D]
            t2_reprTs = [t2_hid0[_arange_t, t2_infoT[..., 0]],
                         t2_hid0[_arange_t, (t2_infoT[..., :2].sum(-1)-1).clamp(min=0,max=_ltsize-1)]]  # [**, Nt, D]
            t2_scores = self.scorer_rel(BK.concat(t2_reprHs, -1), BK.concat(t2_reprTs, -1)).squeeze(-3)  # [**, Nt, V]
            t_scores = t2_scores[t_orig2ext]  # [*, Nh, Nt, V]
        # --
        return t_scores, t_pmask, arr_heads, arr_tails, t_rels, t_strg

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModRelConf = self.conf
        _strgR = float(self.strg_ratio.value)  # current value!
        _strgTh = 0.  # special one for hard-partial
        if rc.ibatch.dataset is not None:  # potentially read dataset specific one!
            _strgR = float(rc.ibatch.dataset.info.get('strgR', 1.)) * _strgR  # multiply!
            _strgTh = float(rc.ibatch.dataset.info.get('strgTh', 0.))
        _fspec_thr, _fspec_alpha = conf.strg_frame_specs
        # --
        # prepare and forward
        t_score, t_pmask, arr_heads, arr_tails, t_rels, t_strg = self._my_forward(rc, strgR=_strgR)
        ret_info = {}
        # --
        if _strgR>0. and _strgTh > 0.:  # special mode: partial(hard) with thresh2
            _max_vs, _max_is = t_strg.max(-1)  # [**, L]
            t_drop = (t_pmask > 0) & (_max_vs < _strgTh)
            _max_is[t_drop] = self.IDX_PA
            _old_strg = t_strg
            t_strg = self._do_inf(t_score, t_pmask, _max_is)
            _i0, _i1 = t_pmask.sum().item(), t_drop.sum().item()
            ret_info.update({'all_rels': _i0, 'all_rstrgV': _i0 - _i1})
        # --
        # special treatment for NIL-args from NIL-frame (note: simply reuse names!)
        _nf_self = conf.nil_frame_self
        _nf_head, _nf_tail = [BK.input_real([0. if (z is None or z.label != self.LAB_NIL) else 1. for z in _arr.flatten()]).view(_arr.shape) for _arr in [arr_heads, arr_tails]]  # [*, N?]
        _nf_args = ((_nf_head.unsqueeze(-1) + _nf_tail.unsqueeze(-2))>0).to(t_pmask) * t_pmask  # [*, Nh, Nt], using OR!
        _nf_m = (_nf_self * _nf_args).unsqueeze(-1)  # [*, Nh, Nt, 1]
        # --
        if _strgR == 0.:
            mm = self._do_inf(t_score, t_pmask, t_rels)
            if _nf_self > 0:
                mm = (1-_nf_m) * mm + _nf_m * self._do_inf(t_score, t_pmask)
        elif _strgR == 1.:
            mm = t_strg
        else:
            mm = self._do_inf(t_score, t_pmask, t_rels)
            if _nf_self > 0:
                mm = (1-_nf_m) * mm + _nf_m * self._do_inf(t_score, t_pmask)
            mm = mm * (1. - _strgR) + t_strg * _strgR
        # --
        if conf.loss_margin > 0.:  # minus margin for gold ones
            t_score = self._do_aug_score(t_score, t_pmask, t_rels, -conf.loss_margin, inplace=False)
        t_weight = t_pmask.clone()
        t_pa = (t_rels == self.IDX_PA) & (t_pmask > 0)
        t_weight[t_pa] = conf.partial_alpha
        if _strgR>0. and conf.strg_thresh > 0.:
            t_ignore = t_pa & (t_strg.max(-1)[0] < conf.strg_thresh)
            t_weight[t_ignore] = 0.001  # set a much smaller value!
        # down-weight?
        _neg_rate, _neg_strg_thr = float(self.neg_rate), float(conf.neg_strg_thr)
        if _neg_rate >= 0.:
            if _strgR > 0 and _neg_strg_thr < 1:
                t_nil = (mm[...,0] > _neg_strg_thr) & (t_pmask > 0)  # NIL=0
                count_nil = t_nil.sum().to(BK.DEFAULT_FLOAT)
                count_pos = t_pmask.sum() - count_nil  # no substracting t_pa!
            else:
                t_nil = (t_rels == self.IDX_NIL) & (t_pmask > 0)
                count_nil = t_nil.sum().to(BK.DEFAULT_FLOAT)
                count_pos = t_pmask.sum() - count_nil - t_pa.sum().to(count_nil)
            nw = (count_pos * _neg_rate / count_nil.clamp(min=1.)).clamp(max=1.)
            t_weight[t_nil] = nw
        # nil-frame
        if conf.nil_frame_w < 1.:
            t_weight[_nf_args>0] *= conf.nil_frame_w  # further down-weighting
        # incorporate frame prob
        if _fspec_thr > 0. or _fspec_alpha != 0.:
            _fp_head = BK.input_real([(0. if z is None else (1. if not z.info.get('is_pred', False) else z.info['pred_prob'])) for z in arr_heads.flatten()]).view(arr_heads.shape)  # [*, Nh]
            _fp_tail = BK.input_real([(0. if z is None else (1. if not z.info.get('is_pred', False) else z.info['pred_prob'])) for z in arr_tails.flatten()]).view(arr_tails.shape)  # [*, Nt]
            _fp = _fp_head.unsqueeze(-1) * _fp_tail.unsqueeze(-2)  # [*, Nh, Nt]
            _fpV = (_fp >= _fspec_thr).to(t_weight)  # [*, Nh, Nt]: valid according to frame-prob
            t_weight = t_weight * _fpV * (_fp ** _fspec_alpha)
            ret_info.update({'all_rzfpV': _fpV.sum().item()})
        # simply CE
        if conf.label_smooth > 0.:
            mm = label_smoothing(mm, conf.label_smooth, kldim=1, t_mask=t_pmask)
        loss_v, loss_c = - (t_weight * (mm * t_score.log_softmax(-1)).sum(-1)).sum(), t_weight.sum()
        # --
        count_all = (t_pmask > 0).sum().item()
        count_pa = ((t_rels == self.IDX_PA) & (t_pmask > 0)).sum().item()
        count_nil = ((t_rels == self.IDX_NIL) & (t_pmask > 0)).sum().item()
        count_nilF = _nf_args.sum().item()
        count_v = ((t_rels > 0) & (t_pmask > 0)).sum().item()
        ret_info.update({"c_all": count_all, "c_pa": count_pa,
                         "c_nil": count_nil, "c_nilF": count_nilF, "c_v": count_v})
        if conf.zrel_debug_print_neg:
            zlog(f"STEP: ALL={count_all}, PA={count_pa}, NIL={count_nil}, NIL_F={count_nilF}, V={count_v}")
        # --
        one_loss = self.compile_leaf_loss('rel', loss_v, loss_c)
        ret = self.compile_losses([one_loss])
        return (ret, ret_info)

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModRelConf = self.conf
        kwargsP = self.process_kwargs(kwargs)  # further process it!
        _pred_for_cali, _pred_use_partial, _pred_do_strg, _pred_strg_hard, _pred_do_dec, _pred_m_tau, _pred_one_dir = conf.obtain_values(["pred_for_cali", "pred_use_partial", "pred_do_strg", "pred_strg_hard", "pred_do_dec", "pred_m_tau", "pred_one_dir"], **kwargsP)
        # --
        # prepare and forward: no cache since frames may change!
        t_score, t_pmask, arr_heads, arr_tails, t_gold, t_strg = self._my_forward(rc, no_cache=True)
        t_gold_backup = t_gold
        if not _pred_use_partial:
            t_gold = None
        # --
        if _pred_do_strg or _pred_for_cali:
            mm = self._do_inf(t_score, t_pmask, t_gold, do_hard_m=_pred_strg_hard)
            if _pred_m_tau != 1.:
                mm = (mm.log() / _pred_m_tau).softmax(-1)
            arr_mm = BK.get_value(mm)  # [*, Lh, Lm, V]
            # --
            if _pred_for_cali:  # simply flatten and put to sents
                if _pred_for_cali == 'm':  # marginal-prob
                    _zscore = mm
                elif _pred_for_cali == 'logm':  # log-m-prob
                    _zscore = mm.log()
                else:
                    _zscore = t_score
                t_comb = BK.concat([t_gold_backup.unsqueeze(-1).to(_zscore), _zscore], -1)  # [*,Lh,Lm,V] compact!
                for _sidx, _item in enumerate(rc.ibatch.items):
                    _sent = _item.sent
                    _sent.arrs[self.KEY_CALI] = BK.get_value(t_comb[_sidx][t_pmask[_sidx] > 0])  # [??, V]
            # --
        else:
            arr_mm = None
        t_prob = self._do_inf(t_score, t_pmask, t_gold, do_m=True)  # still get prob first!
        arr_pmask = BK.get_value(t_pmask)  # [*, Lh, Lm]
        arr_prob = BK.get_value(t_prob)  # [*, Lh, Lm, R]
        # --
        cc = {'r_seq': 0, 'rf_head': 0, 'rf_tail': 0, 'r_arg0': 0, 'r_argV': 0, 'r_argM': 0, 'r_argM1': 0}
        ztask = self.ztask
        _pred_dec_cons = self.pred_dec_cons
        for _sidx, _item in enumerate(rc.ibatch.items):
            _sent = _item.sent
            cc['r_seq'] += 1
            # --
            # check or clear the args!
            existing_map = {}  # already existing links: (idH, idT) -> ArgLink
            center_heads = list(ztask.yield_my_frames(_sent, 'H'))  # [H]
            for one_h in center_heads:  # re-read all since some may be truncated!
                for alink in one_h.get_args():
                    if alink.arg.cate in ztask.cateTs:
                        if _pred_do_dec and ((not _pred_use_partial) or alink.info.get('is_pred', False)):  # delete!
                            alink.del_self()
                        else:  # keep!
                            existing_map[(id(one_h), id(alink.arg))] = alink
            # --
            cc['rf_head'] += sum([z is not None for z in arr_heads[_sidx]])
            cc['rf_tail'] += sum([z is not None for z in arr_tails[_sidx]])
            for hidx, one_h in enumerate(arr_heads[_sidx]):
                if one_h is None: continue
                for tidx, one_t in enumerate(arr_tails[_sidx]):
                    if one_t is None: continue
                    if arr_pmask[_sidx, hidx, tidx] <= 0: continue
                    cc['r_arg0'] += 1
                    existing_alink = existing_map.get((id(one_h), id(one_t)))
                    if _pred_do_strg:  # must add one to hold the results
                        if existing_alink is None:
                            existing_alink = one_h.add_arg(one_t, self.LAB_UNK)  # mark as UNK (PA)
                            existing_alink.info['is_pred'] = True
                            existing_map[(id(one_h), id(one_t))] = existing_alink
                        existing_alink.arrs[self.KEY_STRG] = arr_mm[_sidx, hidx, tidx].copy()  # [V]
            # --
            if not _pred_do_dec:
                continue  # no actual decoding!
            if hasattr(_pred_dec_cons, 'cons_decode'):  # decode
                # _pred_dec_cons.cons_decode(arr_heads[_sidx], arr_tails[_sidx], arr_prob[_sidx], arr_pmask[_sidx])
                raise RuntimeError("Not using this mode!")
            else:
                for hidx, one_h in enumerate(arr_heads[_sidx]):
                    if one_h is None: continue
                    for tidx, one_t in enumerate(arr_tails[_sidx]):
                        if one_t is None: continue
                        if arr_pmask[_sidx, hidx, tidx] <= 0: continue
                        existing_alink = existing_map.get((id(one_h), id(one_t)))
                        _one_parr = arr_prob[_sidx, hidx, tidx].copy()
                        if isinstance(_pred_dec_cons, dict):  # simple constraints
                            _key = (one_h.label, one_t.label)
                            if _key not in _pred_dec_cons:
                                _one_parr[1:] = 0.  # only NIL!
                            else:
                                _one_parr *= _pred_dec_cons[_key]
                        best_lab_idx = _one_parr.argmax().item()
                        best_lab = self.vocR.idx2word(best_lab_idx)
                        if best_lab_idx != self.IDX_NIL:
                            cc['r_argV'] += 1  # valid one!
                            if existing_alink is None:
                                existing_alink = one_h.add_arg(one_t, None)
                                existing_alink.info['is_pred'] = True
                            existing_alink.set_label(best_lab)  # rewrite label!
                        elif existing_alink is not None:
                            existing_alink.set_label(self.LAB_NIL)  # rewrite label!
                        if existing_alink is not None:
                            existing_alink.cache['lab_prob'] = _one_parr[best_lab_idx]
                # --
                if _pred_one_dir:
                    for hidx, one_h in enumerate(arr_heads[_sidx]):
                        if one_h is None: continue
                        for tidx, one_t in enumerate(arr_tails[_sidx]):
                            if one_t is None: continue
                            if arr_pmask[_sidx, hidx, tidx] <= 0: continue
                            IGNORE_LABS = [self.LAB_UNK, self.LAB_NIL]
                            alink1 = [z for z in one_h.args if z.arg is one_t and z.label not in IGNORE_LABS]
                            alink2 = [z for z in one_t.args if z.arg is one_h and z.label not in IGNORE_LABS]
                            if len(alink1) > 0 and len(alink2) > 0:
                                cc['r_argM'] += 1
                                # keep the largest scored one!
                                alink_keep = max(alink1+alink2, key=(lambda x: x.cache.get('lab_prob', -1)))
                                for _alink in alink1 + alink2:
                                    if _alink is not alink_keep:
                                        _alink.set_label(self.LAB_NIL)
                                        cc['r_argM1'] += 1
                # --
        # --
        return cc

# --
# b mspx/tasks/zrel/mod:
