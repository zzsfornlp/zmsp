#

# another mod for relations with token-pair scoring
# note: final prediction is still based on given frames
# note: mostly following "ZModRel"

__all__ = [
    "ZModRel2Conf", "ZModRel2"
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

@ZModConf.rd('rel2')
class ZModRel2Conf(ZModSbConf):
    def __init__(self):
        super().__init__('bmod2')
        self.remain_toks = True  # overwrite!
        self.frame_mname = "extH"  # basic frame's extractor to judge frame's partial
        # special
        self.alink_no_self = True  # no self-link!
        self.allow_cross = False  # allow cross sent relations (as Tails!)
        # --
        self.scorer_rel = PairScoreConf().direct_update(aff_dim=300, use_biaff=True, _rm_names=['isize', 'osize'])
        # --
        self.label_smooth = 0.  # label smoothing
        self.strg_ratio = SVConf.direct_conf(val=0.)  # use soft target for loss?
        self.strg_thresh = 0.  # need strg's max-prob > this
        self.inf_tau = 1.  # temperature for inference (marginals & argmax)
        self.loss_margin = 0.  # minus score of gold (unary) ones
        self.partial_alpha = 1.  # weight for partial parts
        # --
        self.zrel_debug_print_neg = False
        self.neg_rate = SVConf.direct_conf(val=-1.)  # down-weight neg ones (no effect if <0)
        self.neg_hard_alpha = 0.  # mine hard neg ones (used together with neg_rate)
        self.neg_strg_thr = 1.  # for strg, count score-NIL>this as neg!
        self.lambda_non_pf = 0.  # loss for non-pf pairs
        # --
        self.pred_use_partial = False  # use partial annotations as constraints!
        self.pred_do_strg = False  # output marginals to strg?
        self.pred_strg_hard = False  # hard strg (argmax) rather than soft (marginal)
        self.pred_do_dec = True  # do actual decoding (and write results)
        self.pred_for_cali = ''  # store scores (logit or logm) for calibration!
        self.pred_m_tau = 1.  # tau for m-inference at pred
        self.pred_dec_cons = ''  # use constraints when decoding?

@ZModRel2Conf.conf_rd()
class ZModRel2(ZModSb):
    def __init__(self, conf: ZModRel2Conf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModRel2Conf = self.conf
        assert conf.remain_toks
        # --
        self.vocR, self.vocH, self.vocT = ztask.vpack
        self.osize = len(self.vocR)
        _ssize = self.bout.dim_out_hid()
        self.scorer_rel = conf.scorer_rel.make_node(isize=_ssize, osize=self.osize)
        # --
        self.strg_ratio = self.svs['strg_ratio']
        self.neg_rate = self.svs['neg_rate']
        self.FRAME_KEY_PA = f"{conf.frame_mname}_ispart"  # partial info for the frame!
        self.KEY_PA = f"{self.name}_ispart"  # partial info: whether it is partial? note: explicitly mark NIL rels!
        self.LAB_NIL, self.LAB_UNK = ztask.conf.lab_nil, ztask.conf.lab_unk
        self.IDX_PA, self.IDX_NIL = -1, 0  # special vocab indexes
        self.KEY_STRG = f"{self.name}_strg"  # soft target
        self.KEY_CALI = f"{self.name}_cali"  # for calibration!
        self.pred_dec_cons = ztask.compile_dec_cons(conf.pred_dec_cons, self.vocR)
        # --

    def prep_one_trgs(self, item, toks):
        conf: ZModRel2Conf = self.conf
        ztask = self.ztask
        # --
        center_sent = item.sent
        assert not conf.allow_cross, "Currently no support for cross sent ones!"
        # toks -> sent
        toks = list(toks)
        while len(toks) > 0 and toks[-1] is None:
            toks.pop()  # keep minimal!
        _arr_tmp = np.asarray([tok is not None and tok.sent is center_sent for tok in toks], dtype=np.float32)
        arr_pmask = _arr_tmp[:, None] * _arr_tmp  # [N, N]
        # --
        # locate frame position and filtering
        center_heads = list(ztask.yield_my_frames(center_sent, 'H'))  # [H]
        center_tails = list(ztask.yield_my_frames(center_sent, 'T'))  # [Tc]
        final_heads, final_tails = [], []
        final_infoH, final_infoT = [], []  # [posi, wlen, lab]
        posi_map = {id(tok): ii for ii, tok in enumerate(toks) if tok is not None}
        for one_frames, one_trgF, one_trgI, one_voc in \
                zip([center_heads, center_tails], [final_heads, final_tails],
                    [final_infoH, final_infoT], [self.vocH, self.vocT]):
            for one_frame in one_frames:
                tok0 = one_frame.mention.get_tokens()[0]
                _posi = posi_map.get(id(tok0))
                if _posi is None:
                    zwarn(f"Ignoring truncated frame: {one_frame}")
                else:
                    one_trgF.append(one_frame)
                    one_trgI.append([_posi, one_frame.mention.wlen, (
                        one_voc[ztask.process_label(one_frame.cate_label)] if ztask.process_label(
                            one_frame.label) != self.LAB_NIL else self.IDX_NIL)])  # note: simply reuse names!
        # --
        # final lab
        _is_frame_partial = center_sent.info.get(self.FRAME_KEY_PA, False)  # partial for frames?
        arr_pfmask = np.full_like(arr_pmask, 0.)  # whether pair of frames?
        arr_plab = np.full_like(arr_pmask, (self.IDX_PA if _is_frame_partial else self.IDX_NIL), dtype=np.int32)  # [N, N]
        # note: assuming continous tokens!
        for hii, one_head in enumerate(final_heads):  # head
            if one_head.label == self.LAB_NIL:
                _posi, _wlen = final_infoH[hii][:2]
                arr_plab[_posi:_posi+_wlen] = self.IDX_NIL
        for tii, one_tail in enumerate(final_tails):  # tail
            if one_tail.label == self.LAB_NIL:
                _posi, _wlen = final_infoT[tii][:2]
                arr_plab[:, _posi:_posi+_wlen] = self.IDX_NIL
        for hii, one_head in enumerate(final_heads):  # pairwise
            _posiH, _wlenH = final_infoH[hii][:2]
            is_partial = one_head.info.get(self.KEY_PA, False)  # partial for head!
            amap = {id(alink.arg): alink for alink in one_head.args}
            for tii, one_tail in enumerate(final_tails):
                _posiT, _wlenT = final_infoT[tii][:2]
                if one_head is one_tail and conf.alink_no_self:
                    arr_plab[_posiH:_posiH+_wlenH, _posiT:_posiT+_wlenT] = self.IDX_NIL
                else:
                    arr_pfmask[_posiH:_posiH+_wlenH, _posiT:_posiT+_wlenT] = 1.
                    _alink = amap.get(id(one_tail))
                    if _alink is None:
                        if not is_partial:
                            arr_plab[_posiH:_posiH+_wlenH, _posiT:_posiT+_wlenT] = self.IDX_NIL
                    else:
                        _alink_lab = self.LAB_UNK if _alink.info.get('is_pred', False) else _alink.label  # as UNK if pred
                        _alink_lab = ztask.process_label(_alink_lab)
                        _alink_lab_idx = self.IDX_NIL if (_alink_lab == self.LAB_NIL) else \
                            (self.IDX_PA if (_alink_lab == self.LAB_UNK) else self.vocR[_alink_lab])
                        arr_plab[_posiH:_posiH+_wlenH, _posiT:_posiT+_wlenT] = _alink_lab_idx
        # --
        return final_heads, final_tails, final_infoH, final_infoT, arr_pmask, arr_pfmask, arr_plab

    def prep_tok_map(self, csent, toks):
        _idx0, _idx1 = [], []
        for ii, tt in enumerate(toks):
            if tt is not None and tt.sent is csent:
                _idx0.append(ii)
                _idx1.append(tt.widx)
        arr_i0, arr_i1 = np.asarray(_idx0), np.asarray(_idx1)
        return arr_i0, arr_i1

    def prep_ibatch_trgs(self, ibatch, arr_toks, prep_soft: bool, no_cache=False):
        _alink_no_self = self.conf.alink_no_self
        # --
        _bsize = len(ibatch.items)
        # first collect them!
        _shape = BK.get_shape(arr_toks)
        if prep_soft:  # [*, L, L, V], soft targets
            arr_strg = np.full(_shape + [_shape[-1], len(self.vocR)], 0., dtype=np.float32)
        else:
            arr_strg = None
        all_inputs = [[] for _ in range(7)]
        _key = self.form_cache_key('T')
        no_strg = False
        for bidx, item in enumerate(ibatch.items):
            _cache = item.cache.get(_key)
            if _cache is None or no_cache:
                _cache = self.prep_one_trgs(item, arr_toks[bidx])
                if not no_cache:
                    item.cache[_key] = _cache
            # --
            for ii, zz in enumerate(_cache):
                all_inputs[ii].append(zz)
            if prep_soft:
                csent = item.sent
                _strg = csent.arrs.get(self.KEY_STRG)  # [Ls, Ls, V]
                if _strg is None:
                    no_strg = True
                else:  # [L, L, V]
                    arr_i0, arr_i1 = self.prep_tok_map(csent, arr_toks[bidx])
                    arr_strg[bidx, arr_i0[:, None], arr_i0[None, :]] = _strg[arr_i1[:, None], arr_i1[None, :]]
        # --
        # batch
        arr_heads, _ = DataPadder.batch_2d(all_inputs[0], None)  # [*, Nh]
        arr_tails, _ = DataPadder.batch_2d(all_inputs[1], None)  # [*, Nt]
        t_infoH, _ = DataPadder.batch_3d(all_inputs[2], 0)  # [*, Nh, 3]
        t_infoT, _ = DataPadder.batch_3d(all_inputs[3], 0)  # [*, Nh, 3]
        t_infoH, t_infoT = BK.input_idx(t_infoH), BK.input_idx(t_infoT)
        _len = arr_toks.shape[-1]
        t_pmask, _ = DataPadder.batch_3d(all_inputs[4], 0., max_len1=_len, max_len2=_len)  # [*, L, L]
        t_pfmask, _ = DataPadder.batch_3d(all_inputs[5], 0., max_len1=_len, max_len2=_len)  # [*, L, L]
        t_plab, _ = DataPadder.batch_3d(all_inputs[6], self.IDX_NIL, max_len1=_len, max_len2=_len)  # [*, L, L]
        t_pmask, t_pfmask, t_plab = BK.input_real(t_pmask), BK.input_idx(t_pfmask), BK.input_idx(t_plab)
        t_strg = None
        if prep_soft and not no_strg:
            t_strg = BK.input_real(arr_strg)
        return arr_heads, arr_tails, t_infoH, t_infoT, t_pmask, t_pfmask, t_plab, t_strg

    def _do_aug_score(self, t_score, t_pmask, t_gold, adding: float, inplace: bool):
        t_cons = t_pmask * (t_gold != self.IDX_PA).to(t_pmask)  # [*, Lh, Lm]
        tmp_adding = (t_cons * adding).unsqueeze(-1)  # [*, Lh, Lm, 1]
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
        conf: ZModRel2Conf = self.conf
        # prepare
        t_ids, t_mask, t_sublens, arr_toks = rc.get_cache((self.name, 'input'))  # [*, Ls or Lt]
        arr_heads, arr_tails, t_infoH, t_infoT, t_pmask, t_pfmask, t_plab, t_strg = \
            self.prep_ibatch_trgs(rc.ibatch, arr_toks, prep_soft=(strgR>0.), no_cache=no_cache)  # [*, N?]
        # forward
        t_hid = self._do_forward(rc)  # first forward the encoder: [*, L]
        t_scores = self.scorer_rel(t_hid, t_hid)  # [*, L, L, V]
        return t_scores, t_pmask, t_pfmask, arr_heads, arr_tails, t_plab, t_strg

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModRel2Conf = self.conf
        _strgR = float(self.strg_ratio.value)  # current value!
        _strgTh = 0.  # special one for hard-partial
        if rc.ibatch.dataset is not None:  # potentially read dataset specific one!
            _strgR = float(rc.ibatch.dataset.info.get('strgR', 1.)) * _strgR  # multiply!
            _strgTh = float(rc.ibatch.dataset.info.get('strgTh', 0.))
        # --
        # prepare and forward
        t_score, t_pmask, t_pfmask, arr_heads, arr_tails, t_plab, t_strg = self._my_forward(rc, strgR=_strgR)
        ret_info = {}
        # special mode: partial(hard) with thresh2
        if _strgR > 0. and _strgTh > 0.:
            assert False, "Not implementing this in this mod!"
        # --
        if _strgR == 0.:
            mm = self._do_inf(t_score, t_pmask, t_plab)
        elif _strgR == 1.:
            mm = t_strg
        else:
            mm = self._do_inf(t_score, t_pmask, t_plab)
            mm = mm * (1. - _strgR) + t_strg * _strgR
        # --
        if conf.loss_margin > 0.:  # minus margin for gold ones
            t_score = self._do_aug_score(t_score, t_pmask, t_plab, -conf.loss_margin, inplace=False)
        # note: separate for pf and non-pf ones!
        t_pmask_orig = t_pmask
        t_pmask = t_pmask * t_pfmask  # note: mainly for pair-frames!
        t_weight = t_pmask.clone()
        t_pa = (t_plab == self.IDX_PA) & (t_pmask > 0)
        t_weight[t_pa] = conf.partial_alpha
        if _strgR>0. and conf.strg_thresh > 0.:
            t_ignore = t_pa & (t_strg.max(-1)[0] < conf.strg_thresh)
            t_weight[t_ignore] = 0.001  # set a much smaller value!
        # down-weight?
        _neg_rate, _neg_strg_thr = float(self.neg_rate), float(conf.neg_strg_thr)
        _neg_hard_alpha = float(conf.neg_hard_alpha)
        if _neg_rate >= 0.:
            if _strgR > 0 and _neg_strg_thr < 1:
                t_nil = (mm[...,0] > _neg_strg_thr) & (t_pmask > 0)  # NIL=0
                count_nil = t_nil.sum().to(BK.DEFAULT_FLOAT)
                count_pos = t_pmask.sum() - count_nil  # no substracting t_pa!
            else:
                t_nil = (t_plab == self.IDX_NIL) & (t_pmask > 0)
                count_nil = t_nil.sum().to(BK.DEFAULT_FLOAT)
                count_pos = t_pmask.sum() - count_nil - t_pa.sum().to(count_nil)
            if _neg_hard_alpha > 0.:  # further based on neg_rate
                _inc_num = int((count_pos * _neg_rate * _neg_hard_alpha).item())
                _nil_vs = (-t_score.softmax(-1)[..., 0][t_nil])  # [NNil]
                _, _nil_idxes = _nil_vs.topk(min(_inc_num, BK.get_shape(_nil_vs, -1)), -1)
                _assign_weight = BK.zeros_like(_nil_vs)
                _assign_weight[_nil_idxes] = 1 / _neg_hard_alpha
                t_weight[t_nil] = _assign_weight
            else:
                nw = (count_pos * _neg_rate / count_nil.clamp(min=1.)).clamp(max=1.)
                t_weight[t_nil] = nw
        # simply CE
        if conf.label_smooth > 0.:
            mm = label_smoothing(mm, conf.label_smooth, kldim=1, t_mask=t_pmask)
        loss_v, loss_c = - (t_weight * (mm * t_score.log_softmax(-1)).sum(-1)).sum(), t_weight.sum()
        # --
        count_all = (t_pmask > 0).sum().item()
        count_pa = ((t_plab == self.IDX_PA) & (t_pmask > 0)).sum().item()
        count_nil = ((t_plab == self.IDX_NIL) & (t_pmask > 0)).sum().item()
        count_v = ((t_plab > 0) & (t_pmask > 0)).sum().item()
        ret_info.update({"c_all": count_all, "c_pa": count_pa,
                         "c_nil": count_nil, "c_v": count_v})
        if conf.zrel_debug_print_neg:
            zlog(f"STEP: ALL={count_all}, PA={count_pa}, NIL={count_nil}, V={count_v}")
        one_loss = self.compile_leaf_loss('rel2', loss_v, loss_c)
        all_losses = [one_loss]
        # --
        # non-pf loss
        if conf.lambda_non_pf > 0:
            nonpf_weight = ((t_pmask_orig > 0) & (t_pfmask <= 0)).to(t_weight)
            loss_v2, loss_c2 = - (nonpf_weight * (mm * t_score.log_softmax(-1)).sum(-1)).sum(), nonpf_weight.sum()
            one_loss2 = self.compile_leaf_loss('rel2n', loss_v2, loss_c2, loss_lambda=conf.lambda_non_pf)
            all_losses.append(one_loss2)
        # --
        ret = self.compile_losses(all_losses)
        return (ret, ret_info)

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModRel2Conf = self.conf
        kwargsP = self.process_kwargs(kwargs)  # further process it!
        _pred_for_cali, _pred_use_partial, _pred_do_strg, _pred_strg_hard, _pred_do_dec, _pred_m_tau = conf.obtain_values(["pred_for_cali", "pred_use_partial", "pred_do_strg", "pred_strg_hard", "pred_do_dec", "pred_m_tau"], **kwargsP)
        # --
        # prepare and forward: no cache since frames may change!
        t_score, t_pmask, t_pfmask, arr_heads, arr_tails, t_gold, t_strg = self._my_forward(rc, no_cache=True)
        _, _, _, arr_toks = rc.get_cache((self.name, 'input'))  # [*, L]
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
        arr_pmask0 = BK.get_value(t_pmask)  # [*, L, L]
        arr_prob0 = BK.get_value(t_prob)  # [*, L, L, R]
        arr_pmask = np.full(BK.get_shape(arr_heads) + [arr_tails.shape[-1]], 0.)  # [*, Lh, Lm]
        arr_prob = np.full(BK.get_shape(arr_pmask) + [len(self.vocR)], 0.)  # [*, Lh, Lm, R]
        arr_prob[..., self.IDX_NIL] = 1.  # by default NIL
        # --
        cc = {'r_seq': 0, 'rf_head': 0, 'rf_tail': 0, 'r_arg0': 0, 'r_argV': 0}
        ztask = self.ztask
        _pred_dec_cons = self.pred_dec_cons
        for _sidx, _item in enumerate(rc.ibatch.items):
            _sent = _item.sent
            cc['r_seq'] += 1
            arr_i0, arr_i1 = self.prep_tok_map(_sent, arr_toks[_sidx])  # idxes of the two seq
            tok_map = {id(tok): ii for ii, tok in enumerate(arr_toks[_sidx])}
            # --
            if _pred_do_strg:
                _arr = np.zeros([len(_sent), len(_sent)], dtype=np.float32)  # [L, L]
                _arr[arr_i1[:, None], arr_i1[None, :]] = arr_mm[_sidx, arr_i0[:, None], arr_i0[None, :]]
                _sent.arrs[self.KEY_STRG] = _arr
            # --
            # check or clear the args!
            center_heads = list(ztask.yield_my_frames(_sent, 'H'))  # [H]
            for one_h in center_heads:  # re-read all since some may be truncated!
                for alink in one_h.get_args():
                    if alink.arg.cate in ztask.cateTs:
                        if _pred_do_dec and ((not _pred_use_partial) or alink.info.get('is_pred', False)):  # delete!
                            alink.del_self()
            # --
            cc['rf_head'] += sum([z is not None for z in arr_heads[_sidx]])
            cc['rf_tail'] += sum([z is not None for z in arr_tails[_sidx]])
            # prepare frame-pair probs
            for hidx, one_h in enumerate(arr_heads[_sidx]):
                if one_h is None: continue
                arr_htoks = np.asarray([tok_map[id(t)] for t in one_h.mention.get_tokens() if id(t) in tok_map])
                for tidx, one_t in enumerate(arr_tails[_sidx]):
                    if one_t is None: continue
                    if conf.alink_no_self and one_h is one_t: continue
                    cc['r_arg0'] += 1
                    arr_ttoks = np.asarray([tok_map[id(t)] for t in one_t.mention.get_tokens() if id(t) in tok_map])
                    if len(arr_htoks) == 0 or len(arr_ttoks) == 0:
                        zwarn(f"No valid toks for the pair (truncated?): {one_t} {one_h}")
                        continue
                    arr_pmask[_sidx, hidx, tidx] = 1.
                    _cur_probs = arr_prob0[_sidx, arr_htoks[:, None], arr_ttoks[None, :]]
                    arr_prob[_sidx, hidx, tidx] = _cur_probs.mean(0).mean(0)
                    # breakpoint()
            # --
            if not _pred_do_dec:
                continue  # no actual decoding!
            if hasattr(_pred_dec_cons, 'cons_decode'):  # decode
                _pred_dec_cons.cons_decode(arr_heads[_sidx], arr_tails[_sidx], arr_prob[_sidx], arr_pmask[_sidx])
            else:
                for hidx, one_h in enumerate(arr_heads[_sidx]):
                    if one_h is None: continue
                    for tidx, one_t in enumerate(arr_tails[_sidx]):
                        if one_t is None: continue
                        if arr_pmask[_sidx, hidx, tidx] <= 0: continue
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
                            _alink = one_h.add_arg(one_t, None)
                            _alink.info['is_pred'] = True
                            _alink.set_label(best_lab)  # rewrite label!
        # --
        return cc

# --
# b mspx/tasks/zrel/mod2:
