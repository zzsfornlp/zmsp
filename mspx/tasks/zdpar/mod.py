#

# dependency parsing
# note: almost following "m_seq.py"

__all__ = [
    "ZTaskDparConf", "ZTaskDpar", "ZModDparConf", "ZModDpar",
]

import numpy as np
from mspx.data.inst import yield_sents
from mspx.data.vocab import Vocab
from mspx.proc.eval import DparEvalConf
from mspx.utils import zlog, ConfEntryChoices, zwarn
from mspx.nn import BK, ZTaskConf, ZModConf, ZTaskSbConf, ZTaskSb, ZModSbConf, ZModSb, ZRunCache
from mspx.nn import DparLabelerConf, extend_idxes, flatten_dims, label_smoothing
from mspx.proc.run import SVConf

# --

@ZTaskConf.rd('dpar')
class ZTaskDparConf(ZTaskSbConf):
    def __init__(self):
        super().__init__()
        self.mod = ZModDparConf()
        self.eval = DparEvalConf()
        # --
        self.use_l1 = False  # only use L1?
        self.lab_unk = '_UNK_'

    @property
    def lab_level(self):
        return 1 if self.use_l1 else None

@ZTaskDparConf.conf_rd()
class ZTaskDpar(ZTaskSb):
    def __init__(self, conf: ZTaskDparConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZTaskDparConf = self.conf
        # --

    def build_vocab(self, datasets):
        conf: ZTaskDparConf = self.conf
        # --
        voc = Vocab.build_empty(f"voc_{self.name}")
        _level = conf.lab_level
        for dataset in datasets:
            if dataset.name.startswith('train'):
                for sent in yield_sents(dataset.yield_insts()):  # note: ignore UNK
                    voc.feed_iter([z for z in sent.tree_dep.get_labels(level=_level) if z != conf.lab_unk])
        voc.build_sort()
        zlog(f"Finish building for: {voc}")
        return (voc, )
        # --

# --

@ZModConf.rd('dpar')
class ZModDparConf(ZModSbConf):
    def __init__(self):
        super().__init__('bmod2')
        self.remain_toks = True  # overwrite!
        # --
        self.dpar_lab = DparLabelerConf()
        self.dpar_unlab = False  # unlabeled mode!
        self.loss_unlab = 0.  # if >0, further put loss upon edge scores
        self.label_smooth = 0.  # label smoothing
        # related with store att
        self.store_att = False  # store score?
        self.store_att_gratio = SVConf.direct_conf(val=1.)  # use pred or gold?
        self.store_att_gratioT = 0.  # test time?
        # --
        self.strg_ratio = SVConf.direct_conf(val=0.)  # use soft target for loss?
        self.strg_thresh = 0.  # need strg's max-prob > this
        self.inf_tau = 1.  # temperature for inference (marginals & argmax)
        self.loss_margin = 0.  # minus score of gold (unary) ones
        self.partial_alpha = 1.  # weight for partial parts
        # for predicting
        self.pred_use_partial = False  # use partial annotations as constraints!
        self.pred_do_strg = False  # output marginals to strg?
        self.pred_strg_hard = False  # hard strg (argmax) rather than soft (marginal)
        self.pred_do_dec = True  # do actual decoding (and write results)
        self.pred_cons_mode = ''  # '' means same as main, otherwise ...
        self.pred_for_cali = ''  # store scores (logit or logm) for calibration!
        self.pred_m_tau = 1.  # tau for m-inference at pred

@ZModDparConf.conf_rd()
class ZModDpar(ZModSb):
    def __init__(self, conf: ZModDparConf, ztask: ZTaskDpar, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModDparConf = self.conf
        assert conf.remain_toks
        assert conf.partial_alpha > 0
        self.voc = ztask.vpack[0]
        self.osize = 1 if conf.dpar_unlab else len(self.voc)
        _ssize = self.bout.dim_out_hid()
        self.dpar_lab = conf.dpar_lab.make_node(isize=_ssize, csize=self.osize)
        self.strg_ratio = self.svs['strg_ratio']
        self.store_att_gratio = self.svs['store_att_gratio']
        # --
        self.IDX_PA = -1  # special one for UNK head!
        self.LAB_PA = ztask.conf.lab_unk
        self.KEY_STRG = f"{self.name}_strg"  # soft target
        self.KEY_CALI = f"{self.name}_cali"  # label + raw scores
        # --

    # --
    # prepare target labels; note: toks may have extra paddings, but does not matter ...
    def get_tok_map(self, sent, toks, df=-1):
        ret0 = [df] * len(sent)  # sent -> toks
        ret1 = [df] * len(toks)  # toks -> sent
        for ii, tt in enumerate(toks):
            if tt is not None and tt.sent is sent:
                ret0[tt.widx] = ii
                ret1[ii] = tt.widx
        return ret0, ret1

    def prep_one_trgs(self, item, toks):
        _level = self.ztask.conf.lab_level
        # first prepare sent!
        center_sent = item.sent
        center_tree = center_sent.tree_dep
        if center_tree is None:  # no tree!
            g_head = [self.IDX_PA] * len(center_sent)
            g_lab = [self.IDX_PA] * len(center_sent)
        else:
            g_head = center_tree.seq_head.vals.copy()  # [Ls]
            g_lab = [(self.voc[z] if z != self.LAB_PA else self.IDX_PA) for z in center_tree.get_labels(_level)]  # [Ls]
        # --
        # note: only consider center sent as target!
        assert toks[0] is None, "The first one should be [CLS]!"  # note: use [0](CLS) as AROOT
        one_mapR, one_map = self.get_tok_map(center_sent, toks, df=-1)
        one_map = [z+1 for z in one_map]  # [L], mapping to (1+widx); >0 for valid
        # one_mapR = ...  # [Ls], mapping to curr; >0 for valid
        # re-indexing!
        one_goldH = [self.IDX_PA] * len(toks)  # by default PA-UNK
        one_goldL = [self.IDX_PA] * len(toks)
        for curr_ii, curr_tok in enumerate(toks):
            widx = one_map[curr_ii] - 1  # offset
            if widx >= 0:  # translate head idx!
                one_goldL[curr_ii] = g_lab[widx]
                orig_head = g_head[widx]
                if orig_head == self.IDX_PA:  # PA-UNK
                    pass
                elif orig_head == 0:
                    one_goldH[curr_ii] = 0  # root to root
                else:
                    assert orig_head >= 1
                    new_head = one_mapR[orig_head-1]
                    if new_head > 0:
                        one_goldH[curr_ii] = new_head
                    else:  # let it be UNK ...
                        zwarn(f"Head not found, maybe truncated for {center_sent}!")
        return one_map, one_goldH, one_goldL

    def prep_ibatch_trgs(self, ibatch, arr_toks, prep_soft: bool, no_cache=False):
        # first collect them!
        _shape = BK.get_shape(arr_toks)
        _voc = self.voc
        _lenV = len(_voc)  # output size!
        arr_tmask = np.full(_shape, 0., dtype=np.float32)  # [*, L], valid toks or AROOT
        arr_gold_head = np.full(_shape, 0, dtype=np.int32)  # [*, L] gold head, INT_PA as partial
        arr_gold_label = np.full(_shape, 0, dtype=np.int32)  # [*, L] gold label, INT_PA as partial
        if prep_soft:  # [*, L, L, V], soft targets
            arr_strg = np.full(_shape + [_shape[-1], _lenV], 0., dtype=np.float32)
        else:
            arr_strg = None
        no_strg = False  # there are insts without strg!
        # --
        _key = self.form_cache_key('T')
        for bidx, item in enumerate(ibatch.items):
            csent = item.sent
            _cache = item.cache.get(_key)
            if _cache is None or no_cache:
                _cache = self.prep_one_trgs(item, arr_toks[bidx])
                if not no_cache:
                    item.cache[_key] = _cache
            # --
            one_map, one_goldH, one_goldL = _cache
            _size = min(len(one_map), _shape[-1])  # there can be padding differences!
            arr_tmask[bidx, :_size] = [float(z>0) for z in one_map[:_size]]  # valid if 1+widx>=0
            arr_gold_head[bidx, :_size] = one_goldH[:_size]
            arr_gold_label[bidx, :_size] = one_goldL[:_size]
            if prep_soft:
                _strg = csent.arrs.get(self.KEY_STRG)  # [1+Ls, 1+Ls, V]
                if _strg is None:
                    no_strg = True
                else:  # [L, L, V]
                    _tmp_arr = np.asarray(one_map[:_size]).clip(min=0)
                    # note: may assign wrong strg, need dpar_mask later!!
                    arr_strg[bidx, :_size, :_size] = _strg[_tmp_arr[:, None], _tmp_arr[None, :]]
        # --
        # return
        arr_tmask[:, 0] = 1.  # AROOT!
        t_tmask = BK.input_real(arr_tmask)
        t_goldH, t_goldL = BK.input_idx(arr_gold_head), BK.input_idx(arr_gold_label)
        if prep_soft and (not no_strg):  # prepare soft and all has strg!
            dpar_mask = self.dpar_lab.out.make_dpar_mask(t_tmask).unsqueeze(-1)  # [*, L, L, 1]
            t_strg = (BK.input_real(arr_strg) + 1e-8) * dpar_mask  # [*, Lm, Lh, V], note: mask out invalid ones!
            _tmp_n = t_strg.sum(-1).sum(-1).clamp(min=1e-8)  # [*, Lm]
            t_strg /= _tmp_n.unsqueeze(-1).unsqueeze(-1)  # re-normalize!
        else:
            t_strg = None
        if self.conf.dpar_unlab:
            t_goldL[t_goldL != self.IDX_PA] = 0  # mark every non-PA ones as 0!
        return t_tmask, (t_goldH, t_goldL), t_strg

    def _do_aug_score(self, t_score, t_tmask, t_golds, adding: float, inplace: bool):
        t_goldH, t_goldL = t_golds
        t_cons = t_tmask * (t_goldH != self.IDX_PA).to(t_tmask) * (t_goldL != self.IDX_PA).to(t_tmask)  # [*, L]
        ext_goldH, ext_goldL = extend_idxes(t_goldH.clamp(min=0), BK.get_shape(t_goldH, -1)), \
                               extend_idxes(t_goldL.clamp(min=0), self.osize)
        # [*, Lm, Lh, V]
        tmp_m = (ext_goldH.unsqueeze(-1) * ext_goldL.unsqueeze(-2)) * (t_cons.unsqueeze(-1).unsqueeze(-1))
        if inplace:
            t_score += tmp_m * adding
            return t_score
        else:
            ret = t_score + tmp_m * adding
            return ret

    def _do_inf(self, t_score, t_tmask, t_golds=None, cons_mode='', do_m=True, do_hard_m=False):
        ADDING = 100.  # should be enough!
        _tau = self.conf.inf_tau
        with BK.no_grad_env():
            tmp_score = t_score / _tau
            if t_golds is not None:  # add for the constraints
                tmp_score = self._do_aug_score(tmp_score, t_tmask, t_golds, ADDING, inplace=True)
            if (not do_m) or do_hard_m:
                ret = self.dpar_lab.out.get_argmax(tmp_score, t_tmask, mode=cons_mode)  # 2x[*, L]
                if do_m:  # [*, Lm, Lh, V]
                    p0, p1 = ret
                    pm = extend_idxes(p0, BK.get_shape(p0, -1)).unsqueeze(-1) * extend_idxes(p1, self.osize).unsqueeze(-2)
                    ret = pm
            else:  # [*, Lm, Lh, V]
                ret = self.dpar_lab.out.get_marginals(tmp_score, t_tmask, mode=cons_mode)
        return ret

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModDparConf = self.conf
        _strgR = float(self.strg_ratio.value)  # current value!
        _strgTh = 0.  # special one for hard-partial
        if rc.ibatch.dataset is not None:  # potentially read dataset specific one!
            _strgR = float(rc.ibatch.dataset.info.get('strgR', 1.)) * _strgR  # multiply!
            _strgTh = float(rc.ibatch.dataset.info.get('strgTh', 0.))
        # --
        # forward hid
        t_hid = self._do_forward(rc)  # first forward the encoder: [*, L]
        # prepare targets
        _, _, _, arr_toks = rc.get_cache((self.name, 'input'))  # [*, L]
        # TODO(+1): maybe more efficient by selecting with t_tmask ...
        t_score0, t_score = self.dpar_lab.forward_scores(t_hid)  # [*, L, L, V]
        t_tmask, t_golds, t_strg = self.prep_ibatch_trgs(rc.ibatch, arr_toks, prep_soft=(_strgR>0.))
        if conf.store_att:
            self.set_att(t_score, t_golds, t_tmask, rc, float(self.store_att_gratio))
        # --
        ret_info = {}
        if _strgR>0. and _strgTh > 0.:  # special mode: partial(hard) with thresh2
            _lenV = len(self.voc)
            _t_strg0 = flatten_dims(t_strg, -2, None)  # [*, L, L*V]
            _max_vs, _max_is = _t_strg0.max(-1)  # [*, L]
            t_drop = (t_tmask > 0) & (_max_vs < _strgTh)
            _maxH, _maxL = (_max_is // _lenV), (_max_is % _lenV)
            _maxH[t_drop] = self.IDX_PA
            _maxL[t_drop] = self.IDX_PA
            _old_strg = t_strg
            t_strg = self._do_inf(t_score, t_tmask, (_maxH, _maxL))
            _i0, _i1 = t_tmask.sum().item(), t_drop.sum().item()
            ret_info.update({'all_toks': _i0, 'all_strgV': _i0 - _i1})
        # --
        if _strgR == 0.:
            mm = self._do_inf(t_score, t_tmask, t_golds)
        elif _strgR == 1.:
            mm = t_strg
        else:
            mm = self._do_inf(t_score, t_tmask, t_golds)
            mm = mm * (1.-_strgR) + t_strg * _strgR
        # --
        if conf.loss_margin > 0.:  # minus margin for the gold ones
            t_score = self._do_aug_score(t_score, t_tmask, t_golds, -conf.loss_margin, inplace=False)
        t_weight = t_tmask.clone()
        t_pa = ((t_golds[0] == self.IDX_PA) | (t_golds[1] == self.IDX_PA)) & (t_tmask > 0)
        t_weight[t_pa] = conf.partial_alpha
        if _strgR>0. and conf.strg_thresh > 0.:
            t_ignore = t_pa & (t_strg.max(-1)[0].max(-1)[0] < conf.strg_thresh)
            t_weight[t_ignore] = 0.001  # set a much smaller value!
        # --
        if conf.label_smooth > 0:
            dpar_mask = self.dpar_lab.out.make_dpar_mask(t_tmask)  # [*, L, L]
            mm = label_smoothing(mm, conf.label_smooth, kldim=2, t_mask=dpar_mask)
        loss_v, loss_c = self.dpar_lab.forward_loss(t_score, t_weight, mm)
        one_loss = self.compile_leaf_loss('dpar', loss_v, loss_c)
        all_losses = [one_loss]
        if conf.loss_unlab > 0.:  # extra loss for unlab
            if conf.loss_margin > 0.:  # on edge scores!
                t_score0 = self._do_aug_score(t_score0, t_tmask, (t_golds[0], t_golds[1]*0),
                                              -conf.loss_margin, inplace=False)
            loss_v0, loss_c0 = self.dpar_lab.forward_loss(t_score0, t_weight, mm.sum(-1, keepdims=True))
            one_loss0 = self.compile_leaf_loss('dpar0', loss_v0 * conf.loss_unlab, loss_c0)
            all_losses.append(one_loss0)
        ret = self.compile_losses(all_losses)
        return (ret, ret_info)

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModDparConf = self.conf
        kwargsP = self.process_kwargs(kwargs)  # further process it!
        _pred_for_cali, _pred_use_partial, _pred_do_strg, _pred_strg_hard, _pred_do_dec, _pred_m_tau = conf.obtain_values(["pred_for_cali", "pred_use_partial", "pred_do_strg", "pred_strg_hard", "pred_do_dec", "pred_m_tau"], **kwargsP)
        # --
        # forward hid
        t_hid = self._do_forward(rc)  # first forward the encoder: [*, L]
        # prepare targets
        _, _, _, arr_toks = rc.get_cache((self.name, 'input'))  # [*, L]
        # TODO(+1): maybe more efficient by selecting with t_tmask ...
        _, t_score = self.dpar_lab.forward_scores(t_hid)  # [*, L, L, V]
        # --
        t_tmask, t_golds, _ = self.prep_ibatch_trgs(rc.ibatch, arr_toks, prep_soft=False)
        if conf.store_att:
            self.set_att(t_score, t_golds, t_tmask, rc, conf.store_att_gratioT)
        t_golds_backup = t_golds
        if not _pred_use_partial:
            t_golds = None  # no usage of gold!
        if _pred_do_strg or _pred_for_cali:
            _lenV = len(self.voc)
            # marginals
            mm = self._do_inf(t_score, t_tmask, t_golds, cons_mode=conf.pred_cons_mode, do_hard_m=_pred_strg_hard)
            if _pred_m_tau != 1.:
                mm = (mm.log() / _pred_m_tau).softmax(-1)
            arr_mm = BK.get_value(mm)  # [*, L, L, V]
            for _sidx, _item in enumerate(rc.ibatch.items):
                _sent = _item.sent
                _map, _ = self.get_tok_map(_sent, arr_toks[_sidx], df=0)  # Ls -> tok-ii
                _tmp_iarr = np.asarray([0] + _map)  # [1+Ls] add an AROOT
                arr_strg = arr_mm[_sidx][_tmp_iarr[:, None], _tmp_iarr[None, :]].copy()  # [1+Ls, 1+Ls, V]
                # note: handling truncation!
                _valid = np.asarray([1.] + [float(z>0) for z in _map], dtype=np.float32)  # [1+Ls] add an AROOT
                arr_strg *= _valid[:, None]
                _sent.arrs[self.KEY_STRG] = arr_strg
            # --
            if _pred_for_cali:
                if _pred_for_cali == 'm':  # marginal-prob
                    _zscore = mm
                elif _pred_for_cali == 'logm':  # log-m-prob
                    _zscore = mm.log()
                else:
                    _zscore = t_score
                t_full_score, t_full_gold = \
                    flatten_dims(_zscore, -2, None), t_golds_backup[0] * self.osize + t_golds_backup[1]
                t_comb = BK.concat([t_full_gold.unsqueeze(-1).to(t_full_score), t_full_score], -1)  # [*,L,L,1+V] compact!
                t_tmask2 = t_tmask.clone()  # [*, L]
                t_tmask2[:, 0] = 0.  # no AROOT!
                for _sidx, _item in enumerate(rc.ibatch.items):
                    _sent = _item.sent
                    _sent.arrs[self.KEY_CALI] = BK.get_value(t_comb[_sidx][t_tmask2[_sidx] > 0])
            # --
        # --
        cc = {}
        if _pred_do_dec:
            cc = {'sent': 0}
            # decoding
            t_best_heads, t_best_labs = self._do_inf(t_score, t_tmask, t_golds,
                                                     cons_mode=conf.pred_cons_mode, do_m=False)  # [*, L]
            arrH, arrL = BK.get_value(t_best_heads), BK.get_value(t_best_labs)  # [*, L]
            for _sidx, _item in enumerate(rc.ibatch.items):
                cc['sent'] += 1
                _sent = _item.sent
                _map0, _map1 = self.get_tok_map(_sent, arr_toks[_sidx], df=-1)  # Ls <-> tok-ii
                _map0 = [max(0, z) for z in _map0]
                _pred_heads1, _pred_labs0 = arrH[_sidx][_map0], arrL[_sidx][_map0]  # [Ls]
                _pred_heads = [1+_map1[z] for z in _pred_heads1]  # translate back to 1+widx
                _pred_labs = self.voc.seq_idx2word(_pred_labs0)
                _sent.build_dep_tree(_pred_heads, _pred_labs)
        # --
        return cc

    def set_att(self, t_score, t_golds, t_tmask, rc, gratio):
        dpar_mask = self.dpar_lab.out.make_dpar_mask(t_tmask)  # [*, L, L]
        t_score2 = (BK.logsumexp(t_score, -1) + (1-dpar_mask)*-1000.).softmax(-1)  # [*, Lm, Lh]
        t_gold2 = extend_idxes(t_golds[0].clamp(min=0), BK.get_shape(t_golds[0])[-1])  # head [*, Lm, Lh]
        ret = gratio * t_gold2 + (1.-gratio) * t_score2
        ret = ret * dpar_mask
        rc.set_cache((self.name, 't_att'), ret)

# --
# b mspx/tasks/zdpar/mod:
