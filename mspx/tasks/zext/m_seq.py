#

# using the method of seqlab!

import numpy as np
from mspx.data.vocab import Vocab, SeqVocabConf, SeqVocab
from mspx.utils import zwarn, Constants, zlog
from mspx.nn import BK, ZModConf, SeqLabelerConf, ZRunCache, extend_idxes, label_smoothing
from mspx.proc.run import SVConf
from .mod import *

@ZModConf.rd('ext_seqlab')
class ExtSeqlabConf(ZModExtConf):
    def __init__(self):
        super().__init__()
        self.allow_voc_unk = False  # many times we may want to allow this
        # --
        self.seqvoc = SeqVocabConf()
        self.seqlab = SeqLabelerConf().direct_update(_rm_names=['csize', 'isize'])
        self.label_smooth = 0.  # label smoothing
        self.strg_ratio = SVConf.direct_conf(val=0.)  # use soft target for loss?
        self.strg_thresh = 0.  # need strg's max-prob > this
        # self.strg_thresh2 = 0.  # special mode for argmax-marginal + partial
        self.inf_tau = 1.  # temperature for inference (marginals & argmax)
        self.loss_margin = 0.  # minus score of gold (unary) ones
        self.partial_alpha = 1.  # weight for partial parts
        self.train_cons = False  # use BIO constraints in training
        # special masking
        self.ignore_types = []  # ignore types
        # for predicting
        self.pred_use_partial = False  # use partial annotations as constraints!
        self.pred_do_strg = False  # output marginals to strg?
        self.pred_no_strg1 = False  # no store of strg1 (simply to save space ...)
        self.pred_strg_hard = False  # hard strg (argmax) rather than soft (marginal)
        self.pred_do_dec = True  # do actual decoding (and write results)
        self.pred_output_seqlab = False  # output info of seqlab?
        self.pred_cons = False  # use BIO constraints in testing
        self.pred_for_cali = ''  # store scores (logit or logm) for calibration!
        self.pred_m_tau = 1.  # tau for m-inference at pred
        self.pred_score_topk = 0  # store (B-) topk (marginal) scores for the frames
        self.pred_supp_o = 0.  # marginal-thresh for "O"-suppression in decoding!
        self.pred_mark_partial = []  # extra partial marks!

@ExtSeqlabConf.conf_rd()
class ExtSeqlabMod(ZModExt):
    def __init__(self, conf: ExtSeqlabConf, ztask: ZTaskExt, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ExtSeqlabConf = self.conf
        # --
        _ssize = self.bout.dim_out_hid()
        self.seqvoc = SeqVocab(self.voc, conf.seqvoc)
        self.seqlab = conf.seqlab.make_node(isize=_ssize, csize=len(self.seqvoc))
        self.osize = len(self.seqvoc)
        self.tcons_train, self.tcons_pred = None, None
        if conf.train_cons or conf.pred_cons:
            _NINF = -1000.
            _extra_t = BK.input_real(self.seqvoc.get_allowed_transitions())
            _extra_t = (1. - (_extra_t > 0.).to(BK.DEFAULT_FLOAT)) * _NINF
            if conf.train_cons:
                self.tcons_train = _extra_t
            if conf.pred_cons:
                self.tcons_pred = _extra_t
        self.strg_ratio = self.svs['strg_ratio']
        # ignore types
        self.ignore_mask = None
        if conf.ignore_types:
            _m = [int(any(t.endswith("-"+z) for z in conf.ignore_types)) for t in self.seqvoc.full_i2w]
            self.ignore_mask = BK.input_real(_m)
            zlog(f'Prepare ignore_mask with {conf.ignore_types}')
        # --
        # extra info: stored in "Sent"
        self.KEY_PA = f"{self.name}_ispart"  # partial info: whether it is partial? note: explicitly mark NIL frames!
        self.LAB_NIL = ztask.conf.lab_nil
        self.LAB_PA = ztask.conf.lab_unk
        self.IDX_PA, self.IDX_NIL = -1, 0  # special vocab indexes
        self.KEY_STRG = f"{self.name}_strg"  # soft target
        self.KEY_CALI = f"{self.name}_cali"  # for calibration!
        # --

    # --
    # prepare target labels; note: toks may have extra paddings, but does not matter ...
    def prep_one_trgs(self, item, toks):
        _allow_voc_unk = self.conf.allow_voc_unk
        # --
        ztask = self.ztask
        # first prepare sent!
        center_sent = item.sent
        center_frames = center_sent.get_frames(cates=self.frame_cate)
        center_frames = [z for z in center_frames if not z.info.get('is_pred', False)]  # note: ignore pred here!
        # assign non-nil ones
        _voc, _svoc = self.voc, self.seqvoc
        s_tags = []  # directly single tags!
        s_spans = []  # spans!
        for f in center_frames:
            flab = ztask.get_frame_label(f)
            if flab == self.LAB_NIL or flab == self.LAB_PA:
                continue
            _stag = _svoc.word2idx(flab, None)
            _widx, _wlen = f.mention.get_span()
            if _stag is not None:  # directly seq-tag!
                assert _wlen == 1  # todo(+2): currently not supported with multiple cates!
                s_tags.append((_widx, _stag))
            else:
                s_spans.append((_widx, _wlen,
                                (self.voc.get(flab,0) if _allow_voc_unk else self.voc[flab])))
        is_partial = center_sent.info.get(self.KEY_PA, False)
        s_idxes = self.seqvoc.spans2tags_idx(s_spans, len(center_sent), t_o=(self.IDX_PA if is_partial else None))
        if len(s_idxes) > 1:
            zwarn(f"Overlapping mentions in {center_sent}, ignoring: {s_idxes[1:]}")
        s_idxes = s_idxes[0][0]  # [Ls]
        # assign nil ones
        for f in center_frames:
            flab = ztask.get_frame_label(f)
            _widx, _wlen = f.mention.get_span()
            if flab == self.LAB_NIL:  # explicit NIL labeling!
                s_idxes[_widx:_widx+_wlen] = [self.IDX_NIL] * _wlen
            if flab == self.LAB_PA:  # explicit UNK
                s_idxes[_widx:_widx+_wlen] = [self.IDX_PA] * _wlen
        for _widx, _stag in s_tags:
            assert s_idxes[_widx] in [0, self.IDX_PA], "Overlapping annotations?"
            s_idxes[_widx] = _stag
        # --
        # note: only consider center sent as target!
        first_ii, last_ii = None, None
        one_mask = [0.] * len(toks)  # [L]
        for curr_ii, curr_tok in enumerate(toks):
            if curr_tok is not None:
                if curr_tok.sent is center_sent:  # hit it
                    one_mask[curr_ii] = 1.
                    if first_ii is None:
                        first_ii = curr_ii
                    last_ii = curr_ii
        first_widx, last_widx = toks[first_ii].widx, toks[last_ii].widx
        one_gold = [self.IDX_NIL] * len(one_mask)  # [L]
        one_gold[first_ii:(last_ii+1)] = s_idxes[first_widx:(last_widx+1)]
        return one_mask, one_gold, (first_ii, last_ii, first_widx, last_widx)

    def prep_ibatch_trgs(self, ibatch, arr_toks, prep_soft: bool, no_cache=False):
        # first collect them!
        _shape = BK.get_shape(arr_toks)
        _seqvoc = self.seqvoc
        _lenV = len(_seqvoc)  # output size!
        arr_tmask = np.full(_shape, 0., dtype=np.float32)  # [*, L], valid as targets
        arr_gold = np.full(_shape, 0, dtype=np.int32)  # [*, L], gold label, INT_PA as partial
        if prep_soft:  # [*, L, V], [*, V, V], soft targets
            arr_strg0, arr_strg1 = np.full(_shape+[_lenV], 0., dtype=np.float32), \
                                   np.full(_shape+[_lenV, _lenV], 0., dtype=np.float32)
        else:
            arr_strg0, arr_strg1 = None, None
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
            one_mask, one_gold, (first_ii, last_ii, first_widx, last_widx) = _cache
            _size = min(len(one_mask), _shape[-1])  # there can be padding differences!
            arr_tmask[bidx, :_size] = one_mask[:_size]
            arr_gold[bidx, :_size] = one_gold[:_size]
            if prep_soft:
                _strg0, _strg1 = csent.arrs.get(self.KEY_STRG+"0"), csent.arrs.get(self.KEY_STRG+"1")
                if _strg0 is not None and _strg1 is None:  # simply expand
                    # note: avoid underflow by using torch
                    import torch
                    _strg0 = torch.tensor(_strg0)
                    _strg1 = _strg0[:-1, :, np.newaxis] * _strg0[1:, np.newaxis, :]  # [L-1, V, V]
                    _strg1 = _strg1.numpy()
                if _strg0 is None or _strg1 is None:
                    no_strg = True
                else:
                    arr_strg0[bidx, first_ii:(last_ii+1)] = _strg0[first_widx:(last_widx+1)]
                    arr_strg1[bidx, first_ii:last_ii] = _strg1[first_widx:last_widx]  # aligned left
        # --
        # return
        t_tmask = BK.input_real(arr_tmask)
        t_gold = BK.input_idx(arr_gold)
        if prep_soft and (not no_strg):  # prepare soft and all has strg!
            t_strg0, t_strg1 = BK.input_real(arr_strg0), BK.input_real(arr_strg1)
        else:
            t_strg0, t_strg1 = None, None
        return t_tmask, t_gold, (t_strg0, t_strg1)

    def _do_aug_score(self, s_unary, t_tmask, t_gold, adding: float, inplace: bool):
        t_cons = t_tmask * (t_gold != self.IDX_PA).to(t_tmask)  # [*, L]
        tmp_adding = (t_cons * adding).unsqueeze(-1)  # [*, L, 1]
        if inplace:
            s_unary.scatter_add_(-1, t_gold.clamp(min=0).unsqueeze(-1), tmp_adding)
            return s_unary
        else:
            ret = s_unary.scatter_add(-1, t_gold.clamp(min=0).unsqueeze(-1), tmp_adding)
            return ret

    def _do_inf(self, s_unary, s_mat, t_tmask, t_gold=None, t_cons=None, do_m=True, do_hard_m=False):
        ADDING = 100.  # should be enough!
        _tau = self.conf.inf_tau
        with BK.no_grad_env():
            tmp_unary, tmp_mat = s_unary / _tau, (s_mat / _tau) if s_mat is not None else None  # also copy here!
            if t_gold is not None:  # use the constraints
                tmp_unary = self._do_aug_score(tmp_unary, t_tmask, t_gold, ADDING, inplace=True)
            # --
            # constraints?
            if t_cons is not None:
                tmp_mat = t_cons if tmp_mat is None else (tmp_mat + t_cons)
            # --
            if (not do_m) or do_hard_m:
                # breakpoint()
                ret = self.seqlab.out.get_argmax(tmp_unary, tmp_mat, t_tmask)  # [*, L]
                if do_m:  # [*, L, V], [*, L-1, V, V]
                    m0 = extend_idxes(ret, self.osize)  # [*, L, V]
                    m1 = m0[..., :-1, :].unsqueeze(-1) * m0[..., 1:, :].unsqueeze(-2)  # [*, L-1, V, V]
                    ret = (m0, m1)
            else:  # [*, L, V], [*, L-1, V, V]
                ret = self.seqlab.out.get_marginals(tmp_unary, tmp_mat, t_tmask)
        return ret

    def _do_score(self, t_hid):
        s_unary, s_mat = self.seqlab.forward_scores(t_hid)
        if self.ignore_mask is not None:
            s_unary = s_unary + self.ignore_mask * (-100.)
        return s_unary, s_mat

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        conf: ExtSeqlabConf = self.conf
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
        s_unary, s_mat = self._do_score(t_hid)
        t_tmask, t_gold, (t_strg0, t_strg1) = self.prep_ibatch_trgs(rc.ibatch, arr_toks, prep_soft=(_strgR>0.))
        # if _strgR > 0. and t_strg0 is None:
        #     zwarn(f"No strg to serve as target for {_strgR}, fall back to _strgR=0!!")
        #     _strgR = 0.  # no way to do strg!
        _tcons_train = self.tcons_train
        ret_info = {}
        # --
        if _strgR>0. and _strgTh > 0.:  # special mode: partial(hard)-crf with thresh2
            _max_vs, _max_is = t_strg0.max(-1)  # [*, L]
            t_drop = (t_tmask > 0) & (_max_vs < _strgTh)
            _max_is[t_drop] = self.IDX_PA
            _old_strgs = (t_strg0, t_strg1)
            t_strg0, t_strg1 = self._do_inf(s_unary, s_mat, t_tmask, t_gold=_max_is, t_cons=_tcons_train)
            _i0, _i1 = t_tmask.sum().item(), t_drop.sum().item()
            ret_info.update({'all_toks': _i0, 'all_strgV': _i0 - _i1})
        # --
        if _strgR == 0.:
            m0, m1 = self._do_inf(s_unary, s_mat, t_tmask, t_gold=t_gold, t_cons=_tcons_train)
        elif _strgR == 1.:
            m0, m1 = t_strg0, t_strg1
        else:
            m0, m1 = self._do_inf(s_unary, s_mat, t_tmask, t_gold=t_gold, t_cons=_tcons_train)
            m0 = m0 * (1.-_strgR) + t_strg0 * _strgR
            if m1 is not None:
                m1 = m1 * (1.-_strgR) + t_strg1 * _strgR
        if _tcons_train is not None and s_mat is not None:
            s_mat = s_mat + _tcons_train  # note: remember to add here for loss!
        # --
        if conf.loss_margin > 0.:  # minus margin for the gold ones
            s_unary = self._do_aug_score(s_unary, t_tmask, t_gold, -conf.loss_margin, inplace=False)
        t_weight = t_tmask.clone()
        t_pa = (t_gold == self.IDX_PA) & (t_tmask > 0)
        t_weight[t_pa] = conf.partial_alpha
        if _strgR>0. and conf.strg_thresh > 0.:
            t_ignore = t_pa & (t_strg0.max(-1)[0] < conf.strg_thresh)
            t_weight[t_ignore] = 0.001  # set a much smaller value!
        # --
        if conf.label_smooth > 0.:
            m0 = label_smoothing(m0, conf.label_smooth, kldim=1, t_mask=t_tmask)
            # note: simply nope for m1!
        loss_v, loss_c = self.seqlab.forward_loss(s_unary, s_mat, t_weight, m0, m1)
        one_loss = self.compile_leaf_loss('seqlab', loss_v, loss_c)
        ret = self.compile_losses([one_loss])
        return (ret, ret_info)

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        conf: ExtSeqlabConf = self.conf
        kwargsP = self.process_kwargs(kwargs)  # further process it!
        _pred_for_cali, _pred_use_partial, _pred_do_strg, _pred_strg_hard, _pred_do_dec, _pred_score_topk, _pred_m_tau, _pred_supp_o, _p_marks = conf.obtain_values(["pred_for_cali", "pred_use_partial", "pred_do_strg", "pred_strg_hard", "pred_do_dec", "pred_score_topk", "pred_m_tau", "pred_supp_o", "pred_mark_partial"], **kwargsP)
        # --
        # forward hid
        t_hid = self._do_forward(rc)  # first forward the encoder: [*, L]
        # prepare targets
        _, _, _, arr_toks = rc.get_cache((self.name, 'input'))  # [*, L]
        # TODO(+1): maybe more efficient by selecting with t_tmask ...
        s_unary, s_mat = self._do_score(t_hid)
        # --
        t_tmask, t_gold, _ = self.prep_ibatch_trgs(rc.ibatch, arr_toks, prep_soft=False)
        t_gold_backup = t_gold
        if not _pred_use_partial:
            t_gold = None  # no usage of gold!
        if _pred_do_strg or _pred_supp_o > 0 or _pred_score_topk > 0 or _pred_for_cali:
            # marginals
            m0, m1 = self._do_inf(s_unary, s_mat, t_tmask,
                                  t_gold=t_gold, t_cons=self.tcons_pred, do_hard_m=_pred_strg_hard)
            if _pred_m_tau != 1.:
                m0 = (m0.log() / _pred_m_tau).softmax(-1)
            arr_m0, arr_m1 = BK.get_value(m0), BK.get_value(m1)
            _lenV = len(self.seqvoc)  # output size!
            for _sidx, _item in enumerate(rc.ibatch.items):
                _sent = _item.sent
                _len = len(_sent)
                arr_strg0, arr_strg1 = np.full([_len, _lenV], 0., dtype=np.float32), \
                                       np.full([_len-1, _lenV, _lenV], 0., dtype=np.float32)
                arr_strg0[:, self.IDX_NIL] = 1.  # default "O"
                arr_strg1[:, self.IDX_NIL, self.IDX_NIL] = 1.  # default "O"->"O"
                _tok0, _good0 = None, False
                for _ii, _tok in enumerate(arr_toks[_sidx]):
                    if _tok is not None and _tok.sent is _sent:
                        arr_strg0[_tok.widx] = arr_m0[_sidx, _ii]
                        if _good0:
                            arr_strg1[_tok0.widx] = arr_m1[_sidx, _ii]
                        _good0 = True
                    else:
                        _good0 = False
                    _tok0 = _tok
                if _pred_do_strg:
                    _sent.arrs[self.KEY_STRG+"0"] = arr_strg0
                    if not conf.pred_no_strg1:
                        _sent.arrs[self.KEY_STRG+"1"] = arr_strg1
                _sent.cache[self.KEY_STRG] = arr_strg0  # save it temporally!
            # --
            if _pred_for_cali:  # note: for simplicity only s_unary!
                if _pred_for_cali == 'm':  # marginal-prob
                    _zscore = m0
                elif _pred_for_cali == 'logm':  # log-m-prob
                    _zscore = m0.log()
                else:
                    _zscore = s_unary
                t_comb = BK.concat([t_gold_backup.unsqueeze(-1).to(_zscore), _zscore], -1)  # [*,L,V] compact!
                for _sidx, _item in enumerate(rc.ibatch.items):
                    _sent = _item.sent
                    _sent.arrs[self.KEY_CALI] = BK.get_value(t_comb[_sidx][t_tmask[_sidx] > 0])
                    # breakpoint()
        # --
        cc = {}
        if _pred_do_dec:
            cc = {'seq': 0, 'frame': 0}
            # --
            # check or clear the frames!
            existing_masks = []
            for item in rc.ibatch.items:
                _one_exists = [0] * len(item.sent)
                for f in item.sent.get_frames(cates=self.frame_cate):  # copy!
                    if (not _pred_use_partial) or f.info.get('is_pred', False):  # delete!
                        f.del_self()
                    else:  # keep!
                        _widx, _wlen = f.mention.get_span()
                        _one_exists[_widx:(_widx+_wlen)] = [1] * _wlen
                        for _pm in _p_marks:
                            f.info[_pm] = True  # mark partial!
                existing_masks.append(_one_exists if _pred_use_partial else None)
            # --
            # argmax decode
            if _pred_supp_o > 0:  # only change decoding score!
                s_unary2 = s_unary.clone()
                s_unary2[..., 0] -= 50. * (t_tmask * (m0[..., 0] < _pred_supp_o).to(t_tmask))
            else:
                s_unary2 = s_unary
            t_best_labs = self._do_inf(s_unary2, s_mat, t_tmask, t_gold=t_gold, t_cons=self.tcons_pred, do_m=False)
            arr_best_labs = BK.get_value(t_best_labs)
            for _sidx, _labs in enumerate(arr_best_labs):
                cc['seq'] += 1
                _sent = rc.ibatch.items[_sidx].sent
                _toks = arr_toks[_sidx]
                _spans = self.seqvoc.tags2spans_idx(_labs.tolist())  # [start, length, orig_idx]
                # note: put orig label here!
                _seqlabs = [''] * len(_sent)
                _seqlabs0 = self.seqvoc.seq_idx2word(_labs.tolist())
                _sslen = min(len(_seqlabs), len(_seqlabs0))
                _seqlabs[:_sslen] = _seqlabs0[:_sslen]
                # --
                _one_exists = existing_masks[_sidx]
                for _start, _len, _orig_lab in _spans:
                    _valid_toks = [_toks[z] for z in range(_start, _start + _len) if _toks[z] is not None]
                    if len(_valid_toks) > 0:
                        _valid_toks = [z for z in _valid_toks if z.sent is _valid_toks[0].sent]
                        _new_widx, _new_wlen = _valid_toks[0].widx, _valid_toks[-1].widx - _valid_toks[0].widx + 1
                        if _valid_toks[0].sent is not _sent:
                            continue
                        if _one_exists is not None and any(_one_exists[_new_widx:(_new_widx+_new_wlen)]):
                            continue  # keep the existing gold!
                        _new_lab = self.voc.idx2word(_orig_lab)
                        _new_item = _sent.make_frame(_new_widx, _new_wlen, _new_lab, self.frame_cate)
                        _new_item.info['is_pred'] = True  # mark pred!
                        for _pm in _p_marks:
                            _new_item.info[_pm] = True  # mark partial!
                        cc['frame'] += 1
                        # _seqlabs[_new_widx:_new_widx+_new_wlen] = self.seqvoc.span2tags_str(_new_lab, _new_wlen)
                        if _pred_score_topk > 0:  # assign scores; todo(+N): simply use "B-"
                            _widx_scores = _sent.cache[self.KEY_STRG][_new_widx].tolist()  # [V]
                            _b_scores = [(vv[2:], _widx_scores[self.seqvoc[vv]]) for vv in self.seqvoc.keys()
                                         if vv.startswith("B-")]
                            _b2_scores = sorted(_b_scores, key=lambda x: -x[-1])[:_pred_score_topk]
                            _scores = {self.LAB_NIL: _widx_scores[self.seqvoc["O"]]}
                            _scores.update(_b2_scores)
                            _new_item.info["topk"] = _scores
                            _new_item.info["pred_prob"] = _widx_scores[self.seqvoc["B-" + _new_lab]]  # best prob
                if conf.pred_output_seqlab:
                    _sent.info[f'{self.name}_seqlab'] = _seqlabs
        # --
        return cc

# --
# b mspx/tasks/zext/m_seq:
