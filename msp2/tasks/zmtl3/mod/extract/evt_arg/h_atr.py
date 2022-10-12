#

# helper for autoregressive decoding (mode=s2s, do_atr=True)

__all__ = [
    "ArgAtrHelper",
]

from msp2.nn import BK
from msp2.utils import AlgoHelper, zwarn

class ArgAtrHelper:
    def __init__(self, arg_mod):
        self.arg_mod = arg_mod

    # [bs, lq], [bs, ls, D], [bs, ls];; [bs, lq], [bs, Cand, D], [bs, Cand];; t_gold:[bs, Cand]=None
    def decode_s2s(self, t_trg_ids, t_cross, t_cross_mask, mixes, t_qr, t_content, t_vcands, t_cand_emb, evts, arr_toks,
                   t_gold=None, ret_step_res=False):
        # todo(+N): currently simply do greedy
        arg_mod = self.arg_mod
        # --
        step_trg_ids = t_trg_ids.split(1, dim=-1)  # L[bs, 1]
        step_mixes = [(a.split(1, dim=-1), b.split(1, dim=-2)) for a,b in mixes]  # L(L[bs,1], L[bs,1,D])
        step_qr = t_qr.split(1, dim=-1)  # L[bs, 1]
        # --
        # loop
        _neg_t = arg_mod.neg_delta() + arg_mod.conf.pred_neg_delta  # []
        _br, _bc = arg_mod.conf.pred_br, arg_mod.conf.pred_bc  # note: 'bc' for post-pruning!
        _br_margin = arg_mod.conf.pred_br_margin
        _atr_mix_scale = arg_mod.conf.atr_mix_scale
        dec_cache = None
        next_mix = None
        all_logprob, all_pred = [], []
        all_step_res = []
        for cur_ii, cur_trg_ids in enumerate(step_trg_ids[:-1]):  # note: no need to forward the last step!
            cur_mixes = [(a[cur_ii], b[cur_ii]) for a,b in step_mixes]
            # prediction mix?
            if next_mix is not None:
                cur_mixes.append(next_mix)
            # forward
            cur_out = arg_mod.bmod.forward_dec(
                cur_trg_ids, cache_past=dec_cache, t_cross=t_cross, t_cross_mask=t_cross_mask, mixes=cur_mixes)
            if ret_step_res:
                all_step_res.append(cur_out)
            # update cache
            dec_cache = cur_out.past_key_values
            # prediction: following "Arg._pred"
            cur_score = arg_mod.sim(cur_out.last_hidden_state, t_content)  # [bs, 1, C]
            cur_qr = step_qr[cur_ii+1]  # [bs,1] offset by 1!
            # "_process_them"
            cur_valid = t_vcands.unsqueeze(-2) * (cur_qr > 0).float().unsqueeze(-1)  # [bs, 1, C]
            _t_pad = BK.zeros(cur_qr.shape).unsqueeze(-1)  # [bs, 1, 1]
            cur_score1 = BK.concat([_t_pad+_neg_t, cur_score], -1)  # [bs, 1, 1+C]
            cur_mask1 = BK.concat([_t_pad+1., cur_valid], -1)  # [bs, 1, 1+C]
            # "_process_one"
            _NEG = -10000.
            cur_logprob = (cur_score1 + (1.-cur_mask1) * _NEG).log_softmax(dim=-1)[...,1:]  # [bs, 1, C]
            # "_pred"
            if t_gold is not None:  # teacher forcing (using gold)
                cur_pred = cur_valid * (t_gold.unsqueeze(-2) == cur_qr.unsqueeze(-1)).float()  # [bs, 1, C]
            else:
                cur_pred = cur_valid * (cur_score > _neg_t).float() \
                           * (cur_logprob >= cur_logprob.topk(_br, dim=-1)[0].min(-1, keepdims=True)[0]).float() \
                           * (cur_logprob >= (cur_logprob.max(-1, keepdims=True)[0] - _br_margin))  # [bs, 1, C]
            all_logprob.append(cur_logprob)
            all_pred.append(cur_pred)
            # prepare for next mixing
            _w = cur_pred / cur_pred.sum(-1, keepdims=True).clamp(min=1.)
            _e = BK.matmul(_w, t_cand_emb) * _atr_mix_scale  # [bs, 1, D]
            next_mix = ((cur_pred.sum(-1)>0).float(), _e)
        # --
        # create outputs
        t_logprob, t_pred = BK.concat(all_logprob, -2), BK.concat(all_pred, -2)  # [bs, lq-1, C]
        for bidx, (evt, toks) in enumerate(zip(evts, arr_toks)):
            _q, _c = t_pred[bidx].nonzero(as_tuple=True)  # [??]
            _s = t_logprob[bidx][_q, _c]  # [??]
            _r = t_qr[bidx][_q+1]  # [??], note: remember to +1
            _them = sorted(zip(*[z.tolist() for z in [_s, _r, _c]]), reverse=True)
            _count_t = [0] * len(toks)
            for _ss, _rr, _cc in _them:
                if _count_t[_cc] < _bc:
                    _count_t[_cc] += 1
                    assert _rr >= 1, "Must be valid query!"
                    _role, _tok = arg_mod.qmod.ridx2role(_rr), toks[_cc]
                    if t_gold is None:  # do the check but not actually assign!
                        new_ef = _tok.sent.make_entity_filler(_tok.widx, 1)
                        if arg_mod.extender is not None:
                            arg_mod.extender.extend_mention(new_ef.mention)
                        evt.add_arg(new_ef, role=_role.name, score=_ss)
        # --
        if ret_step_res:
            return {}, all_step_res
        else:
            return {}
        # --

    # free generation
    def decode_gen(self, t_cross, t_cross_mask, t_vmask,
                   evts, arr_toks, arr_rs, arr_rq_idxes, pat_ids):
        arg_mod = self.arg_mod
        conf = arg_mod.conf
        _tokenizer = arg_mod.tokenizer
        _nil_id = _tokenizer.mask_token_id
        _br, _bc = arg_mod.conf.pred_br, arg_mod.conf.pred_bc  # note: no considering of 'bc' here!
        # --
        # beam search
        bs_k = conf.gen_beam_size
        bs_kwargs = {k:float(v) for k,v in conf.gen_kwargs.items()}
        t_max_len = BK.input_idx([len(z)*2 for z in pat_ids])  # [bs], max len!
        final_id_t, final_mask_t, final_score_t = ArgAtrHelper.bmod_beam_search(
            arg_mod.bmod, bs_k, t_cross, t_cross_mask, t_max_len, t_vmask, **bs_kwargs)  # [bs, B, ??]
        # --
        # decode
        cc = {'count_evt': 0, 'count_rr': 0, 'count_hasstr': 0, 'count_matched': 0}
        arr_ids, arr_masks, arr_scores = [BK.get_value(z) for z in [final_id_t, final_mask_t, final_score_t]]
        for bidx, (evt, toks) in enumerate(zip(evts, arr_toks)):
            cc['count_evt'] += 1
            _trg_ids = arr_ids[bidx][arr_masks[bidx]>0].tolist()
            evt.info['seq_gen'] = _tokenizer.decode(_trg_ids)
            _trg_scores = arr_scores[bidx][arr_masks[bidx]>0].tolist()
            _pat_ids = pat_ids[bidx]
            # match these two
            tok_words = [t.word if t is not None else '' for t in toks]
            try:
                evt_ii = toks.tolist().index(evt.mention.shead_token)
            except ValueError:
                zwarn(f"Cannot locate evt position: {evt}")
                continue  # ignore the strange case!
            merge_to_a1, merge_to_a2, a1_to_merge, a2_to_merge = AlgoHelper.align_seqs(_pat_ids, _trg_ids)
            for rii, rr in enumerate(arr_rs[bidx]):
                if rr is not None:
                    cc['count_rr'] += 1
                    pat_rr_idx = arr_rq_idxes[bidx][rii]
                    # get the range
                    _m0, _m1 = a1_to_merge[pat_rr_idx-1], a1_to_merge[pat_rr_idx+1]
                    _a2s = [merge_to_a2[z] for z in range(_m0+1, _m1)]
                    _a2s = [z for z in _a2s if z is not None and _trg_ids[z] != _nil_id]
                    if len(_a2s) == 0:
                        continue  # no results!
                    _preds = [z.split() for z in _tokenizer.decode([_trg_ids[z] for z in _a2s]).split(" and ")]
                    _preds = [z for z in _preds if len(z)>0]
                    _score = sum(_trg_scores[z] for z in _a2s)
                    # find them in the toks
                    for _ps in _preds[:_br]:  # preds are usually short, thus simply enumerate
                        matched_iis = [tii for tii in range(len(tok_words)) if _ps == tok_words[tii:tii+len(_ps)]]
                        cc['count_hasstr'] += 1
                        if len(matched_iis) > 0:
                            best_ii = min(matched_iis, key=lambda x: abs(evt_ii-x))  # near the evt-trigger!
                            cc['count_matched'] += 1
                            # --
                            new_ef = toks[best_ii].sent.make_entity_filler(toks[best_ii].widx, len(_ps))
                            if arg_mod.extender is not None:
                                arg_mod.extender.extend_mention(new_ef.mention)
                            evt.add_arg(new_ef, role=rr.name, score=_score)
                        else:
                            # breakpoint()
                            pass
                        # --
        # --
        return cc

    # [bs, src, D], [bs, src]; [bs], [bs, V], note: adapted from "zgen/model/tasks/dec_slm.py"
    @staticmethod
    def bmod_beam_search(bmod, beam_size: int, t_cross, t_cross_mask, t_max_len,
                         t_vmask=None, eos_penalty=0., len_norm_alpha=0., len_reward=0.):
        _tokenizer = bmod.tokenizer
        pad_id, bos_id, eos_id = _tokenizer.pad_token_id, _tokenizer.cls_token_id, _tokenizer.sep_token_id
        bsize = len(t_cross)  # [bs]
        # --
        if beam_size != 1:  # [bs*B]
            t_cross, t_cross_mask, t_vmask, t_max_len = [
                z.repeat_interleave(beam_size, dim=0) if z is not None else None
                for z in [t_cross, t_cross_mask, t_vmask, t_max_len]]
        # --
        _NEG_INF = -10000.
        _PRIZE = 100.
        t_vmask_bias = None if t_vmask is None else (_NEG_INF * (1.-t_vmask))  # [bs, V]
        # --
        # status
        cur_id_t = BK.constants_idx([bsize*beam_size, 1], value=bos_id)  # [bs*B, 1]
        cur_mask_t = BK.constants([bsize*beam_size, 1], value=1.)  # [bs*B, 1]
        cur_finished_t = (cur_mask_t<=0.).squeeze(-1)  # [bs*B]
        cur_score_t = BK.constants([bsize*beam_size, 1], value=0.)  # [bs*B, 1]
        cur_cache = None
        # prepare special masks: [0, -inf, ...]
        _beam_mask_v = BK.constants([beam_size], value=_NEG_INF)  # [B]
        _beam_mask_v[0] = 0.
        _arange_b2_t = BK.arange_idx(bsize*beam_size).unsqueeze(-1)  # [*, 1]
        _arange_bs_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bs, 1]
        # some others
        _shape_b2, _shape_b3 = [bsize, beam_size], [bsize, beam_size, beam_size]
        _prize_t = BK.input_real([_PRIZE] + [0.] * (beam_size - 1))  # [B]
        accu_score_t = BK.zeros(_shape_b2).to(BK.float32)  # not: use float32 to record this!!
        # --
        cur_step = 0
        while True:
            # forward last slice
            cur_out = bmod.forward_dec(
                cur_id_t[:, -1:], cache_past=cur_cache, t_cross=t_cross, t_cross_mask=t_cross_mask)
            # update cache
            cur_cache = cur_out.past_key_values
            # predictions
            t_score0 = bmod.forward_lmhead(cur_out.last_hidden_state).squeeze(-2)  # [bs*B, V]
            if t_vmask_bias is not None:
                t_score0 = t_score0 + t_vmask_bias  # [bs*B, V]
            if eos_penalty != 0.:
                t_score0[..., eos_id] += eos_penalty
            t_logprob = t_score0.log_softmax(-1)  # [bs*B, V]
            # beam search
            # -- inner beam
            inner_score, inner_id = t_logprob.topk(beam_size, dim=-1, sorted=True)  # [*, B]
            inner_id[cur_finished_t] = pad_id  # [*, B], put pad
            inner_score[cur_finished_t] = _beam_mask_v  # [*, B]
            # -- outer beam
            _extra_ranking_score = BK.zeros(_shape_b3)  # [bs, B0, B1] temp scored added for special treatment!
            if cur_step == 0:  # mask out non0s at the first step!
                _extra_ranking_score[:, 1:] = _NEG_INF
            one_is_finished_b = cur_finished_t.view(_shape_b3[:2])  # [bs, B0]
            _extra_ranking_score[one_is_finished_b] += _prize_t  # keep finished ones as they are!
            # rank them
            _ranking_score0 = accu_score_t.unsqueeze(-1) \
                              + inner_score.view(_shape_b3) + _extra_ranking_score  # [bs, B0, B1]
            _ranking_score = _ranking_score0.view(_shape_b3[:-2] + [-1])  # [bs, B0xB1]
            _, _rr_idx = _ranking_score.topk(beam_size, dim=-1, sorted=True)  # [bs, B]
            _rr_b0, _rr_b1 = _rr_idx // beam_size, _rr_idx % beam_size  # [bs, B0], [bs, B1]
            # reindex
            if beam_size != 1:
                batch_reidx_t = (_rr_b0 + _arange_bs_t * beam_size).view([-1])  # [*]
                cur_cache = bmod.dec_reorder_cache(cur_cache, batch_reidx_t)
                accu_score_t = accu_score_t[_arange_bs_t, _rr_b0]  # [bs, B]
                cur_id_t = cur_id_t[batch_reidx_t]  # [bs*B, prev]
                cur_mask_t = cur_mask_t[batch_reidx_t]  # [bs*B, prev]
                cur_finished_t = cur_finished_t[batch_reidx_t]  # [bs*B]
                cur_score_t = cur_score_t[batch_reidx_t]  # [bs*B, prev]
            # gather the corresponding ones
            re_id0, re_score0 = inner_id.view(_shape_b3)[_arange_bs_t, _rr_b0, _rr_b1], \
                                inner_score.view(_shape_b3)[_arange_bs_t, _rr_b0, _rr_b1]  # [bs, B], [bs, B]
            re_id, re_score = re_id0.view([-1]), re_score0.view([-1])  # [*], [*]
            # check EOS & update states
            accu_score_t += re_score0
            cur_id_t = BK.concat([cur_id_t, re_id.unsqueeze(-1)], -1)  # [*, prev+1]
            cur_mask_t = BK.concat([cur_mask_t, (~cur_finished_t).unsqueeze(-1)], -1)  # [*, prev+1]
            cur_finished_t = (cur_finished_t | (re_id == eos_id) | (cur_mask_t.sum(-1) >= t_max_len))  # [*]
            cur_score_t = BK.concat([cur_score_t, re_score.unsqueeze(-1)], -1)  # [*, prev+1]
            # finished?
            cur_step += 1
            if cur_finished_t.all().item():
                break
        # --
        # final ranking
        if beam_size != 1:
            num_step_t = cur_mask_t.sum(-1).view(_shape_b2) - 1  # [bs, 1]
            _ranking_div = ((5+num_step_t)/(5+1)) ** len_norm_alpha  # [bs, 1]
            _ranking_score = ((accu_score_t + len_reward * num_step_t) / _ranking_div)  # [bs, B]
            _, _rr_idx = _ranking_score.topk(beam_size, dim=-1, sorted=True)  # [bs, B]
            final_batch_reidx_t = (_rr_idx + _arange_bs_t * beam_size).view([-1])  # [*]
            # re-arrange
            final_id_t = cur_id_t[final_batch_reidx_t]
            final_mask_t = cur_mask_t[final_batch_reidx_t]
            final_score_t = cur_score_t[final_batch_reidx_t]
        else:
            final_id_t, final_mask_t, final_score_t = cur_id_t, cur_mask_t, cur_score_t
        # --
        # simply return the best one!
        rets = [z.view(_shape_b2+[-1])[:,0] for z in [final_id_t, final_mask_t, final_score_t]]
        return rets  # *[bs, lt]

# --
# b msp2/tasks/zmtl3/mod/extract/evt_arg/h_atr:
