#

# main argument extraction module
# note: assume one frame as one input item!

__all__ = [
    "ZTaskArgConf", "ZTaskArg", "ZModArgConf", "ZModArg",
]

import math
from typing import List
from collections import Counter
import numpy as np

from msp2.proc import FrameEvalConf, FrameEvaler, ResultRecord
from msp2.data.inst import yield_sents, yield_frames, set_ee_heads, DataPadder, SimpleSpanExtender, Sent
from msp2.data.vocab import SimpleVocab
from msp2.utils import zlog, zwarn, ZRuleFilter, Random, StrHelper, F1EvalEntry, MathHelper, wrap_color, default_pickle_serializer, default_json_serializer, zglob1z, ConfEntryChoices
from msp2.nn import BK
from msp2.nn.l3 import *
from ..base import *
from . import onto as zonto
from .m_query import *
from .m_repr import *
from .h_atr import ArgAtrHelper
from .h_aug import ArgAugConf, ArgAugHelper

class ZTaskArgConf(ZTaskBaseEConf):
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name if name is not None else 'arg'
        self.arg_conf = ZModArgConf()
        self.arg_eval = FrameEvalConf.direct_conf(weight_frame=0., weight_arg=1., bd_arg='', bd_arg_lines=50)
        # --
        self.build_oload = ""  # simply load this onto when building?

    def build_task(self):
        return ZTaskArg(self)

class ZTaskArg(ZTaskBaseE):
    def __init__(self, conf: ZTaskArgConf):
        super().__init__(conf)
        conf: ZTaskArgConf = self.conf
        # --
        self.evaler = FrameEvaler(conf.arg_eval)

    def get_onto(self):
        conf: ZTaskArgConf = self.conf
        # --
        if conf.build_oload:
            _path = zglob1z(conf.build_oload)
            onto = zonto.Onto.load_onto(_path)
        else:
            onto, = self.vpack
        return onto

    # build vocab
    def build_vocab(self, datasets: List):
        conf: ZTaskArgConf = self.conf
        # --
        if conf.build_oload:
            return None
        else:  # get a simple one!
            voc_evt = SimpleVocab.build_empty(pre_list=[], post_list=[])
            voc_arg = SimpleVocab.build_empty(pre_list=[], post_list=[])
            for dataset in datasets:
                if dataset.wset == 'train':
                    for evt in yield_frames(dataset.insts):
                        voc_evt.feed_one(evt.label)
                        for arg in evt.args:
                            voc_arg.feed_one(arg.label)
            voc_evt.build_sort()
            voc_arg.build_sort()
            zlog(f"Building evt voc: {voc_arg}")
            zlog(f"Building arg voc: {voc_arg}")
            onto = zonto.Onto.create_simple_onto(voc_evt.full_i2w, voc_arg.full_i2w)
            zlog(f"And get onto: {onto}")
        return (onto, )

    # eval
    def eval_insts(self, gold_insts: List, pred_insts: List, quite=False):
        # --
        if self.mod.conf.do_mlm:  # note: special mode!
            _losses0 = [z.info.get('loss_mlm', 0) for z in yield_frames(pred_insts)]
            _losses1 = [z for z in _losses0 if z>0.]
            avg_nloss = (- np.mean(_losses1).item()) if len(_losses1)>0 else 0.
            zlog(f"{self.name} detailed results:\n\tloss={avg_nloss}", func="result")
            res = ResultRecord(results={'nloss': avg_nloss, 'size0': len(_losses0), 'size1': len(_losses1)},
                               description='', score=avg_nloss)
            return res
        # --
        evaler = self.evaler
        evaler.reset()
        # --
        if evaler.conf.span_mode_arg != 'span':
            set_ee_heads(pred_insts)  # note: also set for preds!
            set_ee_heads(gold_insts)
        # --
        res0 = evaler.eval(gold_insts, pred_insts)
        res = ResultRecord(results=res0.get_summary(), description=res0.get_brief_str(), score=float(res0.get_result()))
        if not quite:
            res_detailed_str0 = res0.get_detailed_str()
            res_detailed_str = StrHelper.split_prefix_join(res_detailed_str0, '\t', sep='\n')
            zlog(f"{self.name} detailed results:\n{res_detailed_str}", func="result")
        return res

    # build mod
    def build_mod(self, model):
        return self.conf.arg_conf.make_node(self, model)

# --

class ZModArgConf(ZModBaseEConf):
    def __init__(self):
        super().__init__()
        # --
        # overall extraction mode
        self.arg_mode = "clf"  # clf/mrc/tpl/s2s
        self.qconf = QmodConf()
        self.mrc_pool_ques = False  # use avg-pool of the question as query
        self.s2s_do_atr = False  # do auto-regressive for s2s?
        self.atr_mix_scale = 1.  # input scale
        self.encq_sg = False  # encode query at the encoding side even in s2s/gen mode
        self.debug_print = False
        # --
        # common ones
        self.ctx_nsent_rates = [1.]  # extend context by sent, how many more before & after?
        self.upos_filter = []  # filtering by UPOS, by default allow all; an example: "+NV,-*"
        self.ctx_as_trg = True  # whether use ctx sents as targets
        self.max_content_len = 128  # maximum seq length for the content
        self.add_arg_conj = False  # propagate conj of args as args
        # mix_evt_ind
        self.mix_evt_ind = '0.'  # mix evt indicator, 0. means no mix!
        self.mix_evt_ind_initscale = 0.05  # init scale
        self.mix_evt_ind_share = ""
        # repr
        self.repr_conf = ZArgReprConf()  # how to represent one arg candidate?
        self.sim_conf = ConfEntryChoices({"simple": SimConf(), "complex": PairScoreConf()}, "simple")
        # train/test
        self.label_smoothing = 0.
        self.train_neg_sample = -1.  # negative down-sampling (relative to positives)
        # loss/pred norm over All/Role/Cand; note: for s2s, only supporting "Role"!
        self.neg_delta = ScalarConf.direct_conf(init=0., fixed=True)  # score delta added to neg score
        self.pred_neg_delta = 0.  # extra one for prediction
        self.train_neg_ratios = [-1., -1., -1.]  # negative down-weighting (relative to positives)
        self.loss_nw = [0., 1., 0.]
        self.pred_nw = [0., 1., 0.]
        self.pred_br = 2  # budget-role: max cand per role
        self.pred_br_margin = 2.  # extra one for budget-role: margin to the best one!
        self.pred_bc = 1  # budget-cand: max role per cand
        self.extend_span = True  # extend span for arg
        self.extend_span_sig = 'ef'  # signature of arg extender
        self.pred_given_posi = False  # special mode: no extraction but simply classification
        self.pred_store_scores = False  # store the scores for each role
        self.pred_no_change = False  # no actual changing (usually used with 'pred_store_scores')
        self.pred_thr = 0.  # >=0 prob, <0 logprob
        self.clear_arg_ef = True  # when clear args, also rm ef if no as_args
        # decoding for gen
        self.gen_beam_size = 1
        self.gen_kwargs = {}  # penalties,rewards,...: see 'bmod_beam_search'
        # special mlm mode
        self.do_mlm = False
        self.mlm_mrate = 0.15  # how much to mask?
        self.mlm_repl_ranges = [0.8, 0.9]  # cumsum: [MASK], random, remaining unchanged!
        # special aug for training
        self.aug_rate = 0.
        self.aug_conf = ArgAugConf()
        # distill
        self.distill_alphas = [0., 0.]  # for missing/existing roles
        self.distill_topk_thr = 0.  # topk thr
        self.distill_topk_k = 2  # topk k
        self.distill_topk_wa = 0.  # topk weight (* (sum-topk-prob ** wa))
        self.distill_dw = -1.  # extra dynamic weight (missing = dw * existing)
        self.distill_beta = 0.  # score -= max(other-existing-cands>0) * beta
        self.distill_tau = 1.  # exp(score/tau)
        # --

@node_reg(ZModArgConf)
class ZModArg(ZModBaseE):
    def __init__(self, conf: ZModArgConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModArgConf = self.conf
        # --
        _onto = ztask.get_onto()  # re-get every time
        _mdim = self.bmod.get_mdim()
        self.qmod = QmodLayer(conf.qconf, _onto, self.bmod.sub_toker, mdim=_mdim)
        self.repr = conf.repr_conf.make_node(arg_layer=self)
        self.sim = conf.sim_conf.make_node(isize=_mdim)
        self.neg_delta = conf.neg_delta.make_node()  # []
        if conf.mix_evt_ind_share:  # also share this one!!
            _mod = zmodel.get_mod(conf.shared_bmod_name)
            assert isinstance(_mod, ZModArg)
            self.evt_ind = _mod.evt_ind
        else:
            self.evt_ind = BK.get_emb_with_initscale(2, _mdim, conf.mix_evt_ind_initscale)
        self.upos_filter = None if len(conf.upos_filter)==0 else \
            ZRuleFilter(conf.upos_filter, {"N": {'NOUN', 'PRON', 'PROPN'}}, default_value=True)
        self.extender = SimpleSpanExtender.get_extender(conf.extend_span_sig) if conf.extend_span else None
        # --
        _modes = [conf.arg_mode==z for z in ['clf', 'mrc', 'tpl', 's2s', 'gen']]
        assert sum(_modes) == 1, "Invalid arg mode!"
        self.qmod_query_f = getattr(self.qmod, f"query_{conf.arg_mode}")
        self.is_mode_clf, self.is_mode_mrc, self.is_mode_tpl, self.is_mode_s2s, self.is_mode_gen = _modes
        # check
        if conf.mrc_pool_ques:
            assert self.is_mode_mrc
        if (self.is_mode_s2s and conf.s2s_do_atr) or self.is_mode_gen:
            assert all(z==0 for z in conf.pred_nw[:1]+conf.pred_nw[2:]+conf.loss_nw[:1]+conf.loss_nw[2:])
            self.atr_helper = ArgAtrHelper(self)
        else:
            self.atr_helper = None
        self.aug_helper = ArgAugHelper(conf.aug_conf, self)
        # special mix_evt_ind
        _tmp = eval(conf.mix_evt_ind)
        if isinstance(_tmp, (float, int)):
            self.mix_evt_ind = {None: float(_tmp)}
        else:
            self.mix_evt_ind = _tmp  # str -> float, None -> default
        zlog(f"Build mix_evt_ind with {self.mix_evt_ind}")
        # ctx-sent rates
        self.ctx_nsent_rates = [z/sum(conf.ctx_nsent_rates) for z in conf.ctx_nsent_rates]
        self.test_ctx_nsent = [i for i,z in enumerate(self.ctx_nsent_rates) if z>0][-1]
        zlog(f"ctx_nsent: train={self.ctx_nsent_rates}, test={self.test_ctx_nsent}")
        # pred thr
        self.pred_thr_logprob = conf.pred_thr if conf.pred_thr<0 else math.log(conf.pred_thr+1e-5)
        zlog(f"pred_thr_logprob={self.pred_thr_logprob}")
        # --

    def do_loss(self, rc: ZRunCache, *args, **kwargs): return self.do_forward(rc.ibatch, is_testing=False)
    def do_predict(self, rc: ZRunCache, *args, **kwargs): return self.do_forward(rc.ibatch, is_testing=True)

    def do_forward(self, ibatch, is_testing: bool):
        conf: ZModArgConf = self.conf
        _pad_id = self.tokenizer.pad_token_id
        all_evts, all_contents, query_res = self.prepare_ibatch(
            ibatch, clear_args=(is_testing and not conf.pred_given_posi and not conf.pred_no_change), is_testing=is_testing)
        if is_testing and len(all_evts)==0:
            zwarn(f"Input batch turns out to be 0 in testing, originally: {[z.frame for z in ibatch.items]}")
            return {}
        # --
        if conf.do_mlm:
            ids_query = [[self.tokenizer.cls_token_id]] * len(all_evts)
            _, full_contents = self._prep_t_content(all_contents, ids_query)  # [?bs, Lf?]
            return self._forward_mlm(all_evts, full_contents, is_testing)
        # --
        # compose & augment input with seq0
        cur_contents = all_contents
        orig_contents = None
        ids_query = query_res[2]  # L[bs, lq]
        if self.is_mode_mrc:  # re-batch to qbs
            arr_bidx = query_res[4]
            cur_contents = [cur_contents[ii] for ii in arr_bidx]  # re-arrange bidx
            _, orig_contents = self._prep_t_content(all_contents, [[]] * len(ids_query))  # [bs, Lf1']
        elif self.is_mode_s2s or self.is_mode_gen:  # only add [cls]
            if not conf.encq_sg:  # if not encoding query, then simply put [CLS]
                ids_query = [[self.tokenizer.cls_token_id]] * len(ids_query)
        t_len_seq0, full_contents = self._prep_t_content(cur_contents, ids_query)  # [?bs, Lf?]
        # --
        # forward seq0 (with enc) to get contents
        _ids_t = full_contents[0]  # [?bs, Lf0]
        _ids_mask_t = (_ids_t != _pad_id).float()
        _mixes = []  # mix embs
        # mix reprs for certain modes (mix=1)
        if self.is_mode_mrc or self.is_mode_tpl:
            arr_e_idx, t_e = query_res[-1]  # [?bs, lq, ??]
            _e_mix = self._prep_e_mix(arr_e_idx, t_e, t_mask=_ids_mask_t)
            if _e_mix is not None:  # note: mixings for the prefixes!
                _mixes.append(_e_mix)
        # mix evt indicator?
        t_ifr = BK.input_idx(full_contents[4])  # [?bs, Lf1]
        # --
        _mix_evt_ind = self.mix_evt_ind[None]
        for _k, _v in self.mix_evt_ind.items():
            if _k is not None and _k in ibatch.dataset.name:
                _mix_evt_ind = _v  # use this one!
                break
        # --
        if _mix_evt_ind > 0.:
            t_sublen = full_contents[1]  # [?bs, Lf1]
            t_ifr1 = (t_ifr>0).long()  # [?bs, Lf1]
            t_ifr0 = _ids_t * 0  # [?bs, Lf0]
            _scatter_idx = t_sublen.cumsum(-1) - 1 - (t_sublen-1).clamp(min=0)  # note: simply assign only first subword!
            t_ifr0.scatter_(-1, _scatter_idx, t_ifr1)
            mix_w_t, mix_emb_t = t_ifr0.float() * _mix_evt_ind, self.evt_ind(t_ifr0)  # [?bs, Lf0]
            _mixes.append((mix_w_t, mix_emb_t))
        bert_out = self.bmod.forward_enc(_ids_t, _ids_mask_t, _mixes)
        t_content = self.repr(bert_out, full_contents[1], full_contents[2], t_ifr)  # [?bs, Lf1, D], cand reprs
        # --
        if self.is_mode_gen:  # note: special gen mode!
            return self._forward_gen(_ids_t, _ids_mask_t, bert_out, full_contents, query_res, all_evts, is_testing)
        # --
        # get queries & query scores
        t_rids = BK.input_idx(query_res[1])  # [bs, R]
        if self.is_mode_clf:
            t_query = self.qmod.get_repr_role(t_rids)  # [bs, R, D]
            t_score = self.sim(t_query, t_content)  # [bs, R, Lf1]
        elif self.is_mode_mrc:
            _arange_t = BK.arange_idx(len(t_content))  # [qbs]
            if conf.mrc_pool_ques:
                _a2 = BK.arange_idx(BK.get_shape(t_content, 1)).unsqueeze(0)  # [1, L]
                _qm = ((_a2>0) & (_a2<t_len_seq0.unsqueeze(-1))).float()  # [qbs, L]
                _qm = _qm / _qm.sum(-1, keepdims=True)
                t_query0 = (t_content * _qm.unsqueeze(-1)).sum(-2, keepdims=True)  # [qbs, 1, D]
            else:  # pick one
                t_query0 = t_content[_arange_t, BK.input_idx(query_res[3])].unsqueeze(-2)  # [qbs, 1, D]
            t_score0 = self.sim(t_query0, t_content).squeeze(-2)  # [qbs, Lf1]
            full_contents = orig_contents  # note: switch to orig_contents!
            _idx0 = BK.input_idx(query_res[5]).unsqueeze(-1)  # [bs, R, 1]
            _len1 = BK.get_shape(full_contents[1], -1)  # Lf1'
            _idx1 = BK.arange_idx(_len1) + t_len_seq0[_idx0]  # [bs, R, Lf1']
            _idx1.clamp_(max=BK.get_shape(t_score0, -1)-1)
            t_score = t_score0[_idx0, _idx1]  # [bs, R, Lf1']
        elif self.is_mode_tpl:
            _arange_t = BK.arange_idx(len(t_content)).unsqueeze(-1)  # [bs, 1]
            t_query = t_content[_arange_t, BK.input_idx(query_res[3])]  # [bs, R, D]
            t_score = self.sim(t_query, t_content)  # [bs, R, Lf1]
        elif self.is_mode_s2s:
            # prepare target seq
            _t_trg_ids = BK.input_idx(DataPadder.go_batch_2d(query_res[2], _pad_id))  # [bs, lq]
            # add e_mixes
            _mixes = []
            arr_e_idx, t_e = query_res[-1]  # [?bs, lq, ??]
            _e_mix = self._prep_e_mix(arr_e_idx, t_e, t_mask=None)  # no need for mask!
            if _e_mix is not None:  # note: mixings for the prefixes!
                _mixes.append(_e_mix)
            # --
            t_cand_emb = None
            if conf.s2s_do_atr:  # used as dec input
                _t_cand_ids = self.repr.sub_pooler.forward_hid(_ids_t, full_contents[1], pool_f='first')  # [bs, Lf1]
                t_cand_emb = self.bmod.forward_emb(_t_cand_ids)  # [bs, Lf1, D]
            # --
            if conf.s2s_do_atr and is_testing:  # note: special decoding
                t_qr = BK.input_idx(query_res[-2])  # [bs, lq]
                ret = self.atr_helper.decode_s2s(
                    _t_trg_ids, bert_out.last_hidden_state, _ids_mask_t, _mixes,
                    t_qr, t_content, full_contents[3], t_cand_emb, all_evts, full_contents[2])
                return ret
            # --
            # mix for atr
            if conf.s2s_do_atr:
                t_qr = BK.input_idx(query_res[-2]).unsqueeze(-1)  # [bs, lq, 1]
                _g = ((t_qr > 0) & (t_qr == full_contents[5].unsqueeze(-2))).float()  # [*, lq, C]
                _w = _g / _g.sum(-1, keepdims=True).clamp(min=1.)  # simply average if there are more than one!
                _e = BK.matmul(_w, t_cand_emb) * conf.atr_mix_scale  # [*, lq, D]
                _mixes.append(((_g.sum(-1)>0).float(), _e))  # [*, lq], note: only mix if having answers
                # for query, we need do at the previous subtoken!
                t_rq = (BK.input_idx(query_res[3]) - 1).clamp(min=0)  # [bs, R]
            else:
                t_rq = BK.input_idx(query_res[3])  # [bs, R]
            # forward for the decoder
            # --
            # debug
            # _DEBUG = True  # note: also need to disable dropout!!
            # if _DEBUG: self.eval()
            # --
            # todo(+2): currently simply use the raw outputs!
            trg_out = self.bmod.forward_dec(
                _t_trg_ids, t_cross=bert_out.last_hidden_state, t_cross_mask=_ids_mask_t, mixes=_mixes)
            t_trg = trg_out.last_hidden_state
            # --
            # debug
            # if _DEBUG:
            #     _, all_res = self.atr_helper.decode_s2s(
            #         _t_trg_ids, bert_out.last_hidden_state, _ids_mask_t, _mixes, BK.input_idx(query_res[-2]),
            #         t_content, full_contents[3], all_evts, full_contents[2], t_gold=full_contents[5], ret_step_res=True)
            #     _res_out = BK.concat([z.last_hidden_state for z in all_res], -2)  # [bs, lq-1, D]
            #     _dd = (_res_out - t_trg[:,:-1]).abs()
            #     breakpoint()
            # --
            # score
            _arange_t = BK.arange_idx(len(t_trg)).unsqueeze(-1)  # [bs, 1]
            t_query = t_trg[_arange_t, t_rq]  # [bs, R, D]
            t_score = self.sim(t_query, t_content)  # [bs, R, Lf1]
        else:
            raise NotImplementedError("Unknown arg-mode!!")
        # --
        # train / test
        t_vcands, t_irr = full_contents[3], full_contents[5]
        arr_roles, arr_toks = query_res[0], full_contents[2]
        if conf.debug_print:
            zlog("#-- Current batch:")
            for _ids in _ids_t:
                zlog(self.tokenizer.decode(_ids))
        if is_testing:
            ret = self._pred(t_score, t_rids, t_vcands, all_evts, arr_roles, arr_toks)
        else:
            ret = self._loss(t_score, t_rids, t_vcands, t_irr, all_evts, arr_roles, arr_toks)
        return ret
        # --

    # special forward for mlm mode
    def _forward_mlm(self, all_evts, full_contents, is_testing: bool):
        conf: ZModArgConf = self.conf
        _tokenizer = self.tokenizer
        _pad_id, _cls_id, _sep_id, _mask_id = \
            _tokenizer.pad_token_id, _tokenizer.cls_token_id, _tokenizer.sep_token_id, _tokenizer.mask_token_id
        _ids_t = full_contents[0]  # [?bs, Lf0]
        _ids_mask_t = (_ids_t != _pad_id).float()
        # -- prepare mlm
        _shape = _ids_t.shape
        # sample mask
        mlm_mask = ((BK.rand(_shape) < conf.mlm_mrate) & (_ids_t != _cls_id) & (_ids_t != _sep_id)).float() \
                   * _ids_mask_t  # [*, elen]
        # sample repl
        _repl_sample = BK.rand(_shape)  # [*, elen], between [0, 1)
        mlm_repl_ids = BK.constants_idx(_shape, _mask_id)  # [*, elen] [MASK]
        _repl_rand, _repl_origin = conf.mlm_repl_ranges
        mlm_repl_ids = BK.where(_repl_sample > _repl_rand, (BK.rand(_shape) * _tokenizer.vocab_size).long(), mlm_repl_ids)
        mlm_repl_ids = BK.where(_repl_sample > _repl_origin, _ids_t, mlm_repl_ids)
        mlm_input_ids = BK.where(mlm_mask > 0., mlm_repl_ids, _ids_t)  # [*, elen]
        # forward & loss
        bert_out = self.bmod.forward_enc(mlm_input_ids, _ids_mask_t)
        t_out = self.bmod.forward_lmhead(bert_out.last_hidden_state)  # [*, elen, V]
        t_loss = BK.loss_nll(t_out, _ids_t, label_smoothing=conf.label_smoothing)  # [bs, elen]
        loss_item = LossHelper.compile_leaf_loss("argMlm", (t_loss * mlm_mask).sum(), mlm_mask.sum(), loss_lambda=1.)
        if is_testing:
            t_sloss = (t_loss * mlm_mask).sum(-1) / mlm_mask.sum(-1).clamp(min=1.)  # [bs]
            l_sloss = BK.get_value(t_sloss).tolist()
            for bidx, evt in enumerate(all_evts):
                evt.info['loss_mlm'] = l_sloss[bidx]
            return {}
        else:
            ret = LossHelper.compile_component_loss(self.name, [loss_item])
            return ret, {}
        # --

    # special forward for gen!
    def _forward_gen(self, _ids_t, _ids_mask_t, bert_out, full_contents, query_res, all_evts, is_testing):
        conf: ZModArgConf = self.conf
        if is_testing:
            _tokenizer = self.tokenizer
            _pad_id = _tokenizer.pad_token_id
            # prepare predicting V-mask: note: assume shared vocab for enc&dec
            vmask = BK.zeros([len(_ids_t), _tokenizer.vocab_size])
            _add_ids = [_tokenizer.mask_token_id, _tokenizer.sep_token_id] \
                       + sum([_tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(z)) for z in ["and", " and"]], [])
            vmask[:, _add_ids] = 1.  # special ones
            _t_arange = BK.arange_idx(len(_ids_t)).unsqueeze(-1)  # [bs, 1]
            vmask[_t_arange, _ids_t] = 1.  # from src
            _t_query_ids = BK.input_idx(DataPadder.go_batch_2d(query_res[2], _pad_id))
            vmask[_t_arange, _t_query_ids] = 1.  # from template
            return self.atr_helper.decode_gen(bert_out.last_hidden_state, _ids_mask_t, vmask,
                                              all_evts, full_contents[2], query_res[0], query_res[3], query_res[2])
        else:  # simply cross-entropy
            _pad_id = self.tokenizer.pad_token_id
            _trg_ids_t = BK.input_idx(DataPadder.go_batch_2d(query_res[-1], _pad_id))  # [bs, lt]
            trg_out = self.bmod.forward_dec(_trg_ids_t, t_cross=bert_out.last_hidden_state, t_cross_mask=_ids_mask_t)
            t_out = self.bmod.forward_lmhead(trg_out.last_hidden_state)  # [bs, lt, V]
            t_loss = BK.loss_nll(t_out[:, :-1], _trg_ids_t[:, 1:], label_smoothing=conf.label_smoothing)  # [bs, lt-1]
            t_w = (_trg_ids_t != _pad_id).float()[:, 1:]  # [bs, lt-1]
            loss_item = LossHelper.compile_leaf_loss("argG", (t_loss * t_w).sum(), t_w.sum(), loss_lambda=1.)
            ret = LossHelper.compile_component_loss(self.name, [loss_item])
            return ret, {}
        # --

    def _prep_e_mix(self, arr_e_idx, t_e, t_mask=None):
        if t_e is None:
            return None
        mix_w_t, mix_emb_t = (BK.input_idx(arr_e_idx) > 0).float(), t_e
        if t_mask is not None:
            _pad_n = BK.get_shape(t_mask, -1) - BK.get_shape(mix_w_t, -1)  # simply pad to full length
            mix_w_t = BK.pad(mix_w_t, [0, _pad_n], value=0.) * t_mask  # [?bs, Lf0]
            mix_emb_t = BK.pad(mix_emb_t, [0, 0, 0, _pad_n], value=0.)
        return mix_w_t, mix_emb_t

    def _prep_t_content(self, contents, ids_seq0):
        lens_seq0 = []
        _pad_id = self.tokenizer.pad_token_id
        ids_full = []
        for one_c, s0_ids in zip(contents, ids_seq0):
            lens_seq0.append(len(s0_ids))
            ids_full.append(s0_ids + sum(one_c[0], []))  # concat together
        # --
        arr_len_seq0 = np.asarray(lens_seq0)  # [?bs]
        arr_ids_full = DataPadder.go_batch_2d(ids_full, _pad_id)  # [?bs, Lf1]
        t_len_seq0 = BK.input_idx(arr_len_seq0)  # [?bs]
        t_ids_full = BK.input_idx(arr_ids_full)  # [?bs, Lf1]
        full_contents = [t_ids_full]
        for ii, (df0, df1, dt) in enumerate(
                zip([1, None, 0., 0, 0], [0, None, 0., 0, 0], [np.long, object, np.float32, np.long, np.long]), 1):
            all_ones = [z[ii] for z in contents]
            _max_len = max(a+len(b) for a,b in zip(lens_seq0, all_ones))
            arr_full_ones = np.full([len(all_ones), _max_len], fill_value=df1, dtype=dt)
            for bidx, (bl0, ones) in enumerate(zip(lens_seq0, all_ones)):
                arr_full_ones[bidx, :bl0] = df0
                arr_full_ones[bidx, bl0:(bl0+len(ones))] = ones
            if df0 is None:  # simply arr
                full_contents.append(arr_full_ones)
            else:
                t_full_ones = BK.input_idx(arr_full_ones) if isinstance(df0, int) else BK.input_real(arr_full_ones)
                full_contents.append(t_full_ones)
        return t_len_seq0, full_contents  # [?bs], L[?bs, Lf]

    # [..., L, ...], 2x[..., L, ...], [..., L, ...]
    def _process_one(self, t_score, t_golds, t_mask, dim: int, loss_name: str, loss_lambda: float, nratio: float):
        ls = self.conf.label_smoothing
        # --
        _NEG = -10000.
        score = t_score + (1.-t_mask) * _NEG
        t_logprob = score.log_softmax(dim=dim)
        if t_golds is not None:  # sum of log
            t_gold0, t_gold1 = t_golds
            t_mask_sum = t_mask.sum(dim=dim, keepdims=True)  # [..., 1, ...]
            t_gold_sum = t_gold1.sum(dim=dim, keepdims=True)  # [..., 1, ...]
            t_trg = ls * t_mask / t_mask_sum.clamp(min=1.) + (1.-ls) * t_gold1 / t_gold_sum.clamp(min=1.)
            t_loss = - (t_logprob * t_trg).sum(dim=dim)  # [...]
            t_w0 = (t_mask_sum.squeeze(dim=dim) > 0).float()  # [...]
            # --
            if nratio >= 0.:
                t_ispos = (t_gold0.select(dim, 0) == 0.).float() * t_w0  # [...], idx0 means NIL
                t_w = down_neg(t_w0, t_ispos, nratio, do_sample=False)  # [...]
            else:
                t_w = t_w0
            # --
            loss_item = LossHelper.compile_leaf_loss(
                loss_name, (t_loss * t_w).sum(), t_w.sum(), loss_lambda=loss_lambda)
        else:
            loss_item = None
        return t_logprob, loss_item

    # prepare target
    def _prepare_target(self, t_gold, t_dtrg, t_valid, dim: int):
        t_slice0 = (~t_gold.any(dim, keepdims=True))  # [..., 1, ...], NIL
        ret0 = BK.concat([t_slice0, t_gold], dim).float()  # [..., 1+?, ...]
        if t_dtrg is not None:
            conf: ZModArgConf = self.conf
            # prep the two
            _thr = 0.  # todo(+1): currently simply put 0.
            t_dtrg = t_dtrg.view_as(t_gold)  # [..., ?, ...]
            tf_slice0 = t_slice0.float()  # [..., 1, ...], is-NIL
            t_dp = BK.concat([tf_slice0*0.+_thr, t_dtrg], dim).softmax(dim)  # [..., 1+?, ...]
            t_rr = ret0 / ret0.sum(dim, keepdims=True).clamp(min=1)  # [..., 1+?, ...]
            # mix
            _alpha0, _alpha1 = conf.distill_alphas
            _dw = conf.distill_dw
            if _dw > 0:
                _c_valid = (t_valid.sum(dim=dim, keepdims=True)>0).float()  # [..., 1, ...]
                _extra_alpha0 = (1.-tf_slice0).sum() * _dw / (_c_valid * tf_slice0).sum().clamp(min=0)
                _alpha0 = _extra_alpha0 * _alpha0
            mix_rate = (tf_slice0 * _alpha0 + (1.-tf_slice0) * _alpha1).clamp(min=0., max=1.)  # [..., 1, ...]
            _tk_thr, _tk_wa = conf.distill_topk_thr, conf.distill_topk_wa
            _tk_k = min(conf.distill_topk_k, BK.get_shape(t_dp, dim)-1)
            if (_tk_thr>0. or _tk_wa!=0.) and _tk_k>0:
                _tok_p = t_dp.narrow(dim, 1, BK.get_shape(t_dp, dim)-1).topk(_tk_k, dim=dim)[0]
                _tok_ps = _tok_p.sum(dim=dim, keepdims=True)
                _thr_pass = (_tok_ps > _tk_thr).float()  # [...1...], pass the thr
                mix_rate = mix_rate * _thr_pass
                if _tk_wa != 0.:
                    mix_rate = mix_rate * (_tok_ps ** _tk_wa)
            ret1 = mix_rate * t_dp + (1.-mix_rate) * t_rr  # [..., 1+?, ...]
        else:
            ret1 = ret0
        return ret0, ret1

    # process all of them: [3], [], [*, R, C], [*, R], [*, C], 2x[*, R, C]
    def _process_them(self, nw, neg_t, t_score, t_rids, t_vcands, t_gold, t_dtrg, nr=None):
        w_all, w_role, w_cand = nw
        nr_all, nr_role, nr_cand = nr if nr is not None else (-1., -1., -1.)
        t_logprob, _loss_items = 0., []
        _shape = BK.get_shape(t_score)  # [bs, R, C]
        t_valid = t_vcands.unsqueeze(-2) * (t_rids > 0).float().unsqueeze(-1)  # [*, R, C]
        if w_all > 0.:
            _shape1 = _shape[:-2] + [_shape[-1] * _shape[-2]]  # [bs, R*C]
            _shape2 = _shape[:-2] + [1]  # [bs, 1]
            _score = BK.concat([(BK.zeros(_shape2) + neg_t), t_score.view(_shape1)], -1)  # [bs, 1+R*C]
            if t_gold is None:
                _gold = None
            else:
                _gold0 = t_gold.view(_shape1)
                _gold = self._prepare_target(_gold0, t_dtrg, t_valid, -1)  # [bs, 1+R*C]
            _mask0 = t_valid.view(_shape1)
            _mask = BK.concat([(_mask0.sum(-1, keepdims=True) > 0).float(), _mask0], -1)  # [bs, 1+R*C]
            _logprob, _item = self._process_one(_score, _gold, _mask, -1, "argA", w_all, nr_all)
            t_logprob = t_logprob + w_all * _logprob[:,1:].view(_shape)  # [bs, R, C]
            _loss_items.append(_item)
        if w_role > 0.:
            _shape2 = list(_shape)
            _shape2[-1] = 1  # [bs, R, 1]
            _score = BK.concat([(BK.zeros(_shape2) + neg_t), t_score], -1)  # [bs, R, 1+C]
            if t_gold is None:
                _gold = None
            else:
                _gold = self._prepare_target(t_gold, t_dtrg, t_valid, -1)  # [bs, R, 1+C]
            _mask = BK.concat([(t_valid.sum(-1, keepdims=True) > 0).float(), t_valid], -1)  # [bs, R, 1+C]
            _logprob, _item = self._process_one(_score, _gold, _mask, -1, "argR", w_role, nr_role)
            t_logprob = t_logprob + w_role * _logprob[:,:,1:].view(_shape)  # [bs, R, C]
            _loss_items.append(_item)
        if w_cand > 0.:
            _shape2 = list(_shape)
            _shape2[-2] = 1  # [bs, 1, C]
            _score = BK.concat([(BK.zeros(_shape2) + neg_t), t_score], -2)  # [bs, 1+R, C]
            if t_gold is None:
                _gold = None
            else:
                _gold = self._prepare_target(t_gold, t_dtrg, t_valid, -2)  # [bs, 1+R, C]
            _mask = BK.concat([(t_valid.sum(-2, keepdims=True) > 0).float(), t_valid], -2)  # [bs, 1+R, C]
            _logprob, _item = self._process_one(_score, _gold, _mask, -2, "argC", w_cand, nr_cand)
            t_logprob = t_logprob + w_cand * _logprob[:,1:,:].view(_shape)  # [bs, R, C]
            _loss_items.append(_item)
        # --
        return t_logprob, t_valid, _loss_items

    # obtain loss
    # [*, R, C], [*, R], [*, C], [*, C];; [*], [*, R], [*, C]
    def _loss(self, t_score, t_rids, t_vcands, t_irr, evts, arr_roles, arr_toks):
        conf: ZModArgConf = self.conf
        # --
        # down-sample?
        if conf.train_neg_sample >= 0.:
            t_vcands = down_neg(t_vcands, (t_irr>0).float(), conf.train_neg_sample, do_sample=True)  # [*, C]
        # --
        t_gold = (t_rids.unsqueeze(-1) > 0) & (t_rids.unsqueeze(-1) == t_irr.unsqueeze(-2))  # [*, R, C]
        _neg_t = self.neg_delta()  # []
        # --
        # get distill targets
        if any(z>0 for z in conf.distill_alphas):
            _alpha0, _alpha1 = conf.distill_alphas
            _beta = conf.distill_beta
            _tau = conf.distill_tau
            _NINF = -100.  # this should be enough!
            _arr_dtrg = np.full(BK.get_shape(t_score), fill_value=_NINF, dtype=np.float32)  # [*, R, C]
            for bidx, (evt, rr, tt) in enumerate(zip(evts, arr_roles, arr_toks)):
                if 'arg_scores' not in evt.info:
                    zwarn(f"No arg_scores found in {evt}!!")
                    continue
                for rii, _one_rr in enumerate(rr):
                    if _one_rr is None: continue
                    ss = evt.info['arg_scores'][_one_rr.name]  # tok->score should be there!
                    for tii, _one_tt in enumerate(tt):
                        if _one_tt is None: continue
                        _vv = ss.get(_one_tt.get_indoc_id(True))
                        if _vv is not None:  # find one!
                            _arr_dtrg[bidx, rii, tii] = _vv
            t_dtrg = BK.input_real(_arr_dtrg)  # [*, R, C]
            if _beta > 0.:  # decrease explicit cands
                _thr = 0.  # todo(+1): currently simply put 0.
                _tf_gold = t_gold.float()
                _t_exc = ((_tf_gold.sum(-2, keepdims=True) - _tf_gold) > 0).float() * (1.-_tf_gold)  # [*, R, C]
                _t_v = (t_dtrg + (1.-_t_exc) * _NINF).max(-1, keepdims=True)[0].clamp(min=0.)  # [*, R, 1]
                t_dtrg = t_dtrg - _beta * _t_v  # decrease!
            if _tau != 1.:  # scale
                t_dtrg = t_dtrg / _tau
        else:
            t_dtrg = None
        # --
        _, _, _loss_items = self._process_them(
            conf.loss_nw, _neg_t, t_score, t_rids, t_vcands, t_gold, t_dtrg, nr=conf.train_neg_ratios)
        ret = LossHelper.compile_component_loss(self.name, _loss_items)
        return ret, {}

    # do (non-s2s) prediction
    # [*, R, C], [*, R], [*, C];; [*], [*, R], [*, C]
    def _pred(self, t_score, t_rids, t_vcands, evts, arr_roles, arr_toks):
        conf: ZModArgConf = self.conf
        # --
        _neg_t = self.neg_delta() + conf.pred_neg_delta  # []
        t_logprob, t_valid, _ = self._process_them(conf.pred_nw, _neg_t, t_score, t_rids, t_vcands, None, None)  # [*, R, C]
        # --
        if conf.pred_given_posi:  # simply do classification
            for bidx, (evt, roles, toks) in enumerate(zip(evts, arr_roles, arr_toks)):
                _s = BK.get_value(t_logprob[bidx].transpose(-1, -2))  # [C, R]
                _tmap = {id(t): i for i,t in enumerate(toks) if t is not None}
                for arg in evt.args:  # note: already there!
                    _atok = arg.mention.shead_token
                    _ii = _tmap.get(id(_atok))
                    arg.info['orig_label'] = arg.label
                    if _ii is None:
                        arg.set_label("UNK")
                        zwarn(f"Arg position out of range: {evt} {arg}")
                    else:  # todo(+N): consider NIL? Currently no need for the use of this mode ...
                        _ascore = _s[_ii]
                        arg.set_label(roles[_ascore.argmax()].name)
                        arg.set_score(_ascore.max().item())
        else:
            # decode: simply do greedy
            # note: another approximation
            _br, _bc = conf.pred_br, conf.pred_bc
            _br_margin = conf.pred_br_margin
            _k = min(_br+_bc, t_logprob.shape[-1])
            # [*, R, C], note: no need for idx0 since ">_neg_t" already test it!
            t_topk = (t_logprob >= t_logprob.topk(_k, dim=-1)[0].min(-1, keepdims=True)[0]).float() \
                     * (t_logprob >= (t_logprob.max(-1, keepdims=True)[0] - _br_margin)).float() \
                     * (t_logprob >= self.pred_thr_logprob).float()
            t_valid = t_valid * t_topk * (t_score > _neg_t).float()  # [*, R, C]
            # only consider these ones!
            arr_score = BK.get_value(t_score)  # [*, R, C]
            for bidx, (evt, roles, toks) in enumerate(zip(evts, arr_roles, arr_toks)):
                _r, _c = t_valid[bidx].nonzero(as_tuple=True)  # [??]
                _s = t_logprob[bidx][_r, _c]  # [??]
                _them = sorted(zip(*[z.tolist() for z in [_s, _r, _c]]), reverse=True)
                _count_r, _count_t = [0]*len(roles), [0]*len(toks)
                # --
                # store scores
                if conf.pred_store_scores:
                    arg_scores = {}
                    for rii, rr in enumerate(roles):
                        if rr is None: continue
                        rr_scores = {}
                        for tii, tok in enumerate(toks):
                            if tok is None: continue
                            rr_scores[tok.get_indoc_id(True)] = round(arr_score[bidx][rii][tii].item(), 3)
                        arg_scores[rr.name] = rr_scores
                    evt.info['arg_scores'] = arg_scores
                # --
                if not conf.pred_no_change:
                    for _ss, _rr, _cc in _them:  # greedy
                        if _count_r[_rr] < _br and _count_t[_cc] < _bc:  # ok
                            _role, _tok = roles[_rr], toks[_cc]
                            new_ef = _tok.sent.make_entity_filler(_tok.widx, 1)
                            if self.extender is not None:
                                self.extender.extend_mention(new_ef.mention)
                            evt.add_arg(new_ef, role=_role.name, score=_ss)
                            _count_r[_rr] += 1
                            _count_t[_cc] += 1
        # --
        return {}

    # --
    # prepare input instances

    def prepare_ibatch(self, ibatch, clear_args=False, is_testing=False, no_cache=False):
        _key = f"_cache_{self.name}"
        conf: ZModArgConf = self.conf
        # --
        _gen0 = Random.get_generator("train")
        _gen1 = Random.stream(_gen0.random_sample)
        _tokenizer = self.tokenizer
        _mask_id, _pad_id = _tokenizer.mask_token_id, _tokenizer.pad_token_id
        # --
        # mostly in testing, we should clear current results!
        if clear_args:
            for item in ibatch.items:
                # clear args (and also ef)
                for arg in list(item.frame.args):  # note: remember to copy!
                    arg.delete_self()
                    if len(arg.arg.as_args) == 0 and conf.clear_arg_ef:
                        arg.arg.sent.delete_frame(arg.arg, 'ef')
        # --
        # collect them!
        all_evts, all_contents = [], []
        for item in ibatch.items:
            # --
            if self.qmod.evt2frame(item.frame) is None:
                zwarn(f"Ignoring UNK frame: {item.frame}")
                continue
            # --
            # get contents
            if is_testing:
                _ctx_nsent = self.test_ctx_nsent
            elif len(self.ctx_nsent_rates) <= 1:
                _ctx_nsent = 0
            else:
                _ctx_nsent = int(_gen0.choice(len(self.ctx_nsent_rates), p=self.ctx_nsent_rates))
            content_cache = item.info.get((_key, _ctx_nsent))
            if content_cache is None or no_cache:
                content_cache = self.prepare_content(item, _ctx_nsent)
                if not no_cache:
                    item.info[(_key, _ctx_nsent)] = content_cache
            # do some transformations as augmentation?
            if not is_testing and conf.aug_rate > 0.:
                if next(_gen1) < conf.aug_rate:
                    content_cache = self.aug_helper.do_aug(item, content_cache, _mask_id, no_cache=no_cache)
            # --
            all_evts.append(item.frame)
            all_contents.append(content_cache)
        # --
        # prepare queries
        query_res = self.qmod_query_f(all_evts, is_testing)
        return all_evts, all_contents, query_res

    def prepare_content(self, item, ctx_nsent: int):
        conf: ZModArgConf = self.conf
        # --
        # prepare for one item
        sent = item.sent
        if sent.doc is not None:  # simply set them all!
            set_ee_heads(sent.doc.sents)
        else:
            set_ee_heads(sent)
        # prepare center sent & ctx
        center_ids, center_toks = self.prep_sent(sent, ret_toks=True)  # get subtokens, [olen], List[List]
        before_ids, after_ids, before_toks, after_toks = self.extend_ctx_sent(sent, ctx_nsent, ret_toks=True)
        seq_ids, seq_toks = before_ids + center_ids + after_ids, before_toks + center_toks + after_toks
        _len = len(seq_ids)
        if conf.ctx_as_trg:
            seq_vcands = [1.] * _len
        else:
            seq_vcands = [0.] * len(before_ids) + [1.] * len(center_ids) + [0.] * len(after_ids)
        # --
        if self.upos_filter is not None:
            for _ii, _tt in enumerate(seq_toks):
                if _tt is None or (not self.upos_filter.filter_by_name(_tt.upos)):
                    seq_vcands[_ii] = 0.
        # --
        # put frame and args in!
        frame = item.frame
        tok_map = {id(t):i for i,t in enumerate(seq_toks)}
        seq_frames, seq_args = [None]*_len, [None]*_len
        seq_ifr, seq_irr = [0]*_len, [0]*_len  # indicator of frame, indexes of roles, note: 0 means NIL!
        truncate_center = tok_map[id(frame.mention.shead_token)]  # note: must be there!
        seq_frames[truncate_center] = frame
        seq_ifr[truncate_center] = self.qmod.evt2frame(frame).idx  # >0
        for arg in frame.args:
            role = self.qmod.arg2role(arg)
            if role is None: continue  # ignore inactive ones!
            _k = id(arg.mention.shead_token)
            if _k not in tok_map:
                zwarn(f"Cannot find {arg} in the current seq of {frame}!")
                continue
            _ii = tok_map[_k]
            seq_args[_ii] = arg  # put it!
            seq_irr[_ii] = role.idx  # put it!
            seq_vcands[_ii] = 1.  # note: also enforce valid!
        # --
        # conj?
        if conf.add_arg_conj:
            c_cc = Counter()
            for arg, ridx in zip(seq_args, seq_irr):
                if arg is None: continue
                tok = arg.mention.shead_token
                # --
                # # previously judge by pred-descendant
                # spine_widxes = tok.sent.tree_dep.get_spine(tok.widx)
                # if tok.sent is frame.sent and frame.mention.shead_widx in spine_widxes:  # must be descendant
                # --
                _span_range = arg.mention.widx, arg.mention.wridx  # simply judge by span range!
                if 1:
                    for tok2 in tok.ch_toks:
                        if tok2.deplab in ['conj', 'appos'] and tok2.widx >= _span_range[0] and tok2.widx < _span_range[1]:
                            _k = id(tok2)
                            if _k in tok_map:
                                _ii = tok_map[_k]
                                if seq_args[_ii] is None:  # should not overwrite!
                                    seq_args[_ii] = arg  # put it!
                                    seq_irr[_ii] = ridx  # put it!
                                    seq_vcands[_ii] = 1.  # note: also enforce valid!
                                    c_cc[arg.label] += 1
            if len(c_cc) > 0:
                zlog(f"Add conj for {frame}: {c_cc}")
        # --
        seq_sub_lens = [len(z) for z in seq_ids]
        # --
        # check max_len
        rets = [seq_ids, seq_sub_lens, seq_toks, seq_vcands, seq_ifr, seq_irr, seq_frames, seq_args]
        if sum(seq_sub_lens) > conf.max_content_len:  # truncate things!
            t_start, t_end = self.truncate_subseq(seq_sub_lens, truncate_center, conf.max_content_len, silent=True)
            rets = [z[t_start:t_end] for z in rets]
            # warning if truncate out args
            del_args = [z for z in seq_args[:t_start]+seq_args[t_end:] if z is not None]
            if len(del_args):
                zwarn(f"Loss of args due to truncation: {del_args}")
        # --
        # only add [sep] here!
        _sep = [self.tokenizer.sep_token_id]
        for ii, vv in enumerate([_sep, 1, None, 0., 0, 0, None, None]):
            rets[ii].append(vv)
        rets = rets[:6]  # note: currently only need these
        return tuple(rets)

# --
# b msp2/tasks/zmtl3/mod/extract/evt_arg/arg:
# p self.bmod.tokenizer.convert_ids_to_tokens(_ids_t[0])
