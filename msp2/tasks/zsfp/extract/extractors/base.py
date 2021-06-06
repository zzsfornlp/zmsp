#

# model-based one-seq extractor
# -- sent-level extractor

__all__ = [
    "BaseExtractorConf", "BaseExtractor", "BaseExtractorHelper",
    "ConstrainerNodeConf", "ConstrainerNode", "LookupConf", "LookupNode",
]

from typing import List, Union, Callable, Type
import numpy as np
from collections import defaultdict
from msp2.nn import BK
from msp2.nn.modules import LossHelper
from msp2.nn.layers import BasicConf, BasicNode, node_reg, MLPConf, MLPNode
from msp2.nn.modules import LossHelper, PlainEncoderConf, PlainEncoder
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab, SeqVocab
from msp2.tasks.common.models.seqlab import SeqLabelerConf, SeqLabelerNode, SimpleLabelerNode
from msp2.utils import ZObject, zlog
from msp2.tasks.common.models.pointer import SpanExpanderConf, SpanExpanderNode
from ..constrainer import Constrainer, LexConstrainer

# =====

class BaseExtractorConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # input expr size
        self.psize = -1  # (optional) pairwise size
        # --
        # which frame (note: arg is special!!)
        self.ftag = "evt"  # ef/evt/arg/...
        self.arg_ftag = "ef"  # special target for args: ef/...
        self.core_span_mode = "span"  # which span to pred as the core: span/hspan/shead
        self.arg_only_rank1 = True  # only consider rank1 arg
        # extractor is left to subclass!
        # --
        # final labeler
        self.lab_conf = SeqLabelerConf()
        # self.lab_conf.labeler_conf.fix_non = True  # by default fix NON to 0.
        # constrainer
        self.cons_lex_conf = ConstrainerNodeConf()
        # lookuper
        self.lookup_conf = LookupConf()
        # extender
        self.ext_conf = ExtenderConf()
        # post-processor
        self.pp_conf = PostProcessConf()
        # --
        # train/loss related
        self.loss_lab = 1.  # labeling loss
        self.loss_weight_non = 1.0  # weight for non (idx==0)
        self.loss_ext = 0.  # extending loss
        self.pred_ext = False  # whether do extension

@node_reg(BaseExtractorConf)
class BaseExtractor(BasicNode):
    def __init__(self, conf: BaseExtractorConf, vocab: SimpleVocab, cons_lex: LexConstrainer=None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: BaseExtractorConf = self.conf
        # --
        self.extract_node = self._build_extract_node(conf)  # can be None, means no extraction at this step
        lab_input_size = self.extract_node.get_output_dims()[0] if self.extract_node is not None else conf.isize
        self.lab_node = SeqLabelerNode(vocab, conf.lab_conf, isize=lab_input_size, psize=conf.psize)
        self.lookup_node = LookupNode(conf.lookup_conf, self.lab_node.laber, _isize=lab_input_size)
        # lex-constrainer
        if cons_lex is not None:
            self.cons_lex_node = ConstrainerNode(cons_lex, None, vocab, conf.cons_lex_conf)
        else:
            self.cons_lex_node = None
        # extender
        self.ext_node = ExtenderNode(conf.ext_conf, isize=conf.isize, psize=self.lookup_node.get_output_dims()[0],
                                     _loss_lambda=conf.loss_ext)
        # post-processor
        self.pp_node = PostProcesser(conf.pp_conf, _span_mode=conf.core_span_mode, _ftag=conf.ftag)

    # output dim for lookup
    def get_output_dims(self, *input_dims):
        return self.lookup_node.get_output_dims(*input_dims)

    # --
    # common helper
    def _sum_scores(self, *scores: BK.Expr):  # simply ignore None and add them all
        ret = None
        for score in scores:
            if score is not None:
                if ret is None:
                    ret = score
                else:
                    ret += score
        return ret

    # some common procedures; todo(+W): should make a core-extractor!
    def _finish_loss(self, core_loss, insts, input_expr, mask_expr, pair_expr, lookup_flatten: bool):
        conf: BaseExtractorConf = self.conf
        if conf.loss_ext>0. or lookup_flatten:
            lookup_res = self.lookup_flatten(insts, input_expr, mask_expr, pair_expr)
        else:
            lookup_res = None
        if conf.loss_ext>0.:
            flt_items, flt_sidx, flt_expr, flt_full_expr = lookup_res  # flatten to make dim0 -> frames
            flt_input_expr, flt_mask_expr = input_expr[flt_sidx], mask_expr[flt_sidx]
            ext_loss = self.ext_node.loss(flt_items, flt_input_expr, flt_expr, flt_full_expr, flt_mask_expr)
            ret_loss = LossHelper.combine_multiple_losses([core_loss, ext_loss])  # with another loss
        else:
            ret_loss = core_loss
        return ret_loss, lookup_res

    def _finish_pred(self, insts, input_expr, mask_expr, pair_expr, lookup_flatten: bool):
        conf: BaseExtractorConf = self.conf
        # --
        if conf.pred_ext or lookup_flatten:
            lookup_res = self.lookup_flatten(insts, input_expr, mask_expr, pair_expr)
        else:
            lookup_res = None
        if conf.pred_ext:
            flt_items, flt_sidx, flt_expr, flt_full_expr = lookup_res  # flatten to make dim0 -> frames
            if len(flt_items) > 0:  # no need to predict if no items
                flt_input_expr, flt_mask_expr = input_expr[flt_sidx], mask_expr[flt_sidx]
                self.ext_node.predict(flt_items, flt_input_expr, flt_expr, flt_full_expr, flt_mask_expr)
        # --
        # finally post processing
        self.pp_node.prune(insts)
        # --
        return lookup_res
    # --

    # [*, slen, D], [*, slen], [*, D']
    def loss(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
             pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr=None):
        raise NotImplementedError("Abstract method!!")

    # [*, slen, D], [*, slen], [*, D']
    def predict(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr=None):
        raise NotImplementedError("Abstract method!!")

    # [*, slen, D], [*, slen], [*, D']
    def lookup(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
               pair_expr: BK.Expr = None):
        raise NotImplementedError("Abstract method!!")

    # plus flatten items's dims
    def lookup_flatten(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                       pair_expr: BK.Expr = None):
        raise NotImplementedError("Abstract method!!")

    # =====
    # to be implemented
    def _build_extract_node(self, conf): raise NotImplementedError()

# =====
# lookup node
class LookupConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1
        # --
        self.lookup_mlp = MLPConf().direct_update(use_out=False, n_hid_layer=1)  # input_main(isize) -> ?
        self.use_emb = True  # otherwise directly use expr
        self.train_repl_emb = 0.1  # replace(dropout) to NON in training

@node_reg(LookupConf)
class LookupNode(BasicNode):
    def __init__(self, conf: LookupConf, lab_node: SimpleLabelerNode, **kwargs):
        super().__init__(conf, **kwargs)
        conf: LookupConf = self.conf
        # --
        isizes = [conf._isize, lab_node.lookup_dim] if conf.use_emb else [conf._isize]
        self.mlp = MLPNode(conf.lookup_mlp, isize=isizes, dim_hid=conf._isize, osize=-1, use_out=False)
        self.setattr_borrow("_lab", lab_node)

    def get_output_dims(self, *input_dims):
        return self.mlp.get_output_dims(*input_dims)

    # return the same shape!!
    def lookup(self, span_expr: BK.Expr, lab_expr: BK.Expr, mask_expr: BK.Expr):
        conf: LookupConf = self.conf
        # --
        _inputs = [span_expr]
        if conf.use_emb:
            if self.is_training() and conf.train_repl_emb>0.:
                # todo(note): assume NON==0
                _rand_mask = (BK.random_bernoulli(BK.get_shape(lab_expr), 1.-conf.train_repl_emb, 1.)).long()
                _input_lab = lab_expr * _rand_mask
            else:
                _input_lab = lab_expr
            _input_emb = self._lab.lookup(_input_lab)  # [*, clen, D]
            _inputs.append(_input_emb)
        mlp_expr = self.mlp(_inputs)
        return mlp_expr

    # helper
    @staticmethod
    def flatten_results(arr_items: np.ndarray, mask_expr: BK.Expr, *other_exprs: BK.Expr):
        sel_mask_expr = (mask_expr > 0.)
        # flatten first dims
        ret_items = [z for z in arr_items.flatten() if z is not None]
        ret_other_exprs = [z[sel_mask_expr] for z in other_exprs]  # [?(flat), D]
        ret_sidx = BK.arange_idx(BK.get_shape(mask_expr, 0)).unsqueeze(-1).expand_as(sel_mask_expr)[sel_mask_expr]
        assert all(len(ret_items) == len(z) for z in ret_other_exprs), "Error: dim0 not matched after flatten!"
        return ret_items, ret_sidx, *ret_other_exprs  # [?(flat), *]

# =====
# post-processing, like rule_pruner for evt and non_overlap&budget pruner for args

class PostProcessConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self._span_mode = "span"
        self._ftag = "evt"
        # --
        self.pp_brule_semafor = False  # use semafor rule for evt-trigger
        self.pp_check_more = False  # an overall switch
        self.label_budget = 2  # max budget for each role
        self.non_overlapping = True  # check overlapping

@node_reg(PostProcessConf)
class PostProcesser(BasicNode):
    def __init__(self, conf: PostProcessConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PostProcessConf = self.conf
        # --
        if conf.pp_brule_semafor:
            from ..rule_target import BlacklistRule_semafor
            self.s_brule = BlacklistRule_semafor()
        else:
            self.s_brule = None
        if conf._ftag == "arg":
            self._list_f = lambda x: x.args
            self._del_f = lambda f, x: x.delete_self()
        else:
            self._list_f = lambda x: x.get_frames(conf._ftag)
            self._del_f = lambda s, x: s.delete_frame(x, conf._ftag)
        # --
        self.span_getter = Mention.create_span_getter(conf._span_mode)
        self.full_span_getter = Mention.create_span_getter("span")  # used for check overlapping!
        # --

    def prune(self, insts: List):
        conf: PostProcessConf = self.conf
        span_getter, full_span_getter = self.span_getter, self.full_span_getter
        ftag = conf._ftag
        # --
        if conf.pp_brule_semafor:
            for s in insts:
                frames = s.get_frames(ftag)
                sent_tokens = s.tokens
                for f in frames:
                    span_widx, span_wlen = span_getter(f.mention)
                    if span_wlen != 1:
                        continue  # todo(+N): currently only check wlen==1
                    key_token = sent_tokens[span_widx]
                    if self.s_brule.hit(key_token, [key_token], sent_tokens):
                        s.delete_frame(f, ftag)
        # --
        if conf.pp_check_more:
            label_budget, non_overlapping = conf.label_budget, conf.non_overlapping
            for s in insts:
                items = sorted(self._list_f(s), key=lambda x: x.score, reverse=True)  # sort by score
                cur_budgets = {}
                cur_hits = [False] * len(s if isinstance(s, Sent) else s.sent)
                for one in items:
                    to_del = False
                    # --
                    one_label = one.label
                    one_label_count = cur_budgets.get(one_label, 0)
                    one_widx, one_wlen = self.full_span_getter(one.mention)
                    if one_label_count >= label_budget:  # first check budget
                        to_del = True
                    elif non_overlapping:  # then check overlap
                        if any(cur_hits[i] for i in range(one_widx, one_widx+one_wlen)):
                            to_del = True
                    # --
                    if to_del:
                        self._del_f(s, one)
                    else:
                        cur_budgets[one_label] = one_label_count + 1
                        if non_overlapping:
                            cur_hits[one_widx:one_widx+one_wlen] = [True]*one_wlen
        # --

# =====
# helper for interacting with instances

class BaseExtractorHelper:
    def __init__(self, conf: BaseExtractorConf, vocab: SimpleVocab):
        self.conf = conf  # borrow conf
        self.core_span_getter = Mention.create_span_getter(conf.core_span_mode)
        self.core_span_setter = Mention.create_span_setter(conf.core_span_mode)
        self.vocab = vocab
        # --
        if conf.ftag == "arg":
            self._get_f = self._get_args
            self._clear_f = lambda inst: inst.clear_args()  # first delete all args if existing
            self._new_f = self._new_arg
        else:
            self._get_f = self._get_frames
            self._clear_f = lambda inst: inst.delete_frames(conf.ftag)  # first delete all frames if existing
            self._new_f = self._new_frame
        # --

    def _get_frames(self, s: Sent):
        return s.get_frames(self.conf.ftag)

    def _get_args(self, f: Frame):
        cur_sent = f.sent
        args = [a for a in f.args if (a.arg.sent is cur_sent)]  # get args at the same sent!
        if self.conf.arg_only_rank1:
            args = [a for a in args if a.info.get("rank", 1) == 1]  # only consider rank1 ones
        return args

    def _new_frame(self, s: Sent, one_widx: int, one_wlen: int, one_lab: int, one_score: float, vocab=None):
        if vocab is None:
            vocab = self.vocab
        # --
        f_type = vocab.idx2word(one_lab)
        f = s.make_frame(one_widx, one_wlen, self.conf.ftag, type=f_type, score=one_score)
        f.set_label_idx(one_lab)
        self.core_span_setter(f.mention, one_widx, one_wlen)  # core_span
        return f

    def _new_arg(self, f: Frame, one_widx: int, one_wlen: int, one_lab: int, one_score: float, vocab=None):
        if vocab is None:
            vocab = self.vocab
        # --
        ef = f.sent.make_frame(one_widx, one_wlen, self.conf.arg_ftag)  # make an ef as mention
        a_role = vocab.idx2word(one_lab)
        alink = f.add_arg(ef, a_role, score=float(one_score))
        alink.set_label_idx(one_lab)  # set idx
        self.core_span_setter(alink.mention, one_widx, one_wlen)
        return alink

    # return ndarray[bsize, slen]
    def get_batched_items(self, insts: List):
        all_items = [self._get_f(z) for z in insts]
        arr_shape = len(insts), max(len(z) for z in all_items)
        arr_items = np.full(arr_shape, None, dtype=object)
        arr_masks = np.full(arr_shape, 0., dtype=np.float32)
        for zidx, zitems in enumerate(all_items):
            zlen = len(zitems)
            arr_items[zidx, :zlen] = zitems
            arr_masks[zidx, :zlen] = 1.
        return arr_items, BK.input_real(arr_masks)

    # input is ndarray[object], None means nope
    @staticmethod
    def get_batched_features(items: np.ndarray, df_val: Union[int, float], attr_f: Union[str, Callable], dtype=None):
        if isinstance(attr_f, str):
            _local_attr_str = str(attr_f)
            attr_f = lambda x: getattr(x, _local_attr_str)
        # --
        flattened_vals = BK.input_tensor([df_val if z is None else attr_f(z) for z in items.flatten()], dtype=dtype)
        ret = flattened_vals.view(BK.get_shape(items))
        return ret

# =====
# Constrainer Helper

class ConstrainerNodeConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --

@node_reg(ConstrainerNodeConf)
class ConstrainerNode(BasicNode):
    def __init__(self, cons: Constrainer, src_vocab: SimpleVocab, trg_vocab: SimpleVocab, conf: ConstrainerNodeConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ConstrainerNodeConf = self.conf
        # --
        # input vocab
        if src_vocab is None:  # make our own src_vocab
            cons_keys = sorted(cons.cmap.keys())  # simply get all the keys
            src_vocab = SimpleVocab.build_by_static(cons_keys, pre_list=["non"], post_list=None)  # non==0!
        # output vocab
        assert trg_vocab is not None
        out_size = len(trg_vocab)  # output size is len(trg_vocab)
        trg_is_seq_vocab = isinstance(trg_vocab, SeqVocab)
        _trg_get_f = (lambda x: trg_vocab.get_range_by_basename(x)) if trg_is_seq_vocab else (lambda x: trg_vocab.get(x))
        # set it up
        _vec = np.full((len(src_vocab), out_size), 0., dtype=np.float32)
        assert src_vocab.non == 0
        _vec[0] = 1.  # by default: src-non is all valid!
        _vec[:,0] = 1.  # by default: trg-non is all valid!
        # --
        stat = {"k_skip": 0, "k_hit": 0, "v_skip": 0, "v_hit": 1}
        for k, v in cons.cmap.items():
            idx_k = src_vocab.get(k)
            if idx_k is None:
                stat["k_skip"] += 1
                continue  # skip no_hit!
            stat["k_hit"] += 1
            for k2 in v.keys():
                idx_k2 = _trg_get_f(k2)
                if idx_k2 is None:
                    stat["v_skip"] += 1
                    continue
                stat["v_hit"] += 1
                if trg_is_seq_vocab:
                    _vec[idx_k, idx_k2[0]:idx_k2[1]] = 1.  # hit range
                else:
                    _vec[idx_k, idx_k2] = 1.  # hit!!
        zlog(f"Setup ConstrainerNode ok: vec={_vec.shape}, stat={stat}")
        # --
        self.cons = cons
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.vec = BK.input_real(_vec)

    def lookup(self, idxes: BK.Expr):
        return self.vec[idxes]  # simply lookup

    def lookup_with_feats(self, arr_feats):
        voc = self.src_vocab
        idxes = BK.input_idx([voc.get(s, 0) for s in arr_feats.flatten()]).view(BK.get_shape(arr_feats))
        return self.lookup(idxes)

# =====
# expander/extender

class ExtenderConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # main input size
        self.psize = -1  # other psize, used for pairwise scoring mode
        self._loss_lambda = 1.0
        # --
        self.ext_span_mode = "span"  # which span to extend from core?
        # special encoder before expand
        self.eenc_mix_center = True  # mix center info for the inputs of eenc
        self.eenc_conf = PlainEncoderConf()
        # --
        self.ext_use_finput = False  # pairwise score or not?
        self.econf = SpanExpanderConf()

@node_reg(ExtenderConf)
class ExtenderNode(BasicNode):
    def __init__(self, conf: ExtenderConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ExtenderConf = self.conf
        # --
        self.ext_span_getter = Mention.create_span_getter(conf.ext_span_mode)
        self.ext_span_setter = Mention.create_span_setter(conf.ext_span_mode)
        self.eenc = PlainEncoder(conf.eenc_conf, input_dim=conf.isize)
        self.enode = SpanExpanderNode(conf.econf, isize=conf.isize, psize=(conf.psize if conf.ext_use_finput else -1))

    # [**, slen, D], [**, slen, D']
    def _forward_eenc(self, flt_input_expr: BK.Expr, flt_full_expr: BK.Expr, flt_mask_expr: BK.Expr):
        if self.conf.eenc_mix_center:
            mixed_input_t = flt_input_expr + flt_full_expr  # simply adding
        else:
            mixed_input_t = flt_input_expr
        eenc_output = self.eenc.forward(mixed_input_t, mask_expr=flt_mask_expr)
        return eenc_output

    # --
    # assume already flattened inputs

    # [*], [*, slen, D], [*, D'], [*, slen]; [*]
    def loss(self, flt_items, flt_input_expr, flt_pair_expr, flt_full_expr, flt_mask_expr, flt_extra_weights=None):
        conf: ExtenderConf = self.conf
        _loss_lambda = conf._loss_lambda
        # --
        enc_t = self._forward_eenc(flt_input_expr, flt_full_expr, flt_mask_expr)  # [*, slen, D]
        s_left, s_right = self.enode.score(enc_t, flt_pair_expr if conf.ext_use_finput else None, flt_mask_expr)  # [*, slen]
        # --
        gold_posi = [self.ext_span_getter(z.mention) for z in flt_items]  # List[(widx, wlen)]
        widx_t = BK.input_idx([z[0] for z in gold_posi])  # [*]
        wlen_t = BK.input_idx([z[1] for z in gold_posi])
        loss_left_t, loss_right_t = BK.loss_nll(s_left, widx_t), BK.loss_nll(s_right, widx_t+wlen_t-1)  # [*]
        if flt_extra_weights is not None:
            loss_left_t *= flt_extra_weights
            loss_right_t *= flt_extra_weights
            loss_div = flt_extra_weights.sum()  # note: also use this!
        else:
            loss_div = BK.constants([len(flt_items)], value=1.).sum()
        loss_left_item = LossHelper.compile_leaf_loss("left", loss_left_t.sum(), loss_div, loss_lambda=_loss_lambda)
        loss_right_item = LossHelper.compile_leaf_loss("right", loss_right_t.sum(), loss_div, loss_lambda=_loss_lambda)
        ret_loss = LossHelper.combine_multiple_losses([loss_left_item, loss_right_item])
        return ret_loss

    # [*], [*, D], [*, D], [*]
    def predict(self, flt_items, flt_input_expr, flt_pair_expr, flt_full_expr, flt_mask_expr):
        conf: ExtenderConf = self.conf
        if len(flt_items) <= 0:
            return None  # no input item!
        # --
        enc_t = self._forward_eenc(flt_input_expr, flt_full_expr, flt_mask_expr)  # [*, D]
        s_left, s_right = self.enode.score(enc_t, flt_pair_expr if conf.ext_use_finput else None, flt_mask_expr)
        # --
        max_scores, left_idxes, right_idxes = SpanExpanderNode.decode_with_scores(s_left, s_right, normalize=True)
        all_arrs = [BK.get_value(z) for z in [left_idxes, right_idxes]]
        for cur_item, cur_left_idx, cur_right_idx in zip(flt_items, *all_arrs):
            new_widx, new_wlen = int(cur_left_idx), int(cur_right_idx+1-cur_left_idx)
            self.ext_span_setter(cur_item.mention, new_widx, new_wlen)
        # --
