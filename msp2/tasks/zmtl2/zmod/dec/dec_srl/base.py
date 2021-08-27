#

# the basic srl decoder
# -- simply single- and pairwise- classification!

__all__ = [
    "ZTaskSrlConf", "ZTaskSrl", "ZDecoderSrlConf", "ZDecoderSrl",
]

from typing import List, Dict
import numpy as np
from collections import Counter
from msp2.data.inst import yield_sents, yield_sent_pairs, set_ee_heads, Frame
from msp2.data.vocab import SimpleVocab, SeqVocab
from msp2.utils import AccEvalEntry, zlog, zwarn, Constants, ConfEntryChoices, Random
from msp2.proc import ResultRecord, FrameEvalConf, FrameEvaler, MyFNEvalConf, MyFNEvaler, MyPBEvalConf, MyPBEvaler
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from ..base import *
from ..base_idec import *
from ..base_idec2 import *
from ...common import ZMediator, ZLabelConf, ZlabelNode, ZBoundaryPointerConf, ZBoundaryPointer
from ...enc import ZEncoder
# --
# note: load which one??
# from .inference import SrlInferenceHelperConf, SrlInferenceHelper
from .inference2 import SrlInferenceHelperConf, SrlInferenceHelper
# import os  # simpler to debug!
# if os.environ.get("USE_INF2", False):
#     from .inference2 import SrlInferenceHelperConf, SrlInferenceHelper
# --

# --

class ZTaskSrlConf(ZTaskDecConf):
    def __init__(self):
        super().__init__()
        self.name = "srl"  # a default name
        # --
        # vocab (extra ones)
        self.srl_evoc_file = ""  # load from file
        self.srl_evoc_name = ""  # find it in prebuilt resources
        # eval
        self.srl_eval: FrameEvaler = \
            ConfEntryChoices({'base': FrameEvalConf(), 'pb': MyPBEvalConf(), 'fn': MyFNEvalConf()}, 'base')
        self.srl_pred_clear_evt = True  # note: should turn this off if using special modes for inference
        self.srl_pred_clear_ef = True
        self.srl_pred_clear_arg = True
        self.srl_delete_noarg_evts = False  # delete evts that have no args (other than V & C-V)
        # model
        self.srl_conf = ZDecoderSrlConf()
        # --

    @classmethod
    def _get_type_hints(cls):
        return {'srl_evoc_file': 'zglob1'}

    @staticmethod
    def make_conf(name: str):
        ret = ZTaskSrlConf()
        ret.name = name  # assign name
        if name.startswith("pb"):
            ret.srl_eval = MyPBEvalConf()
        elif name.startswith("fn"):
            ret.srl_eval = MyFNEvalConf()
            ret.srl_conf.lab_arg.direct_update(emb_size=200, input_act='elu', loss_do_sel=True)
        elif name.startswith("ee"):  # event extraction
            ret.srl_eval = FrameEvalConf().direct_update(match_arg_with_frame=False)
        else:
            raise NotImplementedError()
        return ret

    def build_task(self):
        return ZTaskSrl(self)

class ZTaskSrl(ZTaskDec):
    def __init__(self, conf: ZTaskSrlConf):
        super().__init__(conf)
        conf: ZTaskSrlConf = self.conf
        # --
        eval_conf = conf.srl_eval
        if isinstance(eval_conf, MyFNEvalConf):  # todo(+N): ugly!
            evaler = MyFNEvaler(eval_conf)
        elif isinstance(eval_conf, MyPBEvalConf):
            evaler = MyPBEvaler(eval_conf)
        else:
            evaler = FrameEvaler(eval_conf)
        # --
        self.evaler = evaler
        # --

    # build vocab
    def build_vocab(self, datasets: List):
        conf: ZTaskSrlConf = self.conf
        # --
        voc_ef, voc_evt, voc_arg = [SimpleVocab.build_empty(f"{self.name}_{z}") for z in ["ef", "evt", "arg"]]
        for dataset in datasets:
            for sent in yield_sents(dataset.insts):
                for ef in sent.entity_fillers:
                    voc_ef.feed_one(ef.label)
                for evt in sent.events:
                    voc_evt.feed_one(evt.label)
                    for arg in evt.args:
                        voc_arg.feed_one(arg.label)
        # --
        if conf.srl_evoc_name:
            from msp2.data.resources import get_frames_label_budgets
            _flb = get_frames_label_budgets(conf.srl_evoc_name)
            for k, v in _flb.items():
                voc_evt.feed_one(k)
                for v2 in v.keys():
                    voc_arg.feed_one(v2)
        if conf.srl_evoc_file:
            from msp2.utils import default_pickle_serializer
            _fc = default_pickle_serializer.from_file(conf.srl_evoc_file)
            for f in _fc.frames:
                voc_evt.feed_one(f.name)
                for a in f.roles:
                    voc_arg.feed_one(a.name)
        # --
        for vv in [voc_ef, voc_evt, voc_arg]:
            vv.build_sort()
            zlog(f"Finish building for: {vv}")
        return (voc_ef, voc_evt, voc_arg)

    # prepare one instance
    def prep_inst(self, inst, dataset):
        wset = dataset.wset
        # note: special one!
        if self.conf.srl_delete_noarg_evts:
            for sent in yield_sents(inst):
                for evt in list(sent.events):  # note: remember to copy!!
                    if not getattr(evt, "_noarg_visited", False):
                        evt._noarg_visited = True  # already visited!
                        if len([a for a in evt.args if a.role not in ["V", "C-V"]]) == 0:
                            sent.delete_frame(evt, 'evt')
        # if wset == "train":
        if True:  # note: make it convenient to reuse "_prep_item"
            voc_ef, voc_evt, voc_arg = self.vpack
            for sent in yield_sents(inst):  # assign idx!!
                for ef in sent.entity_fillers:
                    ef.set_label_idx(voc_ef.get_else_unk(ef.label))
                for evt in sent.events:
                    evt.set_label_idx(voc_evt.get_else_unk(evt.label))
                    for arg in evt.args:
                        arg.set_label_idx(voc_arg.get_else_unk(arg.label))
        if wset != "train":  # clear for testing!
            if self.conf.srl_pred_clear_evt:
                for sent in yield_sents(inst):
                    sent.clear_events()  # clear evts
            elif self.conf.srl_pred_clear_arg:
                for sent in yield_sents(inst):
                    for evt in sent.events:
                        evt.clear_args()  # clear args!
            if self.conf.srl_pred_clear_ef:
                for sent in yield_sents(inst):
                    sent.clear_entity_fillers()  # clear efs
        # --

    # prepare one input_item
    def prep_item(self, item, dataset):
        pass  # leave to the mod to handle!!

    # eval
    def eval_insts(self, gold_insts: List, pred_insts: List, quite=False):
        set_ee_heads(gold_insts)  # in case we want to eval heads!
        return ZTaskDec.do_eval(self.name, self.evaler, gold_insts, pred_insts, quite)

    # build mod
    def build_mod(self, model):
        return ZDecoderSrl(self.conf.srl_conf, self, model.encoder)

# --

class ZDecoderSrlConf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        # ef
        self.idec_ef = IdecConf.make_conf('score')
        self.idec_ef.node.core.hid_dim = 300  # add a hidden layer
        self.loss_ef = 0.
        self.lab_ef = ZLabelConf().direct_update(fixed_nil_val=0.)
        self.max_layer_ef = 1  # max overlapping layer for training and testing!
        self.loss_ef_boundary = 0.
        self.ef_boundary = ConfEntryChoices({'yes': ZBoundaryPointerConf(), 'no': None}, 'no')
        # evt
        self.idec_evt = IdecConf.make_conf('score')
        self.idec_evt.node.core.hid_dim = 300  # add a hidden layer
        self.loss_evt = 0.5
        self.binary_evt = False  # only 0/1?
        self.lab_evt = ZLabelConf().direct_update(
            fixed_nil_val=0., loss_neg_sample=1., loss_binary_alpha=0., loss_full_alpha=1.)
        self.max_layer_evt = 1
        self.loss_evt_boundary = 0.
        self.evt_boundary = ConfEntryChoices({'yes': ZBoundaryPointerConf(), 'no': None}, 'no')
        self.layer_add_evt_ind = -100  # add input evt indicators at which layer?
        self.init_scale_evt_ind = 1.  # evt indicator input (no need to be large since bert's is not quite large!)
        # note: special mode: one sequence one frame (osof)!
        #  what to change: data.convert_strategy=frame, lab_evt.neg=0./loss_binary=0., (add_evt_ind>=0??)
        self.assume_osof = False
        # arg
        self.idec_arg = ConfEntryChoices({'idec1': IdecConf.make_conf('pairwise'), 'idec2': Idec2Conf()}, 'idec1')
        self.loss_arg = 1.
        self.lab_arg_negevt = 0.  # neg evt for args
        self.lab_arg = ZLabelConf().direct_update(fixed_nil_val=0., loss_neg_sample=1.)
        self.arg_use_bio = False  # use BIO mode for arg!
        self.max_layer_arg = 1
        self.loss_arg_boundary = 0.
        self.arg_boundary = ConfEntryChoices({'yes': ZBoundaryPointerConf(), 'no': None}, 'no')
        self.arg_allowed_sent_gap = 0  # by default only allow same sent
        # --
        # predicting/inference
        self.inferencer = ConfEntryChoices({'simple': SrlInferenceHelperConf()}, 'simple')

@node_reg(ZDecoderSrlConf)
class ZDecoderSrl(ZDecoder):
    def __init__(self, conf: ZDecoderSrlConf, ztask, main_enc: ZEncoder, **kwargs):
        super().__init__(conf, ztask, main_enc, **kwargs)
        conf: ZDecoderSrlConf = self.conf
        self.voc_ef, self.voc_evt, self.voc_arg = self.ztask.vpack
        if conf.arg_use_bio:
            self.vocab_bio_arg = SeqVocab(self.voc_arg)  # simply BIO vocab
            zlog(f"Use BIO vocab for ARG: {self.vocab_bio_arg}")
            final_voc_arg = self.vocab_bio_arg
        else:  # otherwise original voc!
            self.vocab_bio_arg = None  # no BIO
            final_voc_arg = self.voc_arg
        # --
        _enc_dim, _head_dim = main_enc.get_enc_dim(), main_enc.get_head_dim()
        # ef
        self.lab_ef = ZlabelNode(conf.lab_ef, _csize=len(self.voc_ef))
        self.idec_ef = conf.idec_ef.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=self.lab_ef.get_core_csize())
        self.reg_idec('ef', self.idec_ef)
        self.boundary_ef = None if conf.ef_boundary is None else ZBoundaryPointer(conf.ef_boundary, _isize=_enc_dim)
        # evt: note: 2-way classification if binary
        self.lab_evt = ZlabelNode(conf.lab_evt, _csize=(2 if conf.binary_evt else len(self.voc_evt)))
        self.idec_evt = conf.idec_evt.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=self.lab_evt.get_core_csize())
        self.reg_idec('evt', self.idec_evt)
        if conf.layer_add_evt_ind >= 0:
            self.evt_indicator_embed = EmbeddingNode(None, osize=_enc_dim, n_words=2, init_scale=conf.init_scale_evt_ind)
        else:  # not used!!
            self.evt_indicator_embed = None
        self.boundary_evt = None if conf.evt_boundary is None else ZBoundaryPointer(conf.evt_boundary, _isize=_enc_dim)
        # arg
        self.lab_arg = ZlabelNode(conf.lab_arg, _csize=len(final_voc_arg))
        self.idec_arg = conf.idec_arg.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=self.lab_arg.get_core_csize())
        self.reg_idec('arg', self.idec_arg)
        self.boundary_arg = None if conf.arg_boundary is None else ZBoundaryPointer(conf.arg_boundary, _isize=_enc_dim)
        # --
        # inference
        self.inferencer = conf.inferencer.make_node(dec=self)
        # --

    # special further trigger indicators!
    def layer_end(self, med: ZMediator):
        rets, satisfied = super().layer_end(med)
        if self.conf.layer_add_evt_ind == med.lidx:  # add at where?
            evt_posi = self.prepare_evt_posi(med.ibatch)  # [*, dlen]
            dsel_seq_info = med.ibatch.seq_info
            _arange_t, _back_sel_idxes = dsel_seq_info.arange2_t, dsel_seq_info.enc_back_sel_idxes  # [*, ??]
            # note: still only mark first subword!!
            evt_posi_full = evt_posi[_arange_t, _back_sel_idxes] * dsel_seq_info.enc_back_hits.long()  # [*, elen]
            add_t = self.evt_indicator_embed(evt_posi_full)  # [*, elen, D]
            rets.append(add_t)  # simply append one!
        return rets, satisfied

    def loss(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderSrlConf = self.conf
        # --
        # prepare info
        ibatch = med.ibatch
        # [*, dlen] / [*, dlen, dlen]
        expr_ef, expr_ef_bounds, expr_evt, expr_evt_bounds, expr_arg, expr_arg_bounds = self.prepare(ibatch)
        base_mask_t = self.get_dec_mask(ibatch, conf.msent_loss_center)  # [bs, dlen]
        # --
        # get losses
        loss_items = []
        hid_t = med.get_enc_cache("hid").get_cached_value()  # last enc layer, [*, dlen, D]
        _ds_idxes = ibatch.seq_info.dec_sent_idxes  # [bs, dlen]
        # read extra muls from dataset
        _loss_mul_ef, _loss_mul_evt, _loss_mul_arg = [ibatch.dataset.info.get("_loss_mul_"+z, 1.) for z in ['ef', 'evt', 'arg']]
        # --
        # ef
        _loss_ef, _loss_ef_boundary = _loss_mul_ef*conf.loss_ef, _loss_mul_ef*conf.loss_ef_boundary
        if _loss_ef > 0.:
            loss_items.extend(self.loss_from_lab(self.lab_ef, 'ef', med, expr_ef, base_mask_t, _loss_ef))
        if _loss_ef_boundary > 0.:
            loss_items.append(
                self._loss_boundary(self.boundary_ef, 'ef', _loss_ef_boundary, hid_t, expr_ef, expr_ef_bounds, _ds_idxes))
        # evt
        _loss_evt, _loss_evt_boundary = _loss_mul_evt*conf.loss_evt, _loss_mul_evt*conf.loss_evt_boundary
        if _loss_evt > 0.:
            # labeling
            _evt_loss_neg_sample = ibatch.dataset.info.get("_evt_loss_neg_sample")  # note: special name!!
            _feed_expr_evt = (expr_evt>0).long() if conf.binary_evt else expr_evt  # labeled or not?
            loss_items.extend(self.loss_from_lab(self.lab_evt, 'evt', med, _feed_expr_evt, base_mask_t, _loss_evt,
                                                 loss_neg_sample=_evt_loss_neg_sample))
        if _loss_evt_boundary > 0.:
            loss_items.append(
                self._loss_boundary(self.boundary_evt, 'evt', _loss_evt_boundary, hid_t, expr_evt, expr_evt_bounds, _ds_idxes))
        # arg
        _loss_arg, _loss_arg_boundary = _loss_mul_arg*conf.loss_arg, _loss_mul_arg*conf.loss_arg_boundary
        if _loss_arg > 0.:
            # labeling
            # first prepare (another set of) trigger mask: [*, dlen]
            _evt_pos_t = (expr_evt>0).float()
            _evt_mask = self.lab_evt._get_loss_mask(_evt_pos_t, base_mask_t, loss_neg_sample=conf.lab_arg_negevt)
            # extra mask: [bs, dlen, dlen]
            _mask_t = ((_ds_idxes.unsqueeze(-1)-_ds_idxes.unsqueeze(-2)).abs()<=conf.arg_allowed_sent_gap).float()
            _mask_t *= _evt_mask.unsqueeze(-1)  # only allow selected evts
            _mask_t *= ibatch.seq_info.dec_sel_masks.unsqueeze(-2)  # exclude paddings
            loss_items.extend(self.loss_from_lab(self.lab_arg, 'arg', med, expr_arg, _mask_t, _loss_arg))
        if _loss_arg_boundary > 0.:
            loss_items.append(
                self._loss_boundary(self.boundary_arg, 'arg', _loss_arg_boundary, hid_t, expr_arg, expr_arg_bounds, _ds_idxes))
        # --
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss, {}

    # get boundary loss: ..., [*, dlen, D], [*, *dlen], [*, *dlen, 2], [*, dlen]
    def _loss_boundary(self, node_boundary: ZBoundaryPointer, subname: str, loss_lambda: float,
                       hid_t: BK.Expr, expr_labs: BK.Expr, expr_boundaries: BK.Expr, dec_sent_idxes: BK.Expr):
        valid_mask_t = (expr_labs > 0)  # [*, *dlen]
        valid_idxes = valid_mask_t.nonzero(as_tuple=True)  # *[*]
        # rearrange/flat things (mask should constrain same sent)!
        batch_idxes = valid_idxes[0]  # [*]
        flat_hid_t = hid_t[batch_idxes]  # [??, dlen, D]
        flat_boundaries_t = expr_boundaries[valid_idxes]  # [??, 2]
        flat_mask_t = (dec_sent_idxes[batch_idxes] == dec_sent_idxes[batch_idxes,valid_idxes[-1]].unsqueeze(-1)).float()
        # get loss
        flat_indicators = node_boundary.prepare_indicators(valid_idxes[1:], BK.get_shape(flat_hid_t)[:2])  # *[??, dlen]
        loss_t = node_boundary.gather_losses(flat_hid_t, flat_mask_t, flat_indicators, flat_boundaries_t)  # [??]
        # return
        loss_div = BK.constants([len(loss_t)], value=1.).sum()
        ret = LossHelper.compile_leaf_loss(f"{subname}_bp", loss_t.sum(), loss_div, loss_lambda=loss_lambda)
        return ret

    def predict(self, med: ZMediator, *args, **kwargs):
        self.inferencer.predict(med)  # simple let the inference helper handle this!
        return {}

    # --
    # helpers

    # prepare for one seq
    def _prep_seq(self, ones, length: int, _dec_offsets: Dict, _max_layer: int, _bio_voc: SeqVocab, quite=False):
        # todo(note): simply sort everything by label-idx (which should be sorted by frequency)
        sorted_ones = sorted(ones, key=(lambda x: x.label_idx))
        _span_f = (lambda x: x.mention.get_span()) if _bio_voc is not None else (lambda x: x.mention.get_span(shead=True))
        # --
        # split layers
        layered_hits = [[0]*length for _ in range(_max_layer)]  # check overlap
        layered_objects = [[] for _ in range(_max_layer)]  # put objects
        extra_objects = []
        repeat_sigs = set()  # check repeat
        for one in sorted_ones:
            _offset = _dec_offsets.get(id(one.mention.sent))
            if _offset is None:
                continue  # out of range!!
            _cur_layer = 0
            _widx, _wlen = _span_f(one)
            _widx += _offset
            _label_idx = one.label_idx
            _sig = (_widx, _wlen, _label_idx)
            if _sig in repeat_sigs:
                continue  # repeated!
            while _cur_layer < _max_layer:
                _cur_layered_hits = layered_hits[_cur_layer]
                if all(_cur_layered_hits[i]==0 for i in range(_widx, _widx+_wlen)):  # hit!
                    _cur_layered_hits[_widx:_widx+_wlen] = [1]*_wlen
                    layered_objects[_cur_layer].append(one)
                    repeat_sigs.add(_sig)
                    break
                _cur_layer += 1
            if _cur_layer >= _max_layer:
                extra_objects.append(one)
        # --
        # warnings
        if len(extra_objects)>0 and not quite:
            zwarn(f"Extra ones: ``{extra_objects}'' over base ones: {layered_objects}")
        # --
        # assign them!
        layered_objects = [z for z in layered_objects if len(z)>0]  # delete empty ones
        if len(layered_objects) == 0:
            layered_objects.append([])  # check empty
        _num_layer = len(layered_objects)
        ret_labels = [np.zeros(length, dtype=np.int) for _ in range(_num_layer)]
        if _bio_voc is None:  # assign heads and boundaries
            ret_objs = [np.full(length, None, dtype=object) for _ in range(_num_layer)]
            ret_bounds = np.full([length,2], -1, dtype=np.int)
            for _cur_layer in reversed(range(_num_layer)):  # reversed!
                for one in layered_objects[_cur_layer]:
                    _offset = _dec_offsets[id(one.mention.sent)]
                    _hidx = one.mention.shead_widx + _offset
                    _widx, _wlen = one.mention.get_span()
                    _widx += _offset
                    ret_bounds[_hidx] = [_widx, _widx+_wlen-1]
                    for _layer2 in range(_cur_layer, _num_layer):  # try to fill all later ones!
                        if ret_labels[_layer2][_hidx] == 0:  # if not covered
                            ret_labels[_layer2][_hidx] = one.label_idx
                            ret_objs[_layer2][_hidx] = one
        else:  # only assign bio-tags
            ret_objs = ret_bounds = None
            for _cur_layer in reversed(range(_num_layer)):  # reversed!
                for one in layered_objects[_cur_layer]:
                    _offset = _dec_offsets[id(one.mention.sent)]
                    _widx, _wlen = one.mention.get_span()
                    _widx += _offset
                    for _layer2 in range(_cur_layer, _num_layer):  # try to fill all later ones!
                        if all(z==0 for z in ret_labels[_layer2][_widx:_widx+_wlen]):  # ok to fill in
                            ret_labels[_layer2][_widx:_widx+_wlen] = _bio_voc.output_span_idx(_wlen, one.label_idx)
        return (ret_labels, ret_objs, ret_bounds)

    # prepare for one item (cached)
    def _prep_item(self, item):
        _cache_name = f"_{self.name}_cache"
        _srl_cache = item.info.get(_cache_name)
        if _srl_cache is None:
            conf: ZDecoderSrlConf = self.conf
            # todo(+N): here we prepare all instances, regardless of sidx
            # prepare new one!!
            _dec_offsets = {id(ss):x for ss,x in zip(item.sents, item.seq_info.dec_offsets)}  # use id(sent)
            _len = len(item.seq_info.dec_sel_idxes)
            # set head
            set_ee_heads(item.sents)
            # --
            # tell the mode!!
            if conf.assume_osof:  # note: take individual item.frame
                # depending on the given frame position
                _ge = item.inst  # given evt
                _cache_ef = self._prep_seq([a.arg for a in _ge.args], _len, _dec_offsets, conf.max_layer_ef, None, quite=True)
                _cache_evt = self._prep_seq([_ge], _len, _dec_offsets, conf.max_layer_evt, None)  # only one!
                assert len([z2 for z in _cache_evt[1] for z2 in z if z2 is not None]) == 1  # one inst!
            else:
                _cache_ef = self._prep_seq(
                    sum([s.entity_fillers for s in item.sents], []), _len, _dec_offsets, conf.max_layer_ef, None, quite=True)
                _cache_evt = self._prep_seq(sum([s.events for s in item.sents], []), _len, _dec_offsets, conf.max_layer_evt, None)
            # --
            # aggregate all args!!
            _cache_arg = {}  # evt-hidx -> _cache
            for evt_hidx in range(_len):
                objs = []
                for _obj in [z[evt_hidx] for z in _cache_evt[1]]:
                    if _obj is not None and _obj not in objs:
                        objs.append(_obj)
                if len(objs) > 0:
                    _args = [a for a in sum([e.args for e in objs], []) if (a.info.get("rank",1)==1)]  # note: only rank1!!
                    _cache_arg[evt_hidx] = self._prep_seq(_args, _len, _dec_offsets, conf.max_layer_arg, self.vocab_bio_arg)
            # --
            _srl_cache = (_cache_ef, _cache_evt, _cache_arg)
            item.info[_cache_name] = _srl_cache
        return _srl_cache

    # prepare gold labels
    def prepare(self, ibatch):
        b_seq_info = ibatch.seq_info
        bsize, dlen = BK.get_shape(b_seq_info.dec_sel_masks)
        arr_ef = np.full([bsize, dlen], 0, dtype=np.int)  # by default 0
        arr_ef_bounds = np.full([bsize, dlen, 2], -1, dtype=np.int)  # by default -1
        arr_evt = np.full([bsize, dlen], 0, dtype=np.int)  # by default 0
        arr_evt_bounds = np.full([bsize, dlen, 2], -1, dtype=np.int)  # by default -1
        arr_arg = np.full([bsize, dlen, dlen], 0, dtype=np.int)  # by default 0
        arr_arg_bounds = np.full([bsize, dlen, dlen, 2], -1, dtype=np.int)  # by default -1
        # --
        _gen = Random.get_generator("srl")
        _choice_f = (lambda x: 0 if len(x)<=1 else _gen.choice(len(x)))
        # --
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _len = len(item.seq_info.dec_sel_idxes)
            _cache_ef, _cache_evt, _cache_arg = self._prep_item(item)
            # one for all
            for _one_cache, _one_arr_label, _one_arr_bounds in zip(
                    [_cache_ef, _cache_evt, _cache_arg], [arr_ef, arr_evt, arr_arg], [arr_ef_bounds, arr_evt_bounds, arr_arg_bounds]):
                if isinstance(_one_cache, dict):  # 3d; arg
                    for hidx, _a_cache in _one_cache.items():
                        _cache_labels, _, _cache_bounds = _a_cache
                        _cache_sel = _choice_f(_cache_labels)
                        _one_arr_label[bidx, hidx, :_len] = _cache_labels[_cache_sel]
                        if _cache_bounds is not None:
                            _one_arr_bounds[bidx, hidx, :_len] = _cache_bounds
                else:  # 2d; ef/evt
                    _cache_labels, _, _cache_bounds = _one_cache
                    _cache_sel = _choice_f(_cache_labels)
                    _one_arr_label[bidx, :_len] = _cache_labels[_cache_sel]
                    if _cache_bounds is not None:
                        _one_arr_bounds[bidx, :_len] = _cache_bounds
            # --
        # return
        rets = [BK.input_idx(z) for z in [arr_ef, arr_ef_bounds, arr_evt, arr_evt_bounds, arr_arg, arr_arg_bounds]]
        return rets

    # prepare evt posi inputs (optional feed as input if assuming predicate position)
    def prepare_evt_posi(self, ibatch):
        b_seq_info = ibatch.seq_info
        bsize, dlen = BK.get_shape(b_seq_info.dec_sel_masks)
        arr = np.full([bsize, dlen], 0, dtype=np.int)  # by default 0
        # --
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _len = len(item.seq_info.dec_sel_idxes)
            _cache_ef, _cache_evt, _cache_arg = self._prep_item(item)
            for _arr_evts in _cache_evt[0]:
                arr[bidx, :_len] += _arr_evts  # 0 will always be 0!
        ret = (BK.input_idx(arr)>0).long()  # hit ones will be >0
        return ret

# --
# b msp2/tasks/zmtl2/zmod/dec/dec_srl/base:204
