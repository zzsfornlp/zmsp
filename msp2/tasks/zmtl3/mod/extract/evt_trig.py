#

# extracting trigger words
import math
from typing import List
from itertools import chain
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import pandas as pd

from msp2.data.inst import yield_sents, yield_frames, set_ee_heads, DataPadder, SimpleSpanExtender, Sent
from msp2.data.rw import WriterGetterConf
from msp2.data.vocab import SimpleVocab
from msp2.utils import zlog, zwarn, ZRuleFilter, Random, StrHelper, F1EvalEntry, MathHelper, wrap_color
from msp2.nn import BK
from msp2.nn.l3 import *
from msp2.proc import FrameEvalConf, FrameEvaler, ResultRecord
from .base import *
from .layer_lab import *
from ..pretrained import *

class ZTaskTrigConf(ZTaskBaseEConf):
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name if name is not None else 'evt'
        self.trig_conf = ZModTrigConf()
        self.trig_eval = FrameEvalConf.direct_conf(weight_frame=1., weight_arg=0., bd_frame='', bd_frame_lines=50)

    def build_task(self):
        return ZTaskTrig(self)

class ZTaskTrig(ZTaskBaseE):
    def __init__(self, conf: ZTaskTrigConf):
        super().__init__(conf)
        conf: ZTaskTrigConf = self.conf
        # --
        self.evaler = FrameEvaler(conf.trig_eval)

    # build vocab
    def build_vocab(self, datasets: List):
        # note: still build a vocab that covers all
        voc_evt = SimpleVocab.build_empty(f"voc_{self.name}")
        for dataset in datasets:
            if dataset.wset == 'train':
                for evt in yield_frames(dataset.insts):
                    voc_evt.feed_one(evt.label)
        voc_evt.build_sort()
        zlog(f"Finish building for: {voc_evt}")
        return (voc_evt, )

    # eval
    def eval_insts(self, gold_insts: List, pred_insts: List, quite=False):
        evaler = self.evaler
        evaler.reset()
        # --
        if evaler.conf.span_mode_frame != 'span':
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
        return self.conf.trig_conf.make_node(self, model)

# --

class ZModTrigConf(ZModBaseEConf):
    def __init__(self):
        super().__init__()
        # --
        self.max_seq_len = 128  # maximum seq length
        self.upos_filter = []  # filtering by UPOS, by default allow all; an example: "+NV,-*"
        self.mask_trg_rate = 0.  # repl target inputs with [MASK]?
        self.loss_lab = 1.  # loss weight
        self.lab = LabConf()  # lab head
        self.use_io = False  # use IO tagging rather than head
        self.extend_span = True  # extend span for triggers
        self.frame_ignore_others = True  # whether ignore other pos (mark as -1) in frame mode, otherwise mark as neg
        self.do_standardize = False
        # extend context by sent?
        self.ctx_nsent = 0  # how many more before & after?
        # predict threshold
        self.pred_prob_thresh = 0.  # prob need to be >= this!
        # for special test (also utilized in setup/prep)
        self.special_seed = 54321
        self.special_np_ratio = 1.  # sampling neg-vs-pos in special mode
        self.special_exclude_self = False  # whether rebuild embs by exclude self in special_scoring
        # --
        # setup before training/testing
        self.setup_dname_train = []  # special name to setup things (for training)
        self.setup_dname_test = []  # special name to setup things (for testing)
        self.setup_method = "embs"  # which one to setup: embs/proj
        self.setup_do_prep = False  # further do prep when setup
        self.setup_interval = 10000  # how many interval to setup (by default very large to make it only once at init!)
        # --
        # prepare before testing
        self.prep_method = "neg_delta"  # where to put it: neg_delta/prob_thresh
        self.prep_ranges = [0.]  # from less to more restrict, trying to push towards more but not too restrict
        self.prep_tol = 0.  # accept the more restrict one if <= tol than previous better one
        # --
        # repr
        self.repr_conf = ZEvtReprConf()
        # extra augmentation
        self.aug_conf = ZEvtAugConf()
        # --

@node_reg(ZModTrigConf)
class ZModTrig(ZModBaseE):
    def __init__(self, conf: ZModTrigConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModTrigConf = self.conf
        # --
        self.upos_filter = None if len(conf.upos_filter)==0 else \
            ZRuleFilter(conf.upos_filter, {"NV": {'NOUN', 'VERB'}}, default_value=True)
        self.voc = ztask.vpack[0]
        # --
        self.repr = conf.repr_conf.make_node(evt_layer=self)
        self.aug = conf.aug_conf.make_node(evt_layer=self)
        # --
        _odim = self.repr.get_output_dim()
        self.lab = conf.lab.make_node(voc=self.voc, isize=_odim)  # lab head
        self.extender = SimpleSpanExtender.get_extender('evt') if conf.extend_span else None
        self.pred_lp_thresh = math.log(conf.pred_prob_thresh + 1e-5)  # at least 1e-5
        # --
        self.cache_version = 0  # current cache version!
        self.setup_enter = 0
        self.extra_embs = None
        if conf.do_standardize:
            self.standardize_bias = BK.nn.Parameter(BK.input_real([0.] * _odim))
            self.standardize_weight = BK.nn.Parameter(BK.input_real([1.] * _odim))
        else:
            self.standardize_bias, self.standardize_weight = None, None
        # --

    def update_cache_version(self):
        self.cache_version += 1
        zlog(f"Update cache_version to {self.cache_version}!!")

    def reset_value(self, key: str, value):
        old_value = getattr(self, key, None)
        zlog(f"Reset value {key} from {old_value} to {value}")
        setattr(self, key, value)

    def do_forward(self, ids_t, ids_mask_t, sublens_t, arr_toks, no_standardize=False):
        # forward bert
        bert_out = self.bmod.forward_enc(ids_t, ids_mask_t)
        # how to gather the reprs?
        ret_t = self.repr(bert_out, sublens_t, arr_toks)
        if not no_standardize and self.standardize_bias is not None:
            ret_t = (ret_t - self.standardize_bias) * self.standardize_weight
        return ret_t

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModTrigConf = self.conf
        # prepare data
        ids_t, ids_mask_t, sublens_t, labs_t, lab_mask_t, arr_toks = \
            self.prepare_ibatch(rc.ibatch, mask_trg_rate=conf.mask_trg_rate, frame_ignore_others=conf.frame_ignore_others)
        # forward bert
        hid0_t = self.do_forward(ids_t, ids_mask_t, sublens_t, arr_toks)  # [bs, len0, D]
        # get loss
        _loss_items = self.lab.do_loss(hid0_t, labs_t, lab_mask_t)
        ret = LossHelper.compile_component_loss(self.name, _loss_items, conf.loss_lab)
        return ret, {}

    def setup(self, gr: ZObject, *args, **kwargs):
        conf: ZModTrigConf = self.conf
        setup_dnames = conf.setup_dname_train if gr.training else conf.setup_dname_test
        # note: here we need it to be a training set
        datasets = [z for z in gr.d_center.get_datasets(wset='train') if z.name in setup_dnames]
        if len(datasets) > 0:  # go to setup!
            # --
            self.setup_enter += 1
            if (self.setup_enter - 1) % conf.setup_interval != 0:
                zlog(f"Skip setup for {self.setup_enter}")
                return
            else:
                zlog(f"Enter setup for {self.setup_enter}")
            # --
            zlog(f"Start to setup[training] for {self.extra_repr()} with {datasets}")
            extra_input_ts, extra_lab_ts = [], []
            # --
            # first do std
            if conf.do_standardize:
                with BK.no_grad_env():
                    zlog("First do_standardize before setup!")
                    ibatch_stream = (z for d in datasets for z in d.yield_batches(loop=False))
                    all_input_t, _, _ = self.forward_ibatch_stream(ibatch_stream)
                    _bias, _w = all_input_t.mean(0), 1./all_input_t.std(0)
                    BK.set_value(self.standardize_bias, _bias)
                    BK.set_value(self.standardize_weight, _w)
                    # breakpoint()
            # --
            with BK.no_grad_env():  # try to do selftrain!
                res = self.aug.do_selftrain(datasets)
                if res is not None:
                    extra_input_ts.append(res[0])
                    extra_lab_ts.append(res[1])
            with BK.no_grad_env():  # try to do extrafilter!
                res = self.aug.do_fextra(datasets, gr)
                if res is not None:
                    extra_input_ts.append(res[0])
                    extra_lab_ts.append(res[1])
            with BK.no_grad_env():
                ibatch_stream = (z for d in datasets for z in d.yield_batches(loop=False))
                all_input_t, all_lab_t, all_toks_arr = self.forward_ibatch_stream(ibatch_stream)
                lab_count, lab_pos_count = len(all_lab_t), (all_lab_t>0).sum().item()
                # --
                assert conf.setup_method == 'embs', "Currently only support this mode!"
                if len(extra_lab_ts) > 0:
                    c_input_t, c_lab_t = BK.concat([all_input_t]+extra_input_ts, 0), BK.concat([all_lab_t]+extra_lab_ts, 0)
                else:
                    c_input_t, c_lab_t = all_input_t, all_lab_t
                self.lab.create_embs(c_input_t, c_lab_t, exclude_self=False, assign=True)
                zlog(f"Finish with {lab_count}({lab_pos_count}/{lab_count-lab_pos_count})")
                # --
                # assign extras
                self.extra_embs = [extra_input_ts, extra_lab_ts]
                # --
                # further setup?
                if conf.setup_do_prep:
                    _tune_score_t, _tune_lab_t, _ = self.do_special_scoring(all_input_t, all_lab_t, all_toks_arr)
                    self.prepare_tune(_tune_score_t, _tune_lab_t)
        else:
            zlog(f"No setup[training] for {self.extra_repr()}")
        # --

    # [**, 1+??], [**]
    def prepare_tune(self, all_score_t, all_lab_t):
        conf: ZModTrigConf = self.conf
        # --
        # simple searching
        _ranges = conf.prep_ranges
        assert len(_ranges) > 0
        all_res = []
        for vv in _ranges:
            if conf.prep_method == 'neg_delta':
                _tmp_score_t = all_score_t.clone()
                _tmp_score_t[:, 0] += vv
            elif conf.prep_method == 'prob_thresh':
                _tmp_score_t = all_score_t.softmax(-1)
                _tmp_score_t[_tmp_score_t.max(-1)[0] < vv, 1:] = 0.
            else:
                raise NotImplementedError(f"UNK prep_method {conf.prep_method}")
            _pred_lab_t = _tmp_score_t.argmax(-1)
            _hit_num = ((all_lab_t>0) & (all_lab_t==_pred_lab_t)).sum().item()
            f1 = F1EvalEntry()
            f1.record_p(_hit_num, (_pred_lab_t>0).sum().item())
            f1.record_r(_hit_num, (all_lab_t>0).sum().item())
            all_res.append([round(z, 4) for z in f1.prf])
        # --
        # note: not the best, but the most strict one that does not suffer too much!
        # best_vidx = np.argmax(all_acc)
        best_vidx, best_res = 0, all_res[0]
        _tol = conf.prep_tol
        for _ii, _res in enumerate(all_res[1:], 1):
            if _res[-1] >= best_res[-1] - _tol:
                best_vidx = _ii
                if _res[-1] > best_res[-1]:
                    best_res = _res
        zlog(f"Prep_test({conf.prep_method}): RES={[(a, b) for a, b in zip(_ranges, all_res)]}, "
             f"best={best_vidx}/{_ranges[best_vidx]}/{all_res[best_vidx]}")
        # --
        # set it!
        best_vv = _ranges[best_vidx]
        if conf.prep_method == 'neg_delta':
            self.lab.neg_delta.reset_value(best_vv)
        elif conf.prep_method == 'prob_thresh':
            self.reset_value('pred_lp_thresh', math.log(best_vv + 1e-5))
        else:
            raise NotImplementedError(f"UNK prep_method {conf.prep_method}")
        # --

    # forward full ibatch_stream and collect valid ones
    def forward_ibatch_stream(self, ibatch_stream):
        all_inputs, all_labs, all_toks = [], [], []
        for ibatch in ibatch_stream:
            ids_t, ids_mask_t, sublens_t, labs_t, lab_mask_t, arr_toks = self.prepare_ibatch(ibatch)
            hid0_t = self.do_forward(ids_t, ids_mask_t, sublens_t, arr_toks)  # [bs, len0, D]
            # apply mask
            _mt = (lab_mask_t > 0)  # [bs, len0]
            all_inputs.append(hid0_t[_mt])
            all_labs.append(labs_t[_mt])
            all_toks.extend(arr_toks[BK.get_value(_mt)])
        all_input_t, all_lab_t = BK.concat(all_inputs, 0), BK.concat(all_labs, 0)
        all_toks_arr = np.asarray(all_toks)
        return all_input_t, all_lab_t, all_toks_arr

    # special scoring
    def do_special_scoring(self, all_input_t, all_lab_t, all_toks_arr):
        conf: ZModTrigConf = self.conf
        _special_exclude_self = conf.special_exclude_self
        # --
        # pos_proto with exclude_self
        _pos_t = (all_lab_t > 0)
        _pos_input_t = all_input_t[_pos_t]  # [pos-bs, D]
        _pos_lab_t = all_lab_t[_pos_t]  # [pos-bs]
        if self.extra_embs is not None:
            _pe_input_t, _pe_lab_t = BK.concat([_pos_input_t] + self.extra_embs[0], 0), BK.concat([_pos_lab_t] + self.extra_embs[1], 0)
            _pe_proto_t, _, _ = self.lab.create_embs(_pe_input_t, _pe_lab_t, exclude_self=True, neg_k=0)
            pos_proto_t = _pe_proto_t[:len(_pos_lab_t)]  # only need these!
        else:
            pos_proto_t, _, _ = self.lab.create_embs(_pos_input_t, _pos_lab_t, exclude_self=True, neg_k=0)  # [pos-bs, ??, D]
        # --
        # sample neg targets
        _neg_t = (all_lab_t == 0)
        _gen = Random.get_np_generator(conf.special_seed)  # note: use the same special seed!
        _count_all, _count_pos = len(all_lab_t), len(_pos_lab_t)
        _count_neg = _count_all - _count_pos
        _sel_count = min(_count_neg, int(_count_pos * conf.special_np_ratio + 0.99999))
        # for reproducibility
        _neg_arr = BK.get_value(_neg_t)
        _neg_ids = sorted([(zz.get_full_id(), ii) for ii,zz in enumerate(all_toks_arr) if _neg_arr[ii]])
        _gen.shuffle(_neg_ids)
        _sel_neg_idx_t = BK.input_idx([z[-1] for z in _neg_ids[:_sel_count]])
        _sel_t = (_neg_t.float() > 100)  # all False
        _sel_t[_sel_neg_idx_t] = True
        # two sets of negs
        _neg_sel_t = _neg_t & _sel_t
        _neg_train_t = _neg_t & (~ _sel_t)
        # get neg_proto
        _neg_train_input_t = all_input_t[_neg_train_t]
        _, neg_proto_t, _ = self.lab.create_embs(
            _neg_train_input_t, BK.constants_idx([len(_neg_train_input_t)], 0), exclude_self=False)  # [Nk, D]
        # --
        # we also need plain protos
        pos_proto_tp, neg_proto_tp, _ = self.lab.create_embs(all_input_t, all_lab_t, exclude_self=False)
        if neg_proto_tp is not None:
            neg_proto_tp = neg_proto_tp.unsqueeze(0)  # [1, Nk, D]
        # --
        # scoring
        if _special_exclude_self:
            _pw = self.lab.proj_weight
            _pos_score_t = self.lab.do_score(_pos_input_t.unsqueeze(-2), _pw, pos_proto_t, neg_proto_tp, is_pair=False)
            _neg_score_t = self.lab.do_score(all_input_t[_neg_sel_t], _pw, pos_proto_tp, neg_proto_t, is_pair=True)
        else:  # simply use the pre-build ones!
            _pos_score_t = self.lab.do_my_score(_pos_input_t)
            _neg_score_t = self.lab.do_my_score(all_input_t[_neg_sel_t])
        # put them together: [pos+neg]
        ret_score_t = BK.concat([_pos_score_t, _neg_score_t], 0)  # [pos+neg]
        ret_lab_t = BK.concat([_pos_lab_t, BK.constants_idx([len(_neg_score_t)], 0)], 0)  # [pos+neg]
        zlog(f"Special scoring for pos+neg/neg_all = {len(_pos_score_t)}+{len(_neg_score_t)}/{_neg_t.float().sum().item()}")
        ret_toks = np.concatenate([all_toks_arr[BK.get_value(_pos_t)], all_toks_arr[BK.get_value(_neg_sel_t)]], 0)
        # --
        # breakpoint()
        # --
        return ret_score_t, ret_lab_t, ret_toks

    def get_other_inst(self, curr, other_name: str):
        if other_name in curr._cache:
            return curr._cache[other_name]
        else:
            other_doc = curr.doc._cache[other_name]
            return other_doc.sents[curr.sid]

    # do special test!
    def do_special_test(self, dataset):
        # --
        if self.name not in dataset.tasks:
            return {}
        # --
        from ...core.run import InputBatch, DataItem
        conf: ZModTrigConf = self.conf
        assert not conf.use_io, "not support this mode!"
        with BK.no_grad_env():
            # special scoring
            _ = dataset.gold_insts  # first specify them
            ibatch_stream = (InputBatch([DataItem(self.get_other_inst(z2.sent, 'inst_gold')) for z2 in z.items], dataset)
                             for z in dataset.yield_batches(loop=False))  # get gold stream
            all_input_t, all_lab_t, all_toks_arr = self.forward_ibatch_stream(ibatch_stream)
            p_score_t, _, p_toks = self.do_special_scoring(all_input_t, all_lab_t, all_toks_arr)
            # first clean all insts
            for inst in dataset.insts:
                for sent in yield_sents(inst):
                    sent.clear_events()  # simply clean them all!
            # then do decoding
            arr_p_lprob, arr_p_idx = [BK.get_value(z) for z in p_score_t.log_softmax(-1).max(-1)]
            for gidx, gtok in enumerate(p_toks):
                if arr_p_idx[gidx]>0 and arr_p_lprob[gidx]>=self.pred_lp_thresh:
                    psent = self.get_other_inst(gtok.sent, 'inst_pred')
                    new_frame = psent.make_event(gtok.widx, 1, type=self.voc.idx2word(arr_p_idx[gidx]))
                    new_frame.set_score(arr_p_lprob[gidx].item())
                    if self.extender is not None:
                        self.extender.extend_mention(new_frame.mention)
        # --
        return {}

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModTrigConf = self.conf
        # prepare data
        ids_t, ids_mask_t, sublens_t, labs_t, lab_mask_t, arr_toks = self.prepare_ibatch(rc.ibatch, clear_frames=True)
        # forward bert
        hid0_t = self.do_forward(ids_t, ids_mask_t, sublens_t, arr_toks)  # [bs, len0, D]
        # get score
        score_t = self.lab.do_my_score(hid0_t)  # [bs, len0, 1+L]
        # breakpoint()
        # --
        # simply argmax decode!
        lprob_t = score_t.log_softmax(-1)  # [bs, len0, 1+L]
        _max_lprob_t, _max_idx_t = lprob_t.max(-1)  # [bs, len0]
        _zip_items = [BK.get_value(z) for z in [_max_lprob_t, _max_idx_t, lab_mask_t]] + [arr_toks]
        for bidx, (arr_lp, arr_i, arr_m, arr_ts) in enumerate(zip(*_zip_items)):
            item = rc.ibatch.items[bidx]
            sent, frame = item.sent, item.frame
            if frame is None:  # assign all
                _res = self.labels2trg(arr_lp, arr_i, arr_m, 0, self.pred_lp_thresh)
                for _widx, _wlen, _lab, _score in _res:
                    real_widx = arr_ts[_widx].widx
                    new_frame = sent.make_event(real_widx, _wlen, type=self.voc.idx2word(_lab))
                    new_frame.set_score(_score)
                    if self.extender is not None:
                        self.extender.extend_mention(new_frame.mention)
            else:  # set this specific one, and ignore mask!
                _tidx = [ii for ii,zz in enumerate(arr_ts) if zz is not None and zz.widx==frame.mention.shead_widx]
                if len(_tidx) != 1:
                    zwarn(f"Strange idx: {frame} {arr_ts}")
                    continue
                frame.set_score(arr_lp[_tidx[0]].item())
                frame.set_label(self.voc.idx2word(arr_i[_tidx[0]]))
        # --
        return {}

    def labels2trg(self, scores, labels, masks, offset: int, score_thresh: float):
        conf: ZModTrigConf = self.conf
        # --
        def _avg(x):
            return float(sum(x) / len(x))
        # --
        assert all(z>=0 for z in labels), "Invalid label!!"
        if conf.use_io:
            last_start = -1
            last_one = 0
            ret0 = []
            for widx, (one, mm) in enumerate(zip(chain(labels, [0]), chain(masks, [False]))):  # add a sentinel
                if one > 0 and mm and one == last_one:  # continue
                    pass
                else:
                    # close prev
                    if last_one > 0:
                        ret0.append((offset+last_start, widx-last_start, last_one, _avg(scores[last_start:widx])))
                    # update
                    last_start = widx
                    last_one = one if mm else 0
        else:  # individual ones!
            ret0 = [(offset+widx, 1, one, float(scores[widx])) for widx, one in enumerate(labels) if one>0 and masks[widx]]
        # --
        # filter by score-thresh
        ret = [z for z in ret0 if z[-1]>=score_thresh]
        return ret

    # --
    # prepare input instances

    def prepare_ibatch(self, ibatch, mask_trg_rate=0., frame_ignore_others=False, clear_frames=False, no_cache=False):
        _key = f"_cache_{self.name}_{self.cache_version}"
        # --
        _gen = Random.stream(Random.get_generator("train").random_sample)
        _tokenizer = self.tokenizer
        _mask_id, _pad_id = _tokenizer.mask_token_id, _tokenizer.pad_token_id
        # --
        # mostly in testing, we should first clear current results unless frame-mode
        if clear_frames:
            for item in ibatch.items:
                if item.frame is None:
                    item.sent.clear_events()  # simply clean them all!
        # --
        # collect them!
        arr_ids, arr_sublens, arr_labs, arr_toks = [], [], [], []
        for item in ibatch.items:
            _cache = item.info.get(_key)
            if _cache is None or no_cache:
                _cache = self.prepare_item(item, frame_ignore_others)
                if not no_cache:
                    item.info[_key] = _cache
            sub_ids, trg_labels, seq_toks = _cache
            # do mask_trg
            arr_labs.append(trg_labels)
            cur_ids, cur_lens = [], []  # [len1, len0, len0]
            for _ids, _lab in zip(sub_ids, trg_labels):
                if _lab > 0 and mask_trg_rate > 0. and next(_gen) < mask_trg_rate:
                    cur_ids.append(_mask_id)  # simply put a mask!
                    cur_lens.append(1)
                else:
                    cur_ids.extend(_ids)
                    cur_lens.append(len(_ids))
            arr_ids.append(cur_ids)
            arr_sublens.append(cur_lens)
            arr_toks.append(seq_toks)
        # --
        # batch them!
        arr_ids = DataPadder.go_batch_2d(arr_ids, _pad_id)  # [bs, len1]
        arr_sublens = DataPadder.go_batch_2d(arr_sublens, 0)  # [bs, len0]
        arr_labs = DataPadder.go_batch_2d(arr_labs, -1)  # [bs, len0]
        arr_toks = DataPadder.go_batch_2d(arr_toks, None, dtype=object)  # [bs, len0]
        # tensor
        ids_t, sublens_t, labs_t = [BK.input_idx(z) for z in [arr_ids, arr_sublens, arr_labs]]
        ids_mask_t = (ids_t != _pad_id).float()
        lab_mask_t = (labs_t >= 0).float()  # -1 as either non-target or padding
        labs_t.clamp_(min=0)  # make it all >=0
        return ids_t, ids_mask_t, sublens_t, labs_t, lab_mask_t, arr_toks  # [bs, len1] ... [bs, len0] ...

    def prepare_labels(self, trg_list, mention, lab):
        conf: ZModTrigConf = self.conf
        if conf.use_io:
            widx, wlen = mention.get_span()
            trg_list[widx:widx+wlen] = [lab] * wlen
        else:
            trg_list[mention.shead_widx] = lab
        # --

    def prepare_sent_plus(self, sent):
        sub_ids = self.prep_sent(sent)  # get subtokens, [olen], List[List]
        sub_lens = [len(z) for z in sub_ids]  # [olen]
        # then prepare targets
        trg_labels = [0] * len(sub_ids)  # [olen], only initial ones!
        if self.upos_filter is not None:
            for _widx, _upos in enumerate(sent.seq_upos.vals):
                if not self.upos_filter.filter_by_name(_upos):
                    trg_labels[_widx] = -1  # mark as negative!
        toks = sent.tokens
        return sub_ids, sub_lens, trg_labels, toks

    def prepare_item(self, item, frame_ignore_others: bool):
        conf: ZModTrigConf = self.conf
        # --
        voc_evt = self.voc
        # prepare for one item
        sent = item.sent
        set_ee_heads(sent)
        sub_ids, sub_lens, trg_labels, seq_toks = self.prepare_sent_plus(sent)
        frame = item.frame
        if frame is None:  # put all
            for evt in sent.events:  # note: use 0 if not in voc!
                self.prepare_labels(trg_labels, evt.mention, voc_evt.get(evt.label, 0))
            truncate_center = len(sent) // 2  # simply use center!
        else:  # only one
            if frame_ignore_others:  # note: use 0 if not in voc!
                for evt in sent.events:  # by default set others as -1
                    self.prepare_labels(trg_labels, evt.mention, -1)
            self.prepare_labels(trg_labels, frame.mention, voc_evt.get(frame.label, 0))
            truncate_center = frame.mention.shead_widx
        # extend context?
        if conf.ctx_nsent > 0:
            before_ids, after_ids = self.extend_ctx_sent(sent, conf.ctx_nsent)
            _before_len, _after_len = len(before_ids), len(after_ids)
            truncate_center += _before_len  # note: remember to add this!
            sub_ids, sub_lens, trg_labels, seq_toks = before_ids+sub_ids, [len(z) for z in before_ids]+sub_lens,\
                                                      [-1]*_before_len+trg_labels, [None]*_before_len+seq_toks
            sub_ids, sub_lens, trg_labels, seq_toks = sub_ids+after_ids, sub_lens+[len(z) for z in after_ids], \
                                                      trg_labels+[-1]*_after_len, seq_toks+[None]*_after_len
        # check max_len & truncate
        if sum(sub_lens) > conf.max_seq_len:  # truncate things!
            t_start, t_end = self.truncate_subseq(sub_lens, truncate_center, conf.max_seq_len, silent=True)
            sub_ids, sub_lens, trg_labels = sub_ids[t_start:t_end], sub_lens[t_start:t_end], trg_labels[t_start:t_end]
            seq_toks = seq_toks[t_start:t_end]
        # construct seq
        _tokenizer = self.tokenizer
        ret = [[_tokenizer.cls_token_id]] + sub_ids + [[_tokenizer.sep_token_id]], \
              [-1] + trg_labels + [-1], [None] + seq_toks + [None]
        return ret

# =====
# augmentation: filtering & self-training

class ZEvtAugConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        self.selft_max_iter = 0  # maximum self-train iteration
        self.selft_save_prefix = "selft.tmp"  # "{name}.i{iter}.json"
        self.selft_cluster_num = 1  # number of points (per type) to add per iter
        self.selft_thr = 0.02  # metric should >= this
        # --
        self.fextra_dname_train = []  # special name for extra (for training)
        self.fextra_dname_test = []  # special name for extra (for testing)
        # self.fextra_thr_f = "lambda arr: arr[1:].max()/(5+arr.sum()) >= 0.5"  # lambda function for the judgement
        self.fextra_thr_f = ""
        self.fextra_thr_fthr = 0.5
        self.fextra_thr_fadd = 5
        self.fextra_mix_mthr = 0.  # to mixin the items, score(argmax-idx) should be this>= second (margin)
        self.fextra_cluster_num = 1  # number of points (per fextra type) to add
        # --

@node_reg(ZEvtAugConf)
class ZEvtAugLayer(Zlayer):
    def __init__(self, conf: ZEvtAugConf, evt_layer, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZEvtAugConf = self.conf
        self.setattr_borrow('evt_layer', evt_layer)
        # --
        if conf.fextra_thr_f:
            self.fextra_thr_f = eval(conf.fextra_thr_f)
        else:
            self.fextra_thr_f = (lambda arr: arr[1:].max()/(conf.fextra_thr_fadd+arr.sum()) >= conf.fextra_thr_fthr)
        # --

    def judge_by_pthr(self, arr_counts):
        return self.fextra_thr_f(arr_counts)

    def save_cur_file(self, all_sents, all_toks, cur_lab_t, voc, output: str):
        # first copy all sents
        output_sents = OrderedDict()
        for sent in all_sents:
            output_sents[id(sent)] = sent.copy(ignore_fields={'entity_fillers', 'events'})
        # add valid ones
        arr_iidxes = BK.get_value(BK.arange_idx(len(all_toks))[cur_lab_t > 0])
        arr_labs = BK.get_value(cur_lab_t)
        for ii in arr_iidxes:
            _tok, _lab = all_toks[ii], arr_labs[ii]
            output_sents[id(_tok.sent)].make_event(_tok.widx, 1, type=voc.idx2word(_lab))
        # write
        with WriterGetterConf().get_writer(output_path=output) as writer:
            writer.write_insts(output_sents.values())
        # --

    def forward_sents(self, datasets, frame_only: bool):
        from ...core.run.plain import PlainBatcherConf, PlainBatcher, InputBatch
        # note: in this mode, we simply gather all sents; no worries about cache since they are inside batcher
        all_sents = sum([list(yield_sents(d.insts)) for d in datasets], [])
        batcher = PlainBatcher(PlainBatcherConf.direct_conf(inst_f='sent'))  # simply use the default one
        # --
        # get them all
        all_inputs, all_labs, all_toks, all_frames = [], [], [], []  # collect them all!
        for _sents in batcher.yield_batches(all_sents, loop=False):
            ibatch = InputBatch(_sents, None)
            ids_t, ids_mask_t, sublens_t, labs_t, lab_mask_t, arr_toks = self.evt_layer.prepare_ibatch(ibatch)
            hid0_t = self.evt_layer.do_forward(ids_t, ids_mask_t, sublens_t, arr_toks)  # [bs, len0, D]
            # --
            arr_lab_mask = BK.get_value(lab_mask_t)
            arr_sel = np.zeros_like(arr_lab_mask)  # [bs, len0]
            for bidx, item in enumerate(ibatch.items):
                # first get all frames
                sent = item.sent
                cur_frames = [None] * len(sent)
                for evt in sent.events:
                    cur_frames[evt.mention.shead_widx] = evt
                cur_toks = item.sent.tokens
                # --
                _widxes = [None if z is None else z.widx for z in arr_toks[bidx]]
                for ii, mm in enumerate(arr_lab_mask[bidx]):
                    widx = _widxes[ii]
                    if widx is None or widx<0 or widx>=len(sent): continue
                    # note: if has_frame or not frame-only mode
                    if (cur_frames[widx] is not None) or ((not frame_only) and mm):
                        arr_sel[bidx, ii] = 1.
                        all_toks.append(cur_toks[widx])
                        all_frames.append(cur_frames[widx])
            # --
            _mt = (BK.input_real(arr_sel) > 0)  # [bs, len0], exclude invalid ones
            all_inputs.append(hid0_t[_mt])
            all_labs.append(labs_t[_mt])
            # --
        # --
        all_input_t, all_lab_t = BK.concat(all_inputs, 0), BK.concat(all_labs, 0)
        assert len(all_input_t) == len(all_toks)
        # --
        return all_sents, all_input_t, all_lab_t, all_toks, all_frames  # [*]

    # best score - second score
    def get_margin(self, scores):
        _NEG = -10000.
        _last_dim = BK.get_shape(scores, -1)
        tmp_scores = scores.unsqueeze(-2).repeat(1, _last_dim, 1) + BK.eye(_last_dim) * _NEG  # [*, 1+L, 1+L]
        _margin = scores - tmp_scores.max(-1)[0]  # [*, 1+L]
        return _margin

    def do_selftrain(self, datasets):
        conf: ZEvtAugConf = self.conf
        if conf.selft_max_iter <= 0:
            return None  # no need to do anything!
        # --
        zlog(f"Prepare self-train: Step1-read: {datasets}")
        all_sents, all_input_t, all_lab_t, all_toks, all_frames = self.forward_sents(datasets, False)
        # iterate
        cur_lab_t = all_lab_t * 0  # [*], current assignments (of new labs)
        _voc_evt = self.evt_layer.voc
        _voc_size = self.evt_layer.lab.trg_num + 1
        _arange_t1, _arange_t2 = BK.arange_idx(len(cur_lab_t)), BK.arange_idx(_voc_size)  # [bs], [Laug]
        _NEG = -10000.
        iter_idx = 0
        _ce_input_ts, _ce_lab_ts = [], []  # extra ones
        while True:
            zlog(f"Start do self-train iter_idx={iter_idx}: P/ALL={(cur_lab_t>0).sum().item()}/{len(cur_lab_t)}")
            # --
            # create protos
            _ce_input_t, _ce_lab_t = BK.concat([all_input_t]+_ce_input_ts, 0), BK.concat([all_lab_t]+_ce_lab_ts, 0)
            pos_proto_t, neg_proto_t, label_map = self.evt_layer.lab.create_embs(_ce_input_t, _ce_lab_t, False)
            iter_idx += 1
            if iter_idx > conf.selft_max_iter:
                break
            # --
            # self-assign scores: [*, 1+L]
            self_scores = self.evt_layer.lab.do_score(
                all_input_t, self.evt_layer.lab.proj_weight, pos_proto_t, neg_proto_t, no_neg_delta=True)
            # score for each type
            _margin = self.get_margin(self_scores)
            _ranking_metric = _margin + (all_lab_t > 0).unsqueeze(-1).float() * _NEG  # [*, 1+L]
            # get new ones for each type
            _sorted_metric, _sorted_iidxes = _ranking_metric.sort(dim=0, descending=True)  # [*, 1+L]
            all_metrics, all_iidxes, all_clu_idxes = [], [], []  # for recording
            new_lab_t = all_lab_t * 0  # [*]
            # --
            _ce_input_ts, _ce_lab_ts = [], []
            for lab_idx in range(1, _voc_size):  # note: no idx0!
                _mask = (_sorted_metric[:, lab_idx] > conf.selft_thr)  # [*]
                _cur_metrics, _cur_iidxes = _sorted_metric[:, lab_idx][_mask], _sorted_iidxes[:, lab_idx][_mask]
                all_metrics.append(_cur_metrics), all_iidxes.append(_cur_iidxes)
                new_lab_t[_cur_iidxes] = lab_idx
                # cluster
                _k = min(conf.selft_cluster_num, len(_cur_iidxes))
                if _k > 0:
                    clu, clu_idxes = self.evt_layer.lab.sim.run_kmeans(all_input_t[_cur_iidxes], None, _k, return_idxes=True)  # [k, D]
                    all_clu_idxes.append(clu_idxes)
                    _ce_input_ts.append(clu)
                    _ce_lab_ts.append(BK.input_idx([lab_idx] * len(clu)))
            # --
            # pp (lambda ts,m,i,ci,ii: [(ts[z2],round(float(z1),4),int(z3)) for z1,z2,z3 in zip(m[ii],i[ii],ci[ii])])(all_toks, all_metrics, all_iidxes, all_clu_idxes, 0)
            # breakpoint()
            # go next
            cur_lab_t = new_lab_t
            if conf.selft_save_prefix:
                self.save_cur_file(all_sents, all_toks, cur_lab_t, _voc_evt, conf.selft_save_prefix + f".i{iter_idx}.json")
            # --
        # --
        # todo(+N): need to mark to update data cache??
        if len(_ce_lab_ts) > 0:
            return BK.concat(_ce_input_ts, 0), BK.concat(_ce_lab_ts, 0)
        else:
            return None
        # --

    # filter extra data!
    def do_fextra(self, datasets, gr):
        conf: ZEvtAugConf = self.conf
        _voc_evt = self.evt_layer.voc
        _voc_size = self.evt_layer.lab.trg_num + 1
        _arange_t2 = BK.arange_idx(_voc_size)  # [bs], [Laug]
        # --
        fextra_dnames = conf.fextra_dname_train if gr.training else conf.fextra_dname_test
        # note: here we need it to be a training set
        extra_datasets = [z for z in gr.d_center.get_datasets(wset='train') if z.name in fextra_dnames]
        if len(extra_datasets) == 0:
            return None
        # --
        # first get base ones with main data
        zlog(f"Prepare extra-filter: Step1-read: {datasets}")
        all_sents, all_input_t, all_lab_t, all_toks, all_frames = self.forward_sents(datasets, False)
        zlog(f"Prepare extra-filter: P/ALL={(all_lab_t>0).sum().item()}/{len(all_lab_t)}")
        pos_proto_t, neg_proto_t, label_map = \
            self.evt_layer.lab.create_embs(all_input_t, all_lab_t, False)
        # --
        # read and forward extra data
        mixin_repr_ts, mixin_labs = [], []
        for one_extra_dataset in extra_datasets:
            zlog(f"Processing extra-filter: {one_extra_dataset}")
            _, extra_input_t, _, extra_toks, extra_frames = self.forward_sents([one_extra_dataset], True)
            extra_scores = self.evt_layer.lab.do_score(  # [Ne, 1+L]
                extra_input_t, self.evt_layer.lab.proj_weight, pos_proto_t, neg_proto_t, no_neg_delta=True)
            extra_margin = self.get_margin(extra_scores)  # [Ne, 1+L], margin to idx0(NIL)
            # --
            # assign margin scores
            for ii, ff in enumerate(extra_frames):
                ff._cache['fextra_margin'] = BK.get_value(extra_margin[ii])
            for batcher in one_extra_dataset.cur_batchers:
                batcher.rearrange()  # re-arrange things
            zlog(f"Update fextra_margin for {len(extra_frames)} in {one_extra_dataset}!")
            # --
            # group frames
            frame_cols = OrderedDict()
            for ii, ff in enumerate(extra_frames):
                if ff.type not in frame_cols:
                    frame_cols[ff.type] = ZObject(fidx=len(frame_cols), eidxes=[])
                frame_cols[ff.type].eidxes.append(ii)
            # calculate for each frame
            df0 = defaultdict(list)
            df1 = defaultdict(list)
            for zkey, zobj in frame_cols.items():
                eidx_t = BK.input_idx(zobj.eidxes)
                scores_t = extra_scores[eidx_t]  # [Nf, 1+L]
                _arr_counts = BK.get_value(scores_t.max(-1)[-1].unsqueeze(-1) == _arange_t2).sum(0)  # [1+L]
                best_idx = _arr_counts.argmax()
                best_lab = _voc_evt.idx2word(best_idx)
                # --
                _ac_info = {
                    'zkey': zkey, 'zlab': best_lab,
                    'all': _arr_counts.sum().item(), 'non0': _arr_counts[1:].sum().item(), 'max': _arr_counts[1:].max().item(),
                }
                for k0, k1 in [('non0', 'all'), ('max', 'all'), ('max', 'non0')]:
                    _ac_info[f"{k0}_{k1}"] = round(MathHelper.safe_div(_ac_info[k0], _ac_info[k1]), 4)
                zobj.update(zkey=zkey, zlab=best_lab, eidx_t=eidx_t, scores_t=scores_t, arr_counts=_arr_counts, ac_info=_ac_info)
                for _k, _v in _ac_info.items():
                    df0[_k].append(_v)
                # --
                if self.judge_by_pthr(_arr_counts):  # mix-in
                    midx_t = eidx_t[extra_margin[eidx_t, best_idx] >= conf.fextra_mix_mthr]
                    if len(midx_t) > 0:
                        # mixin_repr_ts.append(extra_input_t[midx_t].mean(0, keepdims=True))  # [1, D]
                        # mixin_labs.append(best_idx)
                        to_add_t = self.evt_layer.lab.sim.run_kmeans(
                            extra_input_t[midx_t].unsqueeze(0), None, conf.fextra_cluster_num).squeeze(0)  # [k, D]
                        mixin_repr_ts.append(to_add_t)
                        mixin_labs.extend([best_idx] * conf.fextra_cluster_num)
                        for _k, _v in _ac_info.items():
                            df1[_k].append(_v)
            # --
            df0, df1 = pd.DataFrame(df0), pd.DataFrame(df1)
            df1 = df1.sort_values(by=['max_all'], ascending=False)
            zlog(f"Hit extra-filter for {one_extra_dataset} [{len(df1)}]:\n{df1.to_string()}")
        # --
        # return things to mix
        zlog(f"Return mixin of {len(mixin_labs)}: {Counter(mixin_labs)}")
        if len(mixin_labs) > 0:
            return BK.concat(mixin_repr_ts, 0), BK.input_idx(mixin_labs)
        else:
            return None
        # --

# =====
# repr

class ZEvtReprConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        # bert output
        self.bert_out = BertOuterConf()  # how to get bert output?
        self.sub_pooler = SubPoolerConf()
        # --
        # note: special formats: "name:...:mix=??"
        # dep: ch_??, par_??; srl: pb_??, nb_??, fn_??
        self.comps = ['self']
        self.comp0_putself = False  # if 0. (for the att-styled ones, simply put self!)
        # --

@node_reg(ZEvtReprConf)
class ZEvtReprLayer(Zlayer):
    def __init__(self, conf: ZEvtReprConf, evt_layer, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZEvtReprConf = self.conf
        self.setattr_borrow('evt_layer', evt_layer)
        # --
        self.bert_out = conf.bert_out.make_node(bert_dim=evt_layer.bmod.get_mdim(), att_num=evt_layer.bmod.get_head_num())
        _bert_out_dim = self.bert_out.dim_out_hid()  # output of bert_outer
        self.sub_pooler = conf.sub_pooler.make_node()  # sub pooler
        self.output_dim = _bert_out_dim * len(conf.comps)  # number of components
        # ==
        # prase comps
        self.comps = []
        for comp in conf.comps:
            fields = comp.split(":")
            _type = fields[0]
            _vals = [] if len(fields)<=1 else fields[1].split("|")
            _others = {z.split('=',1)[0]: float(z.split('=',1)[1]) for z in fields[2:]}
            self.comps.append((_type, _vals, _others))
        # --

    def get_output_dim(self):
        return self.output_dim

    def forward(self, bert_out, sublens_t, arr_toks):
        conf: ZEvtReprConf = self.conf
        # --
        hid1_t = self.bert_out.forward_hid(bert_out)  # [bs, len1, D]
        hid0_t = self.sub_pooler.forward_hid(hid1_t, sublens_t)  # [bs, len0, D]
        # mask0_t = (self.sub_pooler.forward_hid(ids_mask_t, sublens_t)>0).float()  # [bs, len0]
        mask0_arr = np.asarray([0. if z is None else 1. for z in arr_toks.flatten()], dtype=np.float32)
        mask0_t = BK.input_real(mask0_arr).view(arr_toks.shape)
        _len0 = BK.get_shape(mask0_t, -1)  # len0
        rets = []
        for comp in self.comps:
            _type, _vals, _others = comp
            if _type == 'self':
                one_t = hid0_t  # simply self!
            elif _type == 'cls':
                one_t = hid0_t[..., 0:1, :].expand_as(hid0_t)  # use [CLS]
            else:  # other modes need to construct pairwise weights
                if _type == 'att':
                    att1_t = self.bert_out.forward_att(bert_out)  # [bs, len1, len1, D]
                    att0_t = self.sub_pooler.forward_att(att1_t, sublens_t)  # [bs, len0, len0, D]
                    w_t = att0_t.mean(-1)  # [bs, len0, len0]
                elif _type == 'sim':  # pairwise similarity score
                    s0_t = BK.F.cosine_similarity(hid0_t.unsqueeze(-2), hid0_t.unsqueeze(-3), dim=-1)
                    w_t = ((s0_t - 1000. * BK.eye(BK.get_shape(s0_t, -1))) / 0.1).softmax(-1)  # [bs, len0, len0]
                elif _type == 'before':
                    _arange_t = BK.arange_idx(_len0)  # [len0]
                    w_t = (_arange_t.unsqueeze(-1) < _arange_t.unsqueeze(-2)).float()  # [len0, len0]
                elif _type == 'after':
                    _arange_t = BK.arange_idx(_len0)  # [len0]
                    w_t = (_arange_t.unsqueeze(-1) > _arange_t.unsqueeze(-2)).float()  # [len0, len0]
                elif _type == 'win':
                    _arange_t = BK.arange_idx(_len0)  # [len0]
                    _d_t = (_arange_t.unsqueeze(-1) - _arange_t.unsqueeze(-2)).abs()  # [len0, len0]
                    w_t = ((_d_t!=0) & (_d_t<=int(_vals[0]))).float()  # [len0, len0], note: take first val!
                elif _type == 'dep':
                    w_t = self.get_w_pair(_vals, arr_toks, self.get_sw_dep)  # [bs, len0, len0]
                elif _type == 'srl':
                    w_t = self.get_w_pair(_vals, arr_toks, self.get_sw_srl)  # [bs, len0, len0]
                else:
                    raise NotImplementedError(f"UNK comp-type of {_type}")
                w_t = w_t * mask0_t.unsqueeze(-2)  # [bs, len0, len0]
                w_t_sum = w_t.sum(-1, keepdims=True)  # [bs, len0, 1]
                nw_t = (w_t / (1e-8 + w_t_sum))  # [bs, len0, len0]
                one_t = BK.matmul(nw_t, hid0_t)  # [bs, len0, D]
                if conf.comp0_putself:  # simply put self if the component is 0!
                    is0_t = (w_t_sum==0.).float()
                    one_t = is0_t * hid0_t + (1.-is0_t) * one_t
            # --
            # mix
            _mix = _others.get('mix', 0.)
            if _mix > 0.:
                one_t = one_t * (1.-_mix) + hid0_t * _mix
            # --
            rets.append(one_t)
        # --
        # simply concat togather
        final_ret = BK.concat(rets, -1)
        return final_ret

    def get_w_pair(self, vals, arr_toks, sw_ff):
        _shape = list(arr_toks.shape)
        w_arr = np.zeros(_shape + [_shape[-1]], dtype=np.float32)  # [bs, len0, len0]
        val_set = [':'.join(z.split("_", 1)) for z in vals]
        for bidx, one_toks in enumerate(arr_toks):
            valid_toks = [(i,z) for i,z in enumerate(one_toks) if z is not None]
            if len(valid_toks) == 0:
                zwarn("Empty sequence?")
                continue
            (i0,t0), (i1,t1) = valid_toks[0], valid_toks[-1]  # first and last?
            s_arr = sw_ff(t0.sent, val_set)
            w_arr[bidx, i0:(i1+1), i0:(i1+1)] = s_arr[t0.widx:t1.widx+1, t0.widx:t1.widx+1]  # truncate
            # --
            # # for debugging
            # for evt in t0.sent.events:
            #     widx = evt.mention.shead_widx
            #     print(f"{evt} {t0.sent.tokens[widx]} {' '.join([wrap_color(str(pp), bcolor=('blue' if pp[0] is t0.sent.tokens[widx] else ('red' if pp[1] else 'black'))) for pp in zip(arr_toks[bidx], w_arr[bidx, widx-t0.widx+i0])])}")
            # --
        ret = BK.input_real(w_arr)
        return ret

    def get_sw_dep(self, sent, val_set):
        ret = np.zeros([len(sent), len(sent)], dtype=np.float32)  # [slen, slen]
        tree_dep = sent.tree_dep
        _heads, _labels = tree_dep.seq_head.vals, tree_dep.seq_label.vals
        for m, h in enumerate(_heads):
            if h>0:
                h -= 1
                _lab0 = _labels[m].split(":")[0]
                if "ch:" in val_set or f"ch:{_lab0}" in val_set:  # h->m
                    ret[h, m] = 1.
                if "par:" in val_set or f"par:{_lab0}" in val_set:  # m->h
                    ret[m, h] = 1.
        return ret

    def get_sw_srl(self, sent, val_set):
        ret = np.zeros([len(sent), len(sent)], dtype=np.float32)  # [slen, slen]
        arg_info = sent.info.get('args')
        if arg_info is not None:
            for widx, info in enumerate(arg_info):
                for info1 in info:  # there can be multiple ones
                    aidx, arole = info1[:2]
                    if arole in val_set:
                        ret[widx, aidx] = 1.
            # --
            # # for debugging
            # for evt in sent.events:
            #     widx = evt.mention.shead_widx
            #     _arr = ret[widx]
            #     print(f"{evt} {sent.tokens[widx]}: {[sent.tokens[i] for i,m in enumerate(_arr) if m]}")
            # --
        else:
            zwarn("No arg-info when using srl aug reprs!")
        return ret

# --
# b msp2/tasks/zmtl3/mod/extract/evt_trig:??
