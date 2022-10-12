#

# mention extraction

__all__ = [
    "ZTaskMatMConf", "ZTaskMatM", "ZmodeMatMConf", "ZmodMatM",
]

import os
from typing import List

import numpy as np

from msp2.data.inst import yield_sents, yield_frames, set_ee_heads, DataPadder, SimpleSpanExtender, Sent
from msp2.data.vocab import SimpleVocab, SeqVocab, SeqVocabConf
from msp2.utils import zlog, zwarn, ZRuleFilter, Random, StrHelper, F1EvalEntry, zglob1z
from msp2.proc import FrameEvalConf, FrameEvaler, ResultRecord
from msp2.nn import BK
from msp2.nn.l3 import *
from msp2.tasks.zmtl3.mod.extract.base import *
from .mod_common import *
# --
# note: reuse some of zmtl2 (these do not have Dropout nodes, thus should be fine ...)
from msp2.tasks.zmtl2.zmod.common.lab import *
# --

class ZTaskMatMConf(ZTaskBaseEConf):
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name if name is not None else 'matm'
        self.matm_conf = ZmodeMatMConf()
        self.matm_eval = FrameEvalConf.direct_conf(weight_frame=1., weight_arg=0., bd_frame='', bd_frame_lines=50)

    def build_task(self):
        return ZTaskMatM(self)

class ZTaskMatM(ZTaskBaseE):
    def __init__(self, conf: ZTaskMatMConf):
        super().__init__(conf)
        conf: ZTaskMatMConf = self.conf
        self.evaler = FrameEvaler(conf.matm_eval)
        # --

    # build vocab
    def build_vocab(self, datasets: List):
        # note: still build a vocab that covers all
        voc_evt = SimpleVocab.build_empty(f"voc_{self.name}", post_list=[])
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
        return self.conf.matm_conf.make_node(self, model)

# --

class ZmodeMatMConf(ZModBaseEConf):
    def __init__(self):
        super().__init__()
        # --
        self.max_content_len = 128  # maximum seq length
        self.ctx_nsent_rates = [1.]  # extend context by sent, how many more before & after?
        self.loss_lab = 1.  # loss weight
        # --
        self.seqvoc_conf = SeqVocabConf()
        self.repr_conf = ZReprConf()
        self.lab_conf = ZLabelConf()
        # --
        # special mode: use elmo instead of bert!
        self.elmo_dir = ""
        # --

@node_reg(ZmodeMatMConf)
class ZmodMatM(ZModBaseE):
    def __init__(self, conf: ZmodeMatMConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZmodeMatMConf = self.conf
        # make extended vocab
        self.base_voc = ztask.vpack[0]
        self.seq_voc = SeqVocab(self.base_voc, conf=conf.seqvoc_conf)
        zlog(f"Build seq_voc from {self.base_voc} => {self.seq_voc}")
        # --
        # ctx-sent rates
        self.ctx_nsent_rates = [z/sum(conf.ctx_nsent_rates) for z in conf.ctx_nsent_rates]
        self.test_ctx_nsent = [i for i,z in enumerate(self.ctx_nsent_rates) if z>0][-1]
        zlog(f"ctx_nsent: train={self.ctx_nsent_rates}, test={self.test_ctx_nsent}")
        # --
        _cdim = len(self.seq_voc)
        self.repr = conf.repr_conf.make_node(base_layer=self)
        if conf.elmo_dir:
            from allennlp.modules.elmo import Elmo
            elmo_dir = zglob1z(conf.elmo_dir)
            _ELMO_LAYER = 3
            self.elmo = Elmo(os.path.join(elmo_dir, "elmo_options.json"), os.path.join(elmo_dir, "elmo_weights.hdf5"), _ELMO_LAYER)
            _odim = self.elmo.get_output_dim() * _ELMO_LAYER
            zlog(f"Load elmo from {elmo_dir}!")
        else:
            self.elmo = None
            _odim = self.repr.get_output_dim()
        self.unary_scorer = AffineLayer(None, isize=_odim, osize=_cdim)  # scorer
        self.lab = ZlabelNode(conf.lab_conf, _csize=_cdim)  # final output lab layer
        # --

    def do_loss(self, rc: ZRunCache, *args, **kwargs): return self.do_forward(rc.ibatch, is_testing=False)
    def do_predict(self, rc: ZRunCache, *args, **kwargs): return self.do_forward(rc.ibatch, is_testing=True)

    def do_forward(self, ibatch, is_testing: bool):
        conf: ZmodeMatMConf = self.conf
        # --
        # prepare and batch
        _pad_id = self.tokenizer.pad_token_id
        all_sents, all_contents, all_ts = self.prepare_ibatch(ibatch, clear_args=is_testing, is_testing=is_testing)
        # forward
        if self.elmo is not None:  # forward with elmo!
            from allennlp.modules.elmo import batch_to_ids
            arr_toks = all_ts[2]
            elmo_toks = []
            elmo_idxes = np.full(arr_toks.shape, 0, dtype=np.int)  # [*, L0]
            for one_bb, one_toks in enumerate(arr_toks):
                one_elmo_toks = []
                for one_ii, one_tok in enumerate(one_toks):
                    elmo_idxes[one_bb, one_ii] = len(one_elmo_toks)  # idx to next
                    if one_tok is not None:
                        one_elmo_toks.append(one_tok.word)  # add word form
                elmo_toks.append(one_elmo_toks)
            elmo_res = self.elmo(batch_to_ids(elmo_toks).to(BK.DEFAULT_DEVICE))
            elmo_out = BK.concat(elmo_res['elmo_representations'], -1)  # [*, L?, D?]
            t_content = elmo_out[BK.arange_idx(len(elmo_idxes)).unsqueeze(-1),
                                 BK.input_idx(elmo_idxes).clamp(max=(BK.get_shape(elmo_out, 1)-1))]  # [?bs, Lf1, D?]
        else:
            _ids_t = all_ts[0]  # [?bs, Lf0]
            _ids_mask_t = (_ids_t != _pad_id).float()
            bert_out = self.bmod.forward_enc(_ids_t, _ids_mask_t)
            t_content = self.repr(bert_out, all_ts[1])  # [?bs, Lf1, D], cand reprs
        t_score = self.unary_scorer(t_content)  # [?bs, Lf1, C]
        t_vcand, t_lab = all_ts[3], all_ts[4]
        if is_testing:
            t_score_final = self.lab.score_labels([t_score], t_vcand)  # [?bs, Lf1, C]
            t_best_labs = t_score_final.argmax(-1) * (t_vcand>0).long()  # make sure nil0 for invalid ones!
            arr_best_labs = BK.get_value(t_best_labs)
            for _sidx, _labs in enumerate(arr_best_labs):
                _toks = all_contents[_sidx][2]
                _spans = self.seq_voc.tags2spans_idx(_labs.tolist())  # [start, length, orig_idx]
                for _start, _len, _orig_lab in _spans:
                    _valid_toks = [_toks[z] for z in range(_start, _start+_len) if _toks[z] is not None]
                    if len(_valid_toks) > 0:
                        _valid_toks = [z for z in _valid_toks if z.sent is _valid_toks[0].sent]
                        _new_widx, _new_wlen = _valid_toks[0].widx, _valid_toks[-1].widx-_valid_toks[0].widx+1
                        _new_item = _valid_toks[0].sent.make_event(_new_widx, _new_wlen, type=self.base_voc.idx2word(_orig_lab))
            return {}
        else:
            t_loss, t_loss_mask = self.lab.gather_losses([t_score], t_lab, t_vcand)[0]
            loss_item = LossHelper.compile_leaf_loss(
                'matM', (t_loss * t_loss_mask).sum(), t_loss_mask.sum(), loss_lambda=conf.loss_lab)
            ret = LossHelper.compile_component_loss(self.name, [loss_item])
            return ret, {}
        # --

    def prepare_ibatch(self, ibatch, clear_args=False, is_testing=False, no_cache=False):
        _key = f"_cache_{self.name}"
        conf: ZmodeMatMConf = self.conf
        # --
        if clear_args:
            for item in ibatch.items:
                item.sent.clear_events()
                item.sent.clear_entity_fillers()
        # --
        # collect them!
        _gen0 = Random.get_generator("train")
        all_sents, all_contents = [], []
        for item in ibatch.items:
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
            # --
            all_sents.append(item.sent)
            all_contents.append(content_cache)
        # --
        # batch!
        _pad_id = self.tokenizer.pad_token_id
        t_ids = BK.input_idx(DataPadder.go_batch_2d([sum(z[0], []) for z in all_contents], _pad_id))  # [*, Lf0]
        t_sublens = BK.input_idx(DataPadder.go_batch_2d([z[1] for z in all_contents], 0))  # [*, Lf1]
        arr_toks = DataPadder.go_batch_2d([z[2] for z in all_contents], None, dtype=object)
        t_vcands = BK.input_idx(DataPadder.go_batch_2d([z[3] for z in all_contents], 0.))
        t_labels = BK.input_idx(DataPadder.go_batch_2d([z[4] for z in all_contents], 0))
        all_ts = [t_ids, t_sublens, arr_toks, t_vcands, t_labels]
        # --
        return all_sents, all_contents, all_ts

    def prepare_content(self, item, ctx_nsent: int):
        conf: ZmodeMatMConf = self.conf
        # --
        # prepare for one item
        sent = item.sent
        center_ids, center_toks = self.prep_sent(sent, ret_toks=True)  # get subtokens, [olen], List[List]
        before_ids, after_ids, before_toks, after_toks = self.extend_ctx_sent(sent, ctx_nsent, ret_toks=True)
        seq_ids, seq_toks = before_ids + center_ids + after_ids, before_toks + center_toks + after_toks
        _len = len(seq_ids)
        # ctx is not targets!
        seq_vcands = [0.] * len(before_ids) + [1.] * len(center_ids) + [0.] * len(after_ids)
        # put labels
        tok_map = {id(t): i for i, t in enumerate(seq_toks)}
        seq_labels = [0] * _len
        for evt in sent.events:  # note: as events!
            _mention_toks = evt.mention.get_tokens()
            _orig_idx = self.base_voc.get(evt.label)
            _seq_idxes = self.seq_voc.output_span_idx(len(_mention_toks), _orig_idx)
            for _ii, _tt in enumerate(_mention_toks):
                seq_labels[tok_map[id(_tt)]] = _seq_idxes[_ii]  # must be there!
        # -- debug
        # breakpoint()
        # zz = self.seq_voc.seq_idx2word(seq_labels)
        # --
        seq_sub_lens = [len(z) for z in seq_ids]
        # check max_len
        rets = [seq_ids, seq_sub_lens, seq_toks, seq_vcands, seq_labels]
        if sum(seq_sub_lens) > conf.max_content_len:  # truncate things!
            t_start, t_end = self.truncate_subseq(seq_sub_lens, len(before_ids), conf.max_content_len, silent=True)
            rets = [z[t_start:t_end] for z in rets]
            # warning if truncate out args
            del_args = [z for z in seq_labels[:t_start]+seq_labels[t_end:] if z>0]
            if len(del_args):
                zwarn(f"Loss of mentions due to truncation: {seq_labels[:t_start]} ... {seq_labels[t_end:]}")
        # --
        # add [cls] and [sep]!
        _cls, _sep = [self.tokenizer.cls_token_id], [self.tokenizer.sep_token_id]
        for ii, (vv0, vv1) in enumerate(zip([_cls, 1, None, 0., 0], [_sep, 1, None, 0., 0])):
            rets[ii] = [vv0] + rets[ii] + [vv1]
        return tuple(rets)

# --
# b msp2/tasks/zmtl3/mat/mod_mention:??
