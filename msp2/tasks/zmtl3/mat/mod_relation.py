#

# relation extraction

__all__ = [
    "ZTaskMatRConf", "ZTaskMatR", "ZmodeMatRConf", "ZmodMatR",
]

from typing import List
import numpy as np
from msp2.data.inst import yield_sents, yield_frames, set_ee_heads, DataPadder, SimpleSpanExtender, Sent
from msp2.data.vocab import SimpleVocab, SeqVocab, SeqVocabConf
from msp2.utils import zlog, zwarn, ZRuleFilter, Random, StrHelper, F1EvalEntry, MathHelper, wrap_color
from msp2.proc import FrameEvalConf, FrameEvaler, ResultRecord
from msp2.nn import BK
from msp2.nn.l3 import *
from msp2.tasks.zmtl3.mod.extract.base import *

class ZTaskMatRConf(ZTaskBaseEConf):
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name if name is not None else 'matr'
        self.matr_conf = ZmodeMatRConf()
        self.matr_eval = FrameEvalConf.direct_conf(
            match_arg_with_frame_type=False, weight_frame=0., weight_arg=1., bd_arg='', bd_arg_lines=50)
        # for strict match: "matr.labf_arg:(lambda x: (x.label, x.main.label, x.arg.label))"
        # --

    def build_task(self):
        return ZTaskMatR(self)

class ZTaskMatR(ZTaskBaseE):
    def __init__(self, conf: ZTaskMatRConf):
        super().__init__(conf)
        conf: ZTaskMatRConf = self.conf
        self.evaler = FrameEvaler(conf.matr_eval)
        # --

    # build vocab
    def build_vocab(self, datasets: List):
        # note: still build a vocab that covers all
        voc_evt = SimpleVocab.build_empty(f"vocE_{self.name}")
        voc_rel = SimpleVocab.build_empty(f"vocR_{self.name}")
        for dataset in datasets:
            if dataset.wset == 'train':
                for evt in yield_frames(dataset.insts):
                    voc_evt.feed_one(evt.label)
                    for arg in evt.args:
                        voc_rel.feed_one(arg.label)
        voc_evt.build_sort()
        voc_rel.build_sort()
        zlog(f"Finish building for: {voc_evt} & {voc_rel}")
        return (voc_evt, voc_rel)

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
        return self.conf.matr_conf.make_node(self, model)

# --

class ZmodeMatRConf(ZModBaseEConf):
    def __init__(self):
        super().__init__()
        # --
        self.max_content_len = 128  # maximum seq length
        self.ctx_nsent_rates = [1.]  # extend context by sent, how many more before & after?
        self.loss_lab = 1.  # loss weight
        self.neg_rate = 2.
        # --
        self.mix_emb_initscale = 0.05  # init scale
        self.rel_scorer = MlpConf.direct_conf(n_hid_layer=1)
        # --
        self.ctx_as_trg = True  # whether use ctx sents as targets
        self.mark_strategy = ""  # mark what?
        self.mark_diff2 = True  # differ for the start and end?
        self.mark_entity = True
        # --

@node_reg(ZmodeMatRConf)
class ZmodMatR(ZModBaseE):
    def __init__(self, conf: ZmodeMatRConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZmodeMatRConf = self.conf
        self.voc_evt, self.voc_rel = ztask.vpack
        # --
        # ctx-sent rates
        self.ctx_nsent_rates = [z / sum(conf.ctx_nsent_rates) for z in conf.ctx_nsent_rates]
        self.test_ctx_nsent = [i for i, z in enumerate(self.ctx_nsent_rates) if z > 0][-1]
        zlog(f"ctx_nsent: train={self.ctx_nsent_rates}, test={self.test_ctx_nsent}")
        # --
        # marker embeddings: 0=nope, 1/2=S, 3/4=O
        self.mark_s, self.mark_o = ('s' in conf.mark_strategy), ('o' in conf.mark_strategy)
        _mdim = self.bmod.get_mdim()
        self.emb_ent = BK.get_emb_with_initscale(len(self.voc_evt), _mdim, conf.mix_emb_initscale)  # entity
        self.emb_mark = BK.get_emb_with_initscale(5, _mdim, conf.mix_emb_initscale)  # marker
        self.rel_scorer = MlpLayer(conf.rel_scorer, isize=[_mdim]*4, osize=len(self.voc_rel))  # simply MLP scorer
        # --

    def do_loss(self, rc: ZRunCache, *args, **kwargs): return self.do_forward(rc.ibatch, is_testing=False)
    def do_predict(self, rc: ZRunCache, *args, **kwargs): return self.do_forward(rc.ibatch, is_testing=True)

    def do_forward(self, ibatch, is_testing: bool):
        conf: ZmodeMatRConf = self.conf
        # --
        all_sents, all_contents = self.prepare_ibatch(ibatch, clear_args=is_testing, is_testing=is_testing)
        t_ids, t_etypes, arr_items_s, t_swidxes_s, arr_items_o, t_swidxes_o, t_pair_mask, t_rel_lab \
            = self.prepare_batch(all_contents, is_testing)
        _is_zero_s, _is_zero_o = BK.is_zero_shape(t_swidxes_s), BK.is_zero_shape(t_swidxes_o)
        # forward
        _pad_id = self.tokenizer.pad_token_id
        _ids_mask_t = (t_ids != _pad_id).float()
        _ts_mix = []
        if conf.mark_entity:
            _ts_mix.append(self.emb_ent(t_etypes))
        if self.mark_s or self.mark_o:
            _t_marks = 0 * t_etypes
            _diff2 = int(bool(conf.mark_diff2))
            # todo(+N): we can have other schemes
            if self.mark_s and not _is_zero_s:
                _t_marks.scatter_(-1, t_swidxes_s[..., 0], 1)
                _t_marks.scatter_(-1, t_swidxes_s[..., 1], 1+_diff2)
            if self.mark_o and not _is_zero_o:
                _t_marks.scatter_(-1, t_swidxes_o[..., 0], 3)
                _t_marks.scatter_(-1, t_swidxes_o[..., 1], 3+_diff2)
            _t_marks[:, 0] = 0  # clear for pad
            _ts_mix.append(self.emb_mark(_t_marks))
        if len(_ts_mix) > 0:
            mixes = [(BK.input_real(0.5), BK.stack(_ts_mix, 0).mean(0))]  # first mix these!
        else:
            mixes = None
        # --
        try:
            bert_out = self.bmod.forward_enc(t_ids, _ids_mask_t, mixes)
            bert_hid = bert_out.hidden_states[-1]  # [*, L, D]
        except RuntimeError as e:
            if is_testing:
                _bsize = len(all_sents)
                _hids = []
                for _bstart in range(0, len(t_ids), _bsize):
                    _bend = _bstart + _bsize
                    _out = self.bmod.forward_enc(t_ids[_bstart:_bend], _ids_mask_t[_bstart:_bend],
                                                 [(a, b[_bstart:_bend]) for a,b in mixes])
                    _hids.append(_out.hidden_states[-1])
                bert_hid = BK.concat(_hids, 0)  # [*, L, D]
            else:
                raise e
        _arange_t = BK.arange_idx(len(bert_hid)).unsqueeze(-1)  # [*, 1]
        if _is_zero_s or _is_zero_o:
            t_score = BK.zeros([len(bert_hid), arr_items_s.shape[-1], arr_items_o.shape[-1], len(self.voc_rel)])
        else:
            t0_s, t0_o = bert_hid[_arange_t, t_swidxes_s[..., 0]], bert_hid[_arange_t, t_swidxes_o[..., 0]]  # *[*, N?, D]
            t1_s, t1_o = bert_hid[_arange_t, t_swidxes_s[..., 1]], bert_hid[_arange_t, t_swidxes_o[..., 1]]  # *[*, N?, D]
            t_score = self.rel_scorer([unsqueeze_expand(t0_s, -2, arr_items_o.shape[-1]),
                                       unsqueeze_expand(t1_s, -2, arr_items_o.shape[-1]),
                                       unsqueeze_expand(t0_o, -3, arr_items_s.shape[-1]),
                                       unsqueeze_expand(t1_o, -3, arr_items_s.shape[-1])])  # [*, NS, NO, C]
        if not is_testing:  # getting loss
            t_loss0 = BK.loss_nll(t_score, t_rel_lab)  # [*, NS, NO]
            loss_item = LossHelper.compile_leaf_loss(
                'matR', (t_loss0 * t_pair_mask).sum(), t_pair_mask.sum(), loss_lambda=conf.loss_lab)
            ret = LossHelper.compile_component_loss(self.name, [loss_item])
            return ret, {}
        else:  # decode
            if not BK.is_zero_shape(t_score):
                arr_score, arr_lab = [BK.get_value(z) for z in t_score.log_softmax(-1).max(-1)]  # [*, NS, NO]
                for _bidx, (_items_s, _items_o) in enumerate(zip(arr_items_s, arr_items_o)):
                    for _iis, _item_s in enumerate(_items_s):
                        if _item_s is None: continue
                        for _iio, _item_o in enumerate(_items_o):
                            if _item_o is None: continue
                            _score, _lab = float(arr_score[_bidx, _iis, _iio]), int(arr_lab[_bidx, _iis, _iio])
                            if _lab > 0:  # predicted!
                                _item_s.add_arg(_item_o, role=self.voc_rel.idx2word(_lab), score=_score)
            return {}
        # --

    def prepare_batch(self, all_contents, is_testing: bool):
        conf: ZmodeMatRConf = self.conf
        # --
        # extend them by marking strategy
        _contents = [[z[i] for z in all_contents] for i in range(6)]
        for mark_ii, mark_flag in zip([2, 4], [self.mark_s, self.mark_o]):
            if mark_flag:
                _other_fields = sorted(set(range(6)).difference([mark_ii, mark_ii+1]))
                _new_contents = [[] for _ in range(6)]
                for _bidx in range(len(_contents[mark_ii])):
                    for _item, _swidxes in zip(_contents[mark_ii][_bidx], _contents[mark_ii+1][_bidx]):
                        for _fii in _other_fields:
                            _new_contents[_fii].append(_contents[_fii][_bidx])
                        _new_contents[mark_ii].append([_item])  # only add this one!
                        _new_contents[mark_ii+1].append([_swidxes])
                _contents = _new_contents  # replace!
        # --
        # batch
        flat_ids, flat_etypes, items_s, swidxes_s, items_o, swidxes_o = _contents
        _pad_id = self.tokenizer.pad_token_id
        _padder_3d = DataPadder(3)
        # --
        t_ids = BK.input_idx(DataPadder.go_batch_2d(flat_ids, _pad_id))  # [*, L]
        t_etypes = BK.input_idx(DataPadder.go_batch_2d(flat_etypes, 0))  # [*, L]
        arr_items_s = DataPadder.go_batch_2d(items_s, None, dtype=object)  # [*, NS]
        t_swidxes_s = BK.input_idx(_padder_3d.pad(swidxes_s)[0])  # [*, NS, 2]
        arr_items_o = DataPadder.go_batch_2d(items_o, None, dtype=object)  # [*, NO]
        t_swidxes_o = BK.input_idx(_padder_3d.pad(swidxes_o)[0])  # [*, NO, 2]
        # [*, NS, NO]
        t_pair_mask = BK.input_real(arr_items_s!=None).unsqueeze(-1) * BK.input_real(arr_items_o!=None).unsqueeze(-2)
        if is_testing:
            t_rel_lab = None
        else:
            arr_rel_lab = np.full(list(arr_items_s.shape) + [arr_items_o.shape[-1]], fill_value=0, dtype=np.int)
            for _bidx, (_items_s, _items_o) in enumerate(zip(arr_items_s, arr_items_o)):
                for _iis, _item_s in enumerate(_items_s):
                    if _item_s is None: continue
                    _m_lab = {id(a.arg): self.voc_rel.get_else_unk(a.label) for a in _item_s.args}
                    arr_rel_lab[_bidx, _iis] = [_m_lab.get(id(z), 0) for z in _items_o]
            t_rel_lab = BK.input_idx(arr_rel_lab)  # [*, NS, NO]
            # --
            # down sampling neg
            _down_mask = down_neg(t_pair_mask, (t_rel_lab > 0), conf.neg_rate, do_sample=True)  # [*, NS, NO]
            _down_b = (_down_mask.sum(-1).sum(-1) > 0.)  # [*]
            if (_down_b <= 0).any():  # filter
                t_ids, t_etypes, t_swidxes_s, t_swidxes_o, t_pair_mask, t_rel_lab = \
                    [z[_down_b] for z in [t_ids, t_etypes, t_swidxes_s, t_swidxes_o, _down_mask, t_rel_lab]]
                _arr_down_b = BK.get_value(_down_b)
                arr_items_s, arr_items_o = arr_items_s[_arr_down_b], arr_items_o[_arr_down_b]
            else:
                t_pair_mask = _down_mask
        # --
        return t_ids, t_etypes, arr_items_s, t_swidxes_s, arr_items_o, t_swidxes_o, t_pair_mask, t_rel_lab

    def prepare_ibatch(self, ibatch, clear_args=False, is_testing=False, no_cache=False):
        _key = f"_cache_{self.name}"
        conf: ZmodeMatRConf = self.conf
        # --
        if clear_args:
            for item in ibatch.items:
                for evt in item.sent.events:
                    evt.clear_args()
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
        return all_sents, all_contents

    def prepare_content(self, item, ctx_nsent: int):
        conf: ZmodeMatRConf = self.conf
        # --
        # prepare for one item
        sent = item.sent
        center_ids, center_toks = self.prep_sent(sent, ret_toks=True)  # get subtokens, [olen], List[List]
        before_ids, after_ids, before_toks, after_toks = self.extend_ctx_sent(sent, ctx_nsent, ret_toks=True)
        seq_ids, seq_toks = before_ids + center_ids + after_ids, before_toks + center_toks + after_toks
        seq_sents = []
        for _tok in seq_toks:
            if _tok.sent not in seq_sents:
                seq_sents.append(_tok.sent)
        # --
        # flatten
        flat_ids = []
        _tmap = {}  # id(tok) -> (offset, swlen)
        for _ids, _tok in zip(seq_ids, seq_toks):
            _tmap[id(_tok)] = (len(flat_ids), len(_ids))
            flat_ids.extend(_ids)
        # collect items
        flat_etypes = [0] * len(flat_ids)  # item types
        items_s, items_o = [], []  # items as s&o
        swidxes_s, swidxes_o = [], []  # sub-word idxes in the flat-seq
        for one_sent in seq_sents:
            for one_evt in one_sent.events:
                one_tokens = one_evt.mention.get_tokens()
                _posis = [_tmap[id(z)] for z in one_tokens]
                _il, _ir = _posis[0][0], sum(_posis[-1])-1  # left, right
                # add
                _type = self.voc_evt.get_else_unk(one_evt.label)
                flat_etypes[_il:_ir+1] = [_type] * (_ir-_il+1)
                items_o.append(one_evt)
                swidxes_o.append((_il, _ir))
                if one_sent is sent:  # only add s if as center sentence!
                    items_s.append(one_evt)
                    swidxes_s.append((_il, _ir))
        # --
        # check max_len
        if len(flat_ids) > conf.max_content_len:
            _before_ratio = 0.65
            _cel, _cer = _tmap[id(sent.get_tokens()[0])][0], sum(_tmap[id(sent.get_tokens()[-1])])  # center
            _before_budget = int((conf.max_content_len - (_cer-_cel+1)) * _before_ratio)
            _tstart = max(0, _cel - max(0, _before_budget))  # truncate
            _tend = min(_tstart+conf.max_content_len, len(flat_ids))
            # modify
            flat_ids = flat_ids[_tstart:_tend]
            flat_etypes = flat_etypes[_tstart:_tend]
            items_s, swidxes_s, dis_s = self.filter_with_truncate(items_s, swidxes_s, _tstart, _tend)
            items_o, swidxes_o, dis_o = self.filter_with_truncate(items_o, swidxes_o, _tstart, _tend)
            if len(dis_s) > 0 or len(dis_o) > 0:
                zwarn(f"Loss of mentions due to truncation: {dis_s} // {dis_o}")
        else:
            _tstart, _tend = 0, len(flat_ids)
        # --
        def _adjust(_x):  # adjust idx
            return min(max(_tstart, _x), _tend-1) - _tstart + 1
        # --
        # add [cls] and [sep]!
        flat_ids = [self.tokenizer.cls_token_id] + flat_ids + [self.tokenizer.sep_token_id]
        flat_etypes = [0] + flat_etypes + [0]
        swidxes_s = [(_adjust(a), _adjust(b)) for a,b in swidxes_s]  # change idx!
        swidxes_o = [(_adjust(a), _adjust(b)) for a,b in swidxes_o]
        return (flat_ids, flat_etypes, items_s, swidxes_s, items_o, swidxes_o)

    # --
    # helpers
    def filter_with_truncate(self, items, swidxes, t_start: int, t_end: int):
        ret_items, ret_swidxes = [], []
        discarded_items = []
        for item, (il, ir) in zip(items, swidxes):
            if il < t_start or ir >= t_end:  # note: delete it even if partial!
                discarded_items.append(item)
            else:
                ret_items.append(item)
                ret_swidxes.append((il, ir))
        return ret_items, ret_swidxes, discarded_items
    # --

# --
# b msp2/tasks/zmtl3/mat/mod_relation:??
