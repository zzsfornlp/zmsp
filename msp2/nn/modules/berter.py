#

# pre-trained bert model

__all__ = [
    "BertEncoderConf", "BertEncoder", "BerterInputBatch",
]

from typing import Callable, List, Dict, Set, Tuple, Union
from collections import OrderedDict
import numpy as np
from msp2.nn import BK
from msp2.nn.layers import EmbeddingNode, VrecSteppingState, CombinerConf, CombinerNode, ModuleWrapper
from msp2.utils import zlog, zwarn
from msp2.data.inst import InputSubwordSeqField, DataPadder, Sent
from .base import BasicConf, BasicNode
from .berter_impl import BerterImpl

# BertEncoder
class BertEncoderConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        # basic
        self.bert_model = "bert-base-multilingual-cased"  # or "bert-base-cased", "bert-large-cased", "bert-base-chinese", ...
        self.bert_output_layers = [-1]  # which layers to extract for final output (0 means embedding-layer)
        self.bert_cache_dir = ""
        self.bert_ft = False  # whether fine-tuning (add to self?)
        # training
        self.bert_repl_mask_rate = 0.  # acts as dropout
        # final output
        self.bert_combiner = CombinerConf()

    @property
    def cache_dir_or_none(self):
        return self.bert_cache_dir if self.bert_cache_dir else None

class BertEncoder(BasicNode):
    def __init__(self, conf: BertEncoderConf, other_embed_nodes: Dict[str, EmbeddingNode]=None, other_embed_borrowing=True):
        super().__init__(conf)
        # --
        conf: BertEncoderConf = self.conf
        zlog(f"Loading pre-trained bert model for Berter2 of {self.extra_repr()}")
        # make a impl and add module
        self.impl = BerterImpl.create(conf.bert_model, cache_dir=conf.cache_dir_or_none)
        if conf.bert_ft:  # fine-tune it!
            self.add_module("M", ModuleWrapper(self.impl.model, None))  # reg it!
        zlog(f"Load ok, move to default device {BK.DEFAULT_DEVICE}")
        self.impl.model.to(BK.DEFAULT_DEVICE)
        # --
        # other embedding node (by default just borrowing)
        self.other_embed_nodes = other_embed_nodes
        if not other_embed_borrowing and other_embed_nodes is not None:  # include the modules
            for _k, _n in other_embed_nodes.items():
                self.add_module(f"_N{_k}", _n)
        # output combiner
        self.combiner = CombinerNode(conf.bert_combiner, isizes=[self.impl.hidden_size]*len(conf.bert_output_layers))
        # get max layer
        _nl = self.impl.num_hidden_layers  # 0 is emb layer!!
        self.actual_output_layers = [z if z>=0 else (_nl+1+z) for z in conf.bert_output_layers]

    # shortcuts
    @property
    def model(self):
        return self.impl.model

    @property
    def tokenizer(self):
        return self.impl.tokenizer

    @property
    def sub_toker(self):
        return self.impl.sub_toker

    # --
    def extra_repr(self) -> str:
        return f"BertEncoder({self.conf.bert_model})"

    def get_output_dims(self, *input_dims):
        return self.combiner.get_output_dims(*input_dims)

    def forward(self, inputs, vstate: VrecSteppingState=None, inc_cls=False):
        conf: BertEncoderConf = self.conf
        # --
        no_bert_ft = (not conf.bert_ft)  # whether fine-tune bert (if not detach hiddens!)
        impl = self.impl
        # --
        # prepare inputs
        if not isinstance(inputs, BerterInputBatch):
            inputs = self.create_input_batch(inputs)
        all_output_layers = []  # including embeddings
        # --
        # get embeddings (for embeddings, we simply forward once!)
        mask_repl_rate = conf.bert_repl_mask_rate if self.is_training() else 0.
        input_ids, input_masks = inputs.get_basic_inputs(mask_repl_rate)  # [bsize, 1+sub_len+1]
        other_embeds = None
        if self.other_embed_nodes is not None and len(self.other_embed_nodes)>0:
            other_embeds = 0.
            for other_name, other_node in self.other_embed_nodes.items():
                other_embeds += other_node(inputs.other_factors[other_name])  # should be prepared correspondingly!!
        # --
        # forward layers (for layers, we may need to split!)
        # todo(+N): we simply split things apart, thus middle parts may lack CLS/SEP, and not true global att
        # todo(+N): the lengths currently are hard-coded!!
        MAX_LEN = 512  # max len
        INBUF_LEN = 50  # in-between buffer for splits, for both sides!
        cur_sub_len = BK.get_shape(input_ids, 1)  # 1+sub_len+1
        needs_split = (cur_sub_len>MAX_LEN)
        if needs_split:  # decide split and merge points
            split_points = self._calculate_split_points(cur_sub_len, MAX_LEN, INBUF_LEN)
            zwarn(f"Multi-seg for Berter: {cur_sub_len}//{len(split_points)}->{split_points}")
        # --
        # todo(note): we also need split from embeddings
        if needs_split:
            all_embed_pieces = []
            split_extended_attention_mask = []
            for o_s, o_e, i_s, i_e in split_points:
                piece_embeddings, piece_extended_attention_mask = impl.forward_embedding(
                    *[(None if z is None else z[:, o_s:o_e]) for z in
                      [input_ids, input_masks, inputs.batched_token_type_ids, inputs.batched_position_ids, other_embeds]]
                )
                all_embed_pieces.append(piece_embeddings[:, i_s:i_e])
                split_extended_attention_mask.append(piece_extended_attention_mask)
            embeddings = BK.concat(all_embed_pieces, 1)  # concat back to full
            extended_attention_mask = None
        else:
            embeddings, extended_attention_mask = impl.forward_embedding(
                input_ids, input_masks, inputs.batched_token_type_ids, inputs.batched_position_ids, other_embeds)
            split_extended_attention_mask = None
        if no_bert_ft:  # stop gradient
            embeddings = embeddings.detach()
        # --
        cur_hidden = embeddings
        all_output_layers.append(embeddings)  # *[bsize, 1+sub_len+1, D]
        # also prepare mapper idxes for sub <-> orig
        # todo(+N): currently only use the first sub-word!
        idxes_arange2 = inputs.arange2_t  # [bsize, 1]
        batched_first_idxes_p1 = (1+inputs.batched_first_idxes) * (inputs.batched_first_mask.long())  # plus one for CLS offset!
        if inc_cls:  # [bsize, 1+orig_len]
            idxes_sub2orig = BK.concat([BK.constants_idx([inputs.bsize, 1], 0), batched_first_idxes_p1], 1)
        else:  # [bsize, orig_len]
            idxes_sub2orig = batched_first_idxes_p1
        _input_masks0 = None  # used for vstate back, make it 0. for BOS and EOS
        # for ii in range(impl.num_hidden_layers):
        for ii in range(max(self.actual_output_layers)):  # do not need that much if does not require!
            # forward multiple times with splitting if needed
            if needs_split:
                all_pieces = []
                for piece_idx, piece_points in enumerate(split_points):
                    o_s, o_e, i_s, i_e = piece_points
                    piece_res = impl.forward_hidden(ii, cur_hidden[:, o_s:o_e], split_extended_attention_mask[piece_idx])[:, i_s:i_e]
                    all_pieces.append(piece_res)
                new_hidden = BK.concat(all_pieces, 1)  # concat back to full
            else:
                new_hidden = impl.forward_hidden(ii, cur_hidden, extended_attention_mask)
            if no_bert_ft:  # stop gradient
                new_hidden = new_hidden.detach()
            if vstate is not None:
                # from 1+sub_len+1 -> (inc_cls?)+orig_len
                new_hidden2orig = new_hidden[idxes_arange2, idxes_sub2orig]  # [bsize, 1?+orig_len, D]
                # update
                new_hidden2orig_ret = vstate.update(new_hidden2orig)  # [bsize, 1?+orig_len, D]
                if new_hidden2orig_ret is not None:
                    # calculate when needed
                    if _input_masks0 is None:  # [bsize, 1+sub_len+1, 1] with 1. only for real valid ones
                        _input_masks0 = inputs._aug_ends(inputs.batched_input_mask, 0., 0., 0., BK.float32).unsqueeze(-1)
                    # back to 1+sub_len+1; todo(+N): here we simply add and //2, and no CLS back from orig to sub!!
                    tmp_orig2sub = new_hidden2orig_ret[idxes_arange2, int(inc_cls)+inputs.batched_rev_idxes]  # [bsize, sub_len, D]
                    tmp_slice_size = BK.get_shape(tmp_orig2sub)
                    tmp_slice_size[1] = 1
                    tmp_slice_zero = BK.zeros(tmp_slice_size)
                    tmp_orig2sub_aug = BK.concat([tmp_slice_zero, tmp_orig2sub, tmp_slice_zero], 1)  # [bsize, 1+sub_len+1, D]
                    new_hidden = new_hidden * (1.-_input_masks0) + ((new_hidden+tmp_orig2sub_aug)/2.) * _input_masks0
            all_output_layers.append(new_hidden)
            cur_hidden = new_hidden
        # finally, prepare return
        final_output_layers = [all_output_layers[z] for z in conf.bert_output_layers]  # *[bsize,1+sl+1,D]
        combined_output = self.combiner(final_output_layers)  # [bsize, 1+sl+1, ??]
        final_ret = combined_output[idxes_arange2, idxes_sub2orig]  # [bsize, 1?+orig_len, D]
        return final_ret

    def create_input_batch(self, seq_subs: List[InputSubwordSeqField]):
        return BerterInputBatch(self, seq_subs)

    def create_input_batch_from_sents(self, sents: List[Sent]):
        sub_toker = self.sub_toker
        seq_subs = [s.seq_word.get_subword_seq(sub_toker) for s in sents]
        return BerterInputBatch(self, seq_subs)

    def _calculate_split_points(self, len: int, MAX_LEN: int, INBUF_LEN: int):
        forward_pieces = []  # (out_start, out_end, in_start, in_end) -> [out_start:out_end][in_start:in_end]
        cur_p = 0
        cur_offset = 0
        while True:
            cur_p_plus = min(len, cur_p+MAX_LEN)
            if cur_p_plus >= len:
                # hit end
                forward_pieces.append((cur_p, cur_p_plus, cur_offset, None))  # no need to split any more
                break
            else:
                # not yet
                forward_pieces.append((cur_p, cur_p_plus, cur_offset, -INBUF_LEN))  # back INBUF_LEN
                cur_p = cur_p_plus - 2*INBUF_LEN  # one for each
                cur_offset = INBUF_LEN  # offset INBUF_LEN
        return forward_pieces

# Batched Input for Berter
# todo(+N): are these preparings fast on cpu or gpu? (currently mostly on gpu)
class BerterInputBatch:
    def __init__(self, berter: BertEncoder, seq_subs: List[InputSubwordSeqField]):
        self.seq_subs = seq_subs
        self.berter = berter
        self.bsize = len(seq_subs)
        self.arange1_t = BK.arange_idx(self.bsize)  # [bsize]
        self.arange2_t = self.arange1_t.unsqueeze(-1)  # [bsize, 1]
        self.arange3_t = self.arange2_t.unsqueeze(-1)  # [bsize, 1, 1]
        # --
        tokenizer = self.berter.tokenizer
        PAD_IDX = tokenizer.pad_token_id
        # MASK_IDX = tokenizer.mask_token_id
        # CLS_IDX_l = [tokenizer.cls_token_id]
        # SEP_IDX_l = [tokenizer.sep_token_id]
        # make batched idxes
        padder = DataPadder(2, pad_vals=PAD_IDX, mask_range=2)
        batched_sublens = [len(s.idxes) for s in seq_subs]  # [bsize]
        batched_input_ids, batched_input_mask = padder.pad([s.idxes for s in seq_subs])  # [bsize, sub_len]
        self.batched_sublens_p1 = BK.input_idx(batched_sublens) + 1  # also the idx of EOS (if counting including BOS)
        self.batched_input_ids = BK.input_idx(batched_input_ids)
        self.batched_input_mask = BK.input_real(batched_input_mask)
        # make batched mappings (sub->orig)
        padder2 = DataPadder(2, pad_vals=0, mask_range=2)  # pad as 0 to avoid out-of-range
        batched_first_idxes, batched_first_mask = padder2.pad([s.align_info.orig2begin for s in seq_subs])  # [bsize, orig_len]
        self.batched_first_idxes = BK.input_idx(batched_first_idxes)
        self.batched_first_mask = BK.input_real(batched_first_mask)
        # reversed batched_mappings (orig->sub) (created when needed)
        self._batched_rev_idxes = None  # [bsize, sub_len]
        # --
        self.batched_repl_masks = None  # [bsize, sub_len], to replace with MASK
        self.batched_token_type_ids = None  # [bsize, 1+sub_len+1]
        self.batched_position_ids = None  # [bsize, 1+sub_len+1]
        self.other_factors = {}  # name -> aug_batched_ids

    @property
    def batched_rev_idxes(self):
        if self._batched_rev_idxes is None:
            padder = DataPadder(2, pad_vals=0)  # again pad 0
            batched_rev_idxes, _ = padder.pad([s.align_info.split2orig for s in self.seq_subs])  # [bsize, sub_len]
            self._batched_rev_idxes = BK.input_idx(batched_rev_idxes)
        return self._batched_rev_idxes  # [bsize, sub_len]

    # common
    def _transform_factors(self, factors: Union[List[List[int]], BK.Expr], is_orig: bool, PAD_IDX: Union[int, float]):
        if isinstance(factors, BK.Expr):  # already padded
            batched_ids = factors
        else:
            padder = DataPadder(2, pad_vals=PAD_IDX)
            batched_ids, _ = padder.pad(factors)
            batched_ids = BK.input_idx(batched_ids)  # [bsize, orig-len if is_orig else sub_len]
        if is_orig:  # map to subtoks
            final_batched_ids = batched_ids[self.arange2_t, self.batched_rev_idxes]  # [bsize, sub_len]
        else:
            final_batched_ids = batched_ids  # [bsize, sub_len]
        return final_batched_ids

    def _aug_ends(self, t: BK.Expr, BOS, PAD, EOS, dtype):  # add BOS(CLS) and EOS(SEP) for a tensor (sub_len -> 1+sub_len+1)
        slice_shape = [self.bsize, 1]
        slices = [BK.constants(slice_shape, BOS, dtype=dtype), t, BK.constants(slice_shape, PAD, dtype=dtype)]
        aug_batched_ids = BK.concat(slices, -1)  # [bsize, 1+sub_len+1]
        aug_batched_ids[self.arange1_t, self.batched_sublens_p1] = EOS  # assign EOS
        return aug_batched_ids

    def _prepare_factors(self, factors: Union[List[List[int]], BK.Expr], is_orig: bool, voc, dtype=BK.long):  # prepare full one!
        PAD_IDX = voc.pad
        CLS_IDX = voc.bos
        SEP_IDX = voc.eos
        final_batched_ids = self._transform_factors(factors, is_orig, PAD_IDX)
        aug_batched_ids = self._aug_ends(final_batched_ids, CLS_IDX, PAD_IDX, SEP_IDX, dtype)
        return aug_batched_ids

    def set_factors(self, name: str, factors: Union[List[List[int]], BK.Expr], is_orig: bool, voc):
        self.other_factors[name] = self._prepare_factors(factors, is_orig, voc)

    def set_token_type_ids(self, factors: Union[List[List[int]], BK.Expr], is_orig: bool, voc):
        self.batched_token_type_ids = self._prepare_factors(factors, is_orig, voc)

    def set_position_ids(self, factors: Union[List[List[int]], BK.Expr], is_orig: bool, voc):
        self.batched_position_ids = self._prepare_factors(factors, is_orig, voc)

    def set_repl_masks(self, repl_masks: Union[List[List[float]], BK.Expr], is_orig: bool):
        # todo(note): repl_masks does not add 1+?+1 here!!
        self.batched_repl_masks = self._transform_factors(repl_masks, is_orig, 0.)  # pad 0.

    # dynamically create input_ids and input_masks
    def get_basic_inputs(self, repl_mask_rate: float):
        tokenizer = self.berter.tokenizer
        PAD_IDX = tokenizer.pad_token_id
        MASK_IDX = tokenizer.mask_token_id
        CLS_IDX = tokenizer.cls_token_id
        SEP_IDX = tokenizer.sep_token_id
        # --
        input_ids = self.batched_input_ids  # [bsize, sub_len]
        repl_masks = self.batched_repl_masks  # [bsize, sub_len]
        if repl_mask_rate > 0.:  # extra mask
            extra_repl_mask = (BK.rand(input_ids.shape) < repl_mask_rate).float()
            if repl_masks is None:
                repl_masks = extra_repl_mask
            else:
                repl_masks *= extra_repl_mask
        if repl_masks is not None:
            repl_masks = repl_masks.long()
            input_ids = input_ids * (1-repl_masks) + MASK_IDX * repl_masks  # replace the ones with [MASK]
        # aug
        ret_input_ids = self._aug_ends(input_ids, CLS_IDX, PAD_IDX, SEP_IDX, BK.long)
        ret_input_masks = self._aug_ends(self.batched_input_mask, 1., 0., 1., BK.float32)
        return ret_input_ids, ret_input_masks
