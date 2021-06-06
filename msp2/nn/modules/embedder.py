#

__all__ = [
    "InputHelper", "PlainInputHelper", "CharInputHelper", "PosiInputHelper", "BertInputHelper", "InputterGroup",
    "PlainInputEmbedderConf", "PlainInputEmbedderNode", "CharCnnInputEmbedderConf", "CharCnnInputEmbedderNode",
    "PosiInputEmbedderConf", "PosiInputEmbedderNode", "BertInputEmbedderConf", "BertInputEmbedderNode",
    "EmbedderGroupConf", "EmbedderGroup",
]

from typing import Callable, List, Dict, Set, Tuple
from collections import OrderedDict
import numpy as np
from msp2.nn import BK
from msp2.nn.layers.base import *
from msp2.nn.layers import EmbeddingConf, EmbeddingNode, CnnConf, CnnNode, PosiEmbeddingConf, PosiEmbeddingNode
from msp2.data.inst import Sent, DataPadder
from msp2.data.vocab import SimpleVocab
from msp2.utils import zlog, zwarn
from .berter import BertEncoder, BerterInputBatch

# here are just the pieces, need a larger Node to pack them all together

# =====
# InputHelper: preparing the inputs

# basic plain input helper
class InputHelper:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def prepare(self, insts: List): raise NotImplementedError()
    def mask(self, vs, erase_masks): raise NotImplementedError()

    # batch inputs from List[inst]
    @staticmethod
    def prepare_batch(insts: List, idx_f: Callable, padder: DataPadder):
        cur_input_list = [idx_f(z) for z in insts]  # idx List
        cur_input_arr, _ = padder.pad(cur_input_list)
        ret_expr = BK.input_idx(cur_input_arr)
        return ret_expr

    # doing masks
    @staticmethod
    def mask_exprs(v: BK.Expr, erase_mask: BK.Expr, idx_mask: int):
        erase_mask = erase_mask.long()
        ret_arr = v * (1-erase_mask) + idx_mask * erase_mask
        return ret_arr

# plain one: currently the same as base
class PlainInputHelper(InputHelper):
    def __init__(self, name: str, voc: SimpleVocab):
        super().__init__(name)
        # --
        seq_name = "seq_" + name
        self.idx_get_f = lambda x: getattr(x, seq_name).idxes
        self.voc = voc
        self.idx_mask = voc.mask  # use this one!
        self.padder = DataPadder(2, pad_vals=voc.pad)  # dim=2, pad=[pad]

    def prepare(self, insts: List):
        return InputHelper.prepare_batch(insts, self.idx_get_f, self.padder)

    def mask(self, vs, erase_masks):
        return InputHelper.mask_exprs(vs, erase_masks, self.idx_mask)

# input helper for char
class CharInputHelper(PlainInputHelper):
    def __init__(self, name: str, voc: SimpleVocab):
        super().__init__(name+"_char", voc)
        # rewrite default ones
        seq_name = "seq_" + name
        self.idx_get_f = lambda x: getattr(x, seq_name).get_char_seq().idxes
        self.padder = DataPadder(3, pad_vals=voc.pad)  # dim=3, pad=[pad]

    def mask(self, v: BK.Expr, erase_mask: BK.Expr):
        return super().mask(v, erase_mask.unsqueeze(-1))  # todo(note): simply allow all-mask

# Absolute position input helper
class PosiInputHelper(InputHelper):
    def __init__(self, name: str):
        super().__init__(name)

    # simply return a input shape here
    def prepare(self, insts: List):
        shape = (len(insts), max(len(z) for z in insts))
        return shape

    def mask(self, vs, erase_masks):
        return vs  # do nothing!

# BERT input (simple input, no other factors) helper
class BertInputHelper(InputHelper):
    def __init__(self, name: str, berter: BertEncoder):
        super().__init__(name+"_bert")
        # --
        self.berter = berter
        seq_name = "seq_" + name
        toker = berter.sub_toker
        self.subseq_get_f = lambda x: getattr(x, seq_name).get_subword_seq(toker)

    # return BatchedInput
    def prepare(self, insts: List):
        _f = self.subseq_get_f
        ib = self.berter.create_input_batch([_f(z) for z in insts])
        return ib

    # whole word mask!
    def mask(self, ib, erase_mask):
        ib.set_repl_masks(erase_mask, is_orig=True)  # set repl_mask
        return ib

# a group of inputters to handle the same inputs
class InputterGroup:
    def __init__(self):
        self.inputters: Dict[str, InputHelper] = OrderedDict()

    def register_inputter(self, key: str, inputer: InputHelper):
        assert key not in self.inputters
        self.inputters[key] = inputer

    def get_inputter(self, key: str, df=None):
        return self.inputters.get(key, df)

    def prepare_inputs(self, insts: List):
        ret = OrderedDict()
        # first basic masks
        mask_arr = DataPadder.lengths2mask([len(z) for z in insts])
        ret["mask"] = BK.input_real(mask_arr)
        ret["mask_arr"] = mask_arr
        # then the rest
        for key, inputter in self.inputters.items():
            ret[key] = inputter.prepare(insts)
        return ret

    def mask_inputs(self, input_map: Dict, input_erase_mask: BK.Expr, nomask_names_set: Set = None):
        ret = OrderedDict()  # make a new one!
        # directly copy masks
        ret["mask"] = input_map["mask"]
        ret["mask_arr"] = input_map["mask_arr"]
        ret["erase_mask"] = input_erase_mask
        # then mask others
        if nomask_names_set is None:
            nomask_names_set = set()  # empty
        for key, inputter in self.inputters.items():
            if key in nomask_names_set:
                ret[key] = input_map[key]
            else:
                ret[key] = inputter.mask(input_map[key], input_erase_mask)
        return ret

# =====
# InputEmbedder: from prepared inputs to tensor

class InputEmbedderConf(BasicConf):
    def __init__(self):
        super().__init__()
        # basic for Embedding
        self.dim = 0  # also as a switch, <0 means not active!!

@node_reg(InputEmbedderConf)
class InputEmbedderNode(BasicNode):
    def __init__(self, conf: InputEmbedderConf, name="UNK", **kwargs):
        super().__init__(conf, **kwargs)
        self.name = name

    def extra_repr(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def get_output_dims(self, *input_dims):
        return (self.conf.dim, )

# plain one
class PlainInputEmbedderConf(InputEmbedderConf):
    def __init__(self):
        super().__init__()
        self.econf = EmbeddingConf()
        # mainly for word
        self.init_from_pretrain = False  # try to init from pretrain?
        self.rare_unk_rate = 0.5  # replace unk words with UNK when training
        self.rare_unk_thr = 0  # only replace those if freq(count)<=this

@node_reg(PlainInputEmbedderConf)
class PlainInputEmbedderNode(InputEmbedderNode):
    def __init__(self, conf: PlainInputEmbedderConf, voc: SimpleVocab, npvec: np.ndarray=None, name="UNK"):
        super().__init__(conf, name)
        # --
        conf: PlainInputEmbedderConf = self.conf
        self.voc = voc
        # check init embeddings
        if conf.init_from_pretrain:
            zlog(f"Try to init {self.extra_repr()} with npvec.shape={npvec.shape if (npvec is not None) else None}")
            if npvec is None:
                zwarn("warning: cannot get pre-trained embeddings to init!!")
        # get rare unk range
        voc_rare_unk_mask = []
        for w in self.voc.full_i2w:
            c = self.voc.word2count(w, df=None)
            voc_rare_unk_mask.append(float(c is not None and c<=conf.rare_unk_thr))
        self.rare_unk_mask = BK.input_real(voc_rare_unk_mask)  # stored tensor!
        # self.register_buffer()  # todo(note): do we need register buffer?
        self.use_rare_unk = (conf.rare_unk_rate>0. and conf.rare_unk_thr>0)
        # add the real embedding node
        self.E = EmbeddingNode(conf.econf, npvec=npvec, osize=conf.dim, n_words=len(self.voc))

    # [*, len] -> [*, 1?+len+1?]
    def forward(self, inputs, add_bos=False, add_eos=False):
        conf: PlainInputEmbedderConf = self.conf
        # --
        voc = self.voc
        input_t = BK.input_idx(inputs)  # [*, len]
        # rare unk in training
        if self.is_training() and self.use_rare_unk:
            rare_unk_rate = conf.rare_unk_rate
            cur_unk_imask = (self.rare_unk_mask[input_t] * (BK.rand(BK.get_shape(input_t))<rare_unk_rate)).long()
            input_t = input_t * (1-cur_unk_imask) + voc.unk * cur_unk_imask
        # bos and eos
        all_input_slices = []
        slice_shape = BK.get_shape(input_t)[:-1]+[1]
        if add_bos:
            all_input_slices.append(BK.constants(slice_shape, voc.bos, dtype=input_t.dtype))
        all_input_slices.append(input_t)  # [*, len]
        if add_eos:
            all_input_slices.append(BK.constants(slice_shape, voc.eos, dtype=input_t.dtype))
        final_input_t = BK.concat(all_input_slices, -1)  # [*, 1?+len+1?]
        # finally
        ret = self.E(final_input_t)  # [*, ??, dim]
        return ret

# char-embed + char-cnn
class CharCnnInputEmbedderConf(PlainInputEmbedderConf):
    def __init__(self):
        super().__init__()  # dim still means switch, which is CharEmbed dim
        # cnn
        self.cnn_out = 50
        self.cnn = CnnConf().direct_update(n_layers=1, out_pool="max")

@node_reg(CharCnnInputEmbedderConf)
class CharCnnInputEmbedderNode(PlainInputEmbedderNode):
    def __init__(self, conf: CharCnnInputEmbedderConf, voc: SimpleVocab, npvec: np.ndarray=None, name="CHAR"):
        super().__init__(conf, voc, npvec=npvec, name=name)
        conf: CharCnnInputEmbedderConf = self.conf
        # --
        self.cnn = CnnNode(conf.cnn, isize=conf.dim, osize=conf.cnn_out)

    def get_output_dims(self, *input_dims):
        return (self.cnn.get_output_dims()[0], )

    def forward(self, inputs, add_bos=False, add_eos=False):
        conf: CharCnnInputEmbedderConf = self.conf
        # --
        voc = self.voc
        char_input_t = BK.input_idx(inputs)  # [*, len]
        # todo(note): no need for replacing to unk for char!!
        # bos and eos
        all_input_slices = []
        slice_shape = BK.get_shape(char_input_t)
        slice_shape[-2] = 1  # [*, 1, clen]
        if add_bos:
            all_input_slices.append(BK.constants(slice_shape, voc.bos, dtype=char_input_t.dtype))
        all_input_slices.append(char_input_t)  # [*, len, clen]
        if add_eos:
            all_input_slices.append(BK.constants(slice_shape, voc.eos, dtype=char_input_t.dtype))
        final_input_t = BK.concat(all_input_slices, -2)  # [*, 1?+len+1?, clen]
        # char embeddings
        char_embed_expr = self.E(final_input_t)  # [*, ??, dim]
        # char cnn
        ret = self.cnn(char_embed_expr)
        return ret

# absolute posi embeddings
class PosiInputEmbedderConf(InputEmbedderConf):
    def __init__(self):
        super().__init__()
        self.posi = PosiEmbeddingConf().direct_update(min_val=0)  # >=0

@node_reg(PosiInputEmbedderConf)
class PosiInputEmbedderNode(InputEmbedderNode):
    def __init__(self, conf: PosiInputEmbedderConf, name="POSI"):
        super().__init__(conf, name=name)
        conf: PosiInputEmbedderConf = self.conf
        # --
        self.E = PosiEmbeddingNode(conf.posi, osize=conf.dim)

    # [*, len] -> [*, 1?+len+1?, D]
    def forward(self, inputs, add_bos=False, add_eos=False):
        conf: PosiInputEmbedderConf = self.conf
        # --
        try:
            # input is a shape as prepared by "PosiHelper"
            batch_size, max_len = inputs
            if add_bos:
                max_len += 1
            if add_eos:
                max_len += 1
            posi_idxes = BK.arange_idx(max_len)  # [?len?]
            ret = self.E(posi_idxes).unsqueeze(0).expand(batch_size, -1, -1)
        except:
            # input is tensor
            posi_idxes = BK.input_idx(inputs)  # [*, len]
            cur_maxlen = BK.get_shape(posi_idxes, -1)
            # --
            all_input_slices = []
            slice_shape = BK.get_shape(posi_idxes)[:-1]+[1]
            if add_bos:  # add 0 and offset
                all_input_slices.append(BK.constants(slice_shape, 0, dtype=posi_idxes.dtype))
                cur_maxlen += 1
                posi_idxes += 1
            all_input_slices.append(posi_idxes)  # [*, len]
            if add_eos:
                all_input_slices.append(BK.constants(slice_shape, cur_maxlen, dtype=posi_idxes.dtype))
            final_input_t = BK.concat(all_input_slices, -1)  # [*, 1?+len+1?]
            # finally
            ret = self.E(final_input_t)  # [*, ??, dim]
        return ret

# bert embeddings: a wrapper for berter
# -- (mostly for bert features, thus no too much complexities here, simply forward with seq&mask for subwords)
class BertInputEmbedderConf(InputEmbedderConf):
    def __init__(self):
        super().__init__()

# forward the whole BERT, rather than just embeddings (mostly used as freezed features)
@node_reg(BertInputEmbedderConf)
class BertInputEmbedderNode(InputEmbedderNode):
    def __init__(self, berter: BertEncoder, conf: BertInputEmbedderConf, name="BERT"):
        super().__init__(conf, name)
        conf: BertInputEmbedderConf = self.conf
        # --
        # todo(note): not adding as a sub-module!!
        self.setattr_borrow("_berter", berter)
        # reset output dim (mainly as records, not used in other places!)
        conf.dim = self._berter.get_output_dims()[0]

    def forward(self, ib: BerterInputBatch, add_bos=False, add_eos=False):
        res = self._berter(ib, inc_cls=add_bos)
        return res

# =====
# the embedder group

class EmbedderGroupConf(BasicConf):
    def __init__(self):
        super().__init__()
        # overall control since we need to make them consistent!!
        self.add_bos = False
        self.add_eos = False

@node_reg(EmbedderGroupConf)
class EmbedderGroup(BasicNode):
    def __init__(self, conf: EmbedderGroupConf, **kwargs):
        super().__init__(conf, **kwargs)
        self.embedders: Dict[str, Tuple[InputEmbedderNode, str]] = OrderedDict()
        # todo(+N): currently disallow add_eos since not added in a reasonable way!!
        assert not self.conf.add_eos, "To be implemented!"

    def register_embedder(self, key: str, embedder: InputEmbedderNode, inputter_key: str):
        assert key not in self.embedders
        self.embedders[key] = (embedder, inputter_key)
        # extraly register module!!
        self.add_module(f"_M{key}", embedder)

    def get_embedder(self, key: str, df=None):
        zz = self.embedders.get(key)
        return df if zz is None else zz[0]

    def get_inputter(self, key: str, df=None):
        zz = self.embedders.get(key)
        return df if zz is None else zz[1]

    def forward(self, input_map: Dict):
        add_bos, add_eos = self.conf.add_bos, self.conf.add_eos
        ret = OrderedDict()  # [*, len, ?]
        for key, embedder_pack in self.embedders.items():  # according to REG order!!
            embedder, input_name = embedder_pack
            one_expr = embedder(input_map[input_name], add_bos=add_bos, add_eos=add_eos)
            ret[key] = one_expr
        # mask expr
        mask_expr = input_map.get("mask")
        if mask_expr is not None:
            all_input_slices = []
            mask_slice = BK.constants(BK.get_shape(mask_expr)[:-1]+[1], 1, dtype=mask_expr.dtype)  # [*, 1]
            if add_bos:
                all_input_slices.append(mask_slice)
            all_input_slices.append(mask_expr)
            if add_eos:
                all_input_slices.append(mask_slice)
            mask_expr = BK.concat(all_input_slices, -1)  # [*, ?+len+?]
        return mask_expr, ret

    def extra_repr(self) -> str:
        return f"EmbedderGroup({list(self.embedders.keys())})"

    def get_output_dims(self, *input_dims):
        raise RuntimeError("Return dict for this node!")
