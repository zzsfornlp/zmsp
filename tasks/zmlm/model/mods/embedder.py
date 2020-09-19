#

import numpy as np
from typing import Tuple, Iterable, Dict, List, Set
from collections import OrderedDict
from copy import deepcopy

from msp.utils import Conf, zcheck, zwarn, zlog
from msp.data import VocabPackage
from msp.zext.seq_helper import DataPadder
from msp.nn import BK
from msp.nn.layers import BasicNode, PosiEmbedding2, Embedding, CnnLayer, NoDropRop, Dropout, Affine, Sequential, LayerNorm
from msp.nn.modules import Berter2, Berter2Conf, Berter2Seq
from ...data.insts import GeneralSentence

# confs for one component: some may be effective only for certain cases
class EmbedderCompConf(Conf):
    def __init__(self):
        self.comp_dim = 0  # by default 0 meaning this comp is not active
        self.comp_drop = 0.  # by default 0.
        self.comp_init_scale = 0.  # init-scale for params
        self.comp_init_from_pretrain = False  # try to init from pretrain?
        # mainly for word
        self.comp_rare_unk = 0.5  # replace unk words with UNK when training
        self.comp_rare_thr = 0  # only replace those if freq<=this

# overall conf
class EmbedderNodeConf(Conf):
    def __init__(self):
        # embedder components
        self.ec_word = EmbedderCompConf().init_from_kwargs(comp_dim=100)  # we mostly needs words
        self.ec_pos = EmbedderCompConf()
        self.ec_char = EmbedderCompConf()
        self.ec_posi = EmbedderCompConf()
        self.ec_bert = EmbedderCompConf()
        # add speical node for root
        self.add_root_token = True  # add the special bos/root/cls/...
        # the padding idx 0, should it be zero? note: nope since there may be NAN problems with pre-norm
        self.embed_fix_row0 = False
        # final projection layer
        self.emb_proj_dim = 0  # 0 means no, and thus simply concat
        self.emb_proj_act = "linear"  # proj_act
        self.emb_proj_init_scale = 1.
        self.emb_proj_norm = False  # cannot add&norm since possible dim mismatch
        # special case: cnn-char encoding
        self.char_cnn_hidden = 50  # split by windows
        self.char_cnn_windows = [3, 5]
        # special case: bert setting
        self.bert_conf = Berter2Conf().init_from_kwargs(bert2_retinc_cls=True, bert2_training_mask_rate=0.,
                                                        bert2_output_mode="weighted")

    @property
    def ec_dict(self):
        return OrderedDict([('word', self.ec_word), ('pos', self.ec_pos), ('char', self.ec_char),
                            ('posi', self.ec_posi), ('bert', self.ec_bert)])

    def do_validate(self):
        assert self.bert_conf.bert2_output_mode != "layered", "Not applicable here!"
        if self.add_root_token:  # setup for bert (CLS as the special node)
            self.bert_conf.bert2_retinc_cls = True
        else:
            self.bert_conf.bert2_retinc_cls = False

# =====
# ModelNode and Helper for each type of input

# -----
# Input helper is used to prepare and transform inputs
class InputHelper:
    def __init__(self, comp_name, vpack: VocabPackage):
        self.comp_name = comp_name
        self.comp_seq_name = f"{comp_name}_seq"
        self.voc = vpack.get_voc(comp_name)
        self.padder = DataPadder(2, pad_vals=0)  # pad 0

    # return batched arr
    def prepare(self, insts: List[GeneralSentence]):
        cur_input_list = [getattr(z, self.comp_seq_name).idxes for z in insts]
        cur_input_arr, _ = self.padder.pad(cur_input_list)
        return cur_input_arr

    def mask(self, v, erase_mask):
        IDX_MASK = self.voc.err  # todo(note): this one is unused, thus just take it!
        ret_arr = v * (1-erase_mask) + IDX_MASK * erase_mask
        return ret_arr

    @staticmethod
    def get_input_helper(name, *args, **kwargs):
        helper_type = {"char": CharCnnHelper, "posi": PosiHelper, "bert": BertInputHelper}.get(name, PlainSeqHelper)
        return helper_type(*args, **kwargs)

# InputEmbedNode is used for embedding corresponding to the Input
class InputEmbedNode(BasicNode):
    def __init__(self, pc: BK.ParamCollection, comp_name: str, ec_conf: EmbedderCompConf,
                 conf: EmbedderNodeConf, vpack: VocabPackage):
        super().__init__(pc, None, None)
        # -----
        self.ec_conf = ec_conf
        self.conf = conf
        self.comp_name, self.comp_dim, self.comp_dropout, self.comp_init_scale = \
            comp_name, ec_conf.comp_dim, ec_conf.comp_drop, ec_conf.comp_init_scale
        self.voc = vpack.get_voc(comp_name, None)
        self.output_dim = self.comp_dim  # by default the input size (may be changed later)
        self.dropout = None  # created later (after deciding output shape)

    def create_dropout_node(self):
        if self.comp_dropout > 0.:
            self.dropout = self.add_sub_node(
                f"D{self.comp_name}", Dropout(self.pc, (self.output_dim,), fix_rate=self.comp_dropout))
        else:
            self.dropout = lambda x: x
        return self.dropout

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    @staticmethod
    def get_input_embed_node(name, *args, **kwargs):
        node_type = {"char": CharCnnNode, "posi": PosiNode, "bert": BertInputNode}.get(name, PlainSeqNode)
        return node_type(*args, **kwargs)

# -----
# plain seq

# the basic one is the plain one
PlainSeqHelper = InputHelper

#
class PlainSeqNode(InputEmbedNode):
    def __init__(self, pc: BK.ParamCollection, comp_name: str, ec_conf: EmbedderCompConf,
                 conf: EmbedderNodeConf, vpack: VocabPackage):
        super().__init__(pc, comp_name, ec_conf, conf, vpack)
        # -----
        # get embeddings
        npvec = None
        if self.ec_conf.comp_init_from_pretrain:
            npvec = vpack.get_emb(comp_name)
            zlog(f"Try to init InputEmbedNode {comp_name} with npvec.shape={npvec.shape if (npvec is not None) else None}")
            if npvec is None:
                zwarn("Warn: cannot get pre-trained embeddings to init!!")
        # get rare unk range
        # - get freq vals, make sure special ones will not be pruned; todo(note): directly use that field
        voc_rare_mask = [float(z is not None and z<=ec_conf.comp_rare_thr) for z in self.voc.final_vals]
        self.rare_mask = BK.input_real(voc_rare_mask)
        self.use_rare_unk = (ec_conf.comp_rare_unk>0. and ec_conf.comp_rare_thr>0)
        # --
        # dropout outside explicitly
        self.E = self.add_sub_node(f"E{self.comp_name}", Embedding(
            pc, len(self.voc), self.comp_dim, fix_row0=conf.embed_fix_row0, npvec=npvec, name=comp_name,
            init_rop=NoDropRop(), init_scale=self.comp_init_scale))
        self.create_dropout_node()

    # [*, slen] -> [*, 1+slen, D]
    def __call__(self, input, add_root_token: bool):
        voc = self.voc
        # todo(note): append a [cls/root] idx, currently use "bos"
        input_t = BK.input_idx(input)  # [*, 1+slen]
        # rare unk in training
        if self.rop.training and self.use_rare_unk:
            rare_unk_rate = self.ec_conf.comp_rare_unk
            cur_unk_imask = (self.rare_mask[input_t] * (BK.rand(BK.get_shape(input_t))<rare_unk_rate)).detach().long()
            input_t = input_t * (1-cur_unk_imask) + self.voc.unk * cur_unk_imask
        # root
        if add_root_token:
            input_t_p0 = BK.constants(BK.get_shape(input_t)[:-1]+[1], voc.bos, dtype=input_t.dtype)  # [*, 1+slen]
            input_t_p1 = BK.concat([input_t_p0, input_t], -1)
        else:
            input_t_p1 = input_t
        expr = self.E(input_t_p1)  # [*, 1?+slen]
        return self.dropout(expr)

# -----
# char-cnn seq

class CharCnnHelper(InputHelper):
    def __init__(self, comp_name, vpack: VocabPackage):
        super().__init__(comp_name, vpack)
        self.padder = DataPadder(3, pad_vals=0)  # replace the padder

    def mask(self, v, erase_mask):
        return super().mask(v, np.expand_dims(erase_mask, axis=-1))  # todo(note): for simplicity, simply allow all-unk

class CharCnnNode(PlainSeqNode):
    def __init__(self, pc: BK.ParamCollection, comp_name: str, ec_conf: EmbedderCompConf,
                 conf: EmbedderNodeConf, vpack: VocabPackage):
        super().__init__(pc, comp_name, ec_conf, conf, vpack)
        # -----
        per_cnn_size = conf.char_cnn_hidden // len(conf.char_cnn_windows)
        self.char_cnns = [self.add_sub_node("char_cnn", CnnLayer(
            self.pc, self.comp_dim, per_cnn_size, z, pooling="max", act="tanh", init_rop=NoDropRop()))
                          for z in conf.char_cnn_windows]
        self.output_dim = conf.char_cnn_hidden
        self.create_dropout_node()

    # [*, slen, wlen] -> [*, 1+slen, wlen, D]
    def __call__(self, char_input, add_root_token: bool):
        char_input_t = BK.input_idx(char_input)  # [*, slen, wlen]
        if add_root_token:
            slice_shape = BK.get_shape(char_input_t)
            slice_shape[-2] = 1
            char_input_t0 = BK.constants(slice_shape, 0, dtype=char_input_t.dtype)  # todo(note): simply put 0 here!
            char_input_t1 = BK.concat([char_input_t0, char_input_t], -2)  # [*, 1?+slen, wlen]
        else:
            char_input_t1 = char_input_t
        char_embeds = self.E(char_input_t1)  # [*, 1?+slen, wlen, D]
        char_cat_expr = BK.concat([z(char_embeds) for z in self.char_cnns])
        return self.dropout(char_cat_expr)  # todo(note): only final dropout

# -----
# positional seq (absolute position)

class PosiHelper(InputHelper):
    def __init__(self, comp_name, vpack: VocabPackage):
        super().__init__(comp_name, vpack)

    # return batched arr
    def prepare(self, insts: List[GeneralSentence]):
        shape = (len(insts), max(len(z) for z in insts))
        return shape

    def mask(self, v, erase_mask):
        return v  # todo(note): generally no need to change posi info for word mask

class PosiNode(InputEmbedNode):
    def __init__(self, pc: BK.ParamCollection, comp_name: str, ec_conf: EmbedderCompConf,
                 conf: EmbedderNodeConf, vpack: VocabPackage):
        super().__init__(pc, comp_name, ec_conf, conf, vpack)
        # -----
        self.node = self.add_sub_node("posi", PosiEmbedding2(pc, self.comp_dim, init_scale=self.comp_init_scale))
        self.create_dropout_node()

    def __call__(self, input_v, add_root_token: bool):
        if isinstance(input_v, np.ndarray):
            # direct use this [batch_size, slen] as input
            posi_idxes = BK.input_idx(input_v)
            expr = self.node(posi_idxes)  # [batch_size, slen, D]
        else:
            # input is a shape as prepared by "PosiHelper"
            batch_size, max_len = input_v
            if add_root_token:
                max_len += 1
            posi_idxes = BK.arange_idx(max_len)  # [1?+slen] add root=0 here
            expr = self.node(posi_idxes).unsqueeze(0).expand(batch_size, -1, -1)
        return self.dropout(expr)

# -----
# for bert

class BertInputHelper(InputHelper):
    def __init__(self, comp_name, vpack: VocabPackage):
        super().__init__(comp_name, vpack)
        # ----
        self.berter: Berter2 = None
        self.mask_id = None

    def set_berter(self, berter):
        self.berter = berter  # todo(note): need to set from outside!
        self.mask_id = self.berter.tokenizer.mask_token_id

    # return List[Bert]
    def prepare(self, insts: List[GeneralSentence]):
        _key = "_bert2seq"
        rets = []
        for z in insts:
            _v = z.features.get(_key)
            if _v is None:  # no cache, rebuild it
                _v = self.berter.subword_tokenize2(z.word_seq.vals, True)  # here use orig strs
                z.features[_key] = _v
            rets.append(_v)
        return rets

    def mask(self, v, erase_mask):
        ret = [one_b2s.apply_mask_new(one_mask, self.mask_id) for one_b2s, one_mask in zip(v, erase_mask)]
        return ret

class BertInputNode(InputEmbedNode):
    def __init__(self, pc: BK.ParamCollection, comp_name: str, ec_conf: EmbedderCompConf,
                 conf: EmbedderNodeConf, vpack: VocabPackage):
        super().__init__(pc, comp_name, ec_conf, conf, vpack)
        # -----
        if conf.add_root_token:
            assert conf.bert_conf.bert2_retinc_cls, "Require add root token by include CLS for Berter2"
        self.berter = self.add_sub_node("bert", Berter2(pc, conf.bert_conf))
        self.output_dim = self.berter.get_output_dims()[0]  # fix size
        self.create_dropout_node()

    def __call__(self, input_list: List[Berter2Seq], add_root_token: bool):
        # no need to put add_root_token since already setup for berter
        expr = self.berter(input_list)
        return self.dropout(expr)  # [bs, 1?+slen, D]

# =====
# the overall manager
class EmbedderNode(BasicNode):
    def __init__(self, pc: BK.ParamCollection, conf: EmbedderNodeConf, vpack: VocabPackage):
        super().__init__(pc, None, None)
        self.conf = conf
        self.vpack = vpack
        self.add_root_token = conf.add_root_token
        # -----
        self.nodes = []  # params
        self.comp_names = []
        self.comp_dims = []  # real dims
        self.berter: Berter2 = None
        for comp_name, comp_conf in conf.ec_dict.items():
            if comp_conf.comp_dim > 0:
                # directly get the nodes
                one_node = InputEmbedNode.get_input_embed_node(comp_name, pc, comp_name, comp_conf, conf, vpack)
                comp_dim = one_node.get_output_dims()[0]  # fix dim
                # especially for berter
                if comp_name == "bert":
                    assert self.berter is None
                    self.berter = one_node.berter
                # general steps
                self.comp_names.append(comp_name)
                self.nodes.append(self.add_sub_node(f"EC{comp_name}", one_node))
                self.comp_dims.append(comp_dim)
        # final projection?
        self.has_proj = (conf.emb_proj_dim > 0)
        if self.has_proj:
            proj_layer = Affine(self.pc, sum(self.comp_dims), conf.emb_proj_dim,
                                act=conf.emb_proj_act, init_scale=conf.emb_proj_init_scale)
            if conf.emb_proj_norm:
                norm_layer = LayerNorm(self.pc, conf.emb_proj_dim)
                self.final_layer = self.add_sub_node("fl", Sequential(self.pc, [proj_layer, norm_layer]))
            else:
                self.final_layer = self.add_sub_node("fl", proj_layer)
            self.output_dim = conf.emb_proj_dim
        else:
            self.final_layer = None
            self.output_dim = sum(self.comp_dims)

    def get_node(self, name, df=None):
        try:
            _idx = self.comp_names.index(name)
            return self.nodes[_idx]
        except:
            return df

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    def __repr__(self):
        comp_ss = [f"{n}({d})" for n, d in zip(self.comp_names, self.comp_dims)]
        return f"# EmbedderNode: {comp_ss} -> {self.output_dim}"

    # mainly need inputs of [*, slen] or other stuffs specific to the nodes
    # return reprs, masks
    def __call__(self, input_map: Dict):
        exprs = []
        # get masks: this mask is for validing of inst batching
        final_masks = BK.input_real(input_map["mask"])  # [*, slen]
        if self.add_root_token:  # append 1
            slice_t = BK.constants(BK.get_shape(final_masks)[:-1]+[1], 1.)
            final_masks = BK.concat([slice_t, final_masks], -1)  # [*, 1+slen]
        # -----
        # for each component
        for idx, name in enumerate(self.comp_names):
            cur_node = self.nodes[idx]
            cur_input = input_map[name]
            cur_expr = cur_node(cur_input, self.add_root_token)
            exprs.append(cur_expr)
        # -----
        concated_exprs = BK.concat(exprs, dim=-1)
        # optional proj
        if self.has_proj:
            final_expr = self.final_layer(concated_exprs)
        else:
            final_expr = concated_exprs
        return final_expr, final_masks

# =====
# the overall input helper
class Inputter:
    def __init__(self, embedder: EmbedderNode, vpack: VocabPackage):
        self.vpack = vpack
        self.embedder = embedder
        # -----
        # prepare the inputter
        self.comp_names = embedder.comp_names
        self.comp_helpers = []
        for comp_name in self.comp_names:
            one_helper = InputHelper.get_input_helper(comp_name, comp_name, vpack)
            self.comp_helpers.append(one_helper)
            if comp_name == "bert":
                assert embedder.berter is not None
                one_helper.set_berter(berter=embedder.berter)
        # ====
        self.mask_padder = DataPadder(2, pad_vals=0., mask_range=2)

    def __call__(self, insts: List[GeneralSentence]):
        # first pad words to get masks
        _, masks_arr = self.mask_padder.pad([z.word_seq.idxes for z in insts])
        # then get each one
        ret_map = {"mask": masks_arr}
        for comp_name, comp_helper in zip(self.comp_names, self.comp_helpers):
            ret_map[comp_name] = comp_helper.prepare(insts)
        return ret_map

    # todo(note): return new masks (input is read only!!)
    def mask_input(self, input_map: Dict, input_erase_mask, nomask_names_set: Set):
        ret_map = {"mask": input_map["mask"]}
        for comp_name, comp_helper in zip(self.comp_names, self.comp_helpers):
            if comp_name in nomask_names_set:  # direct borrow that one
                ret_map[comp_name] = input_map[comp_name]
            else:
                ret_map[comp_name] = comp_helper.mask(input_map[comp_name], input_erase_mask)
        return ret_map

# =====
# aug word2: both inputter and embedder
class AugWord2Node(BasicNode):
    def __init__(self, pc: BK.ParamCollection, ref_conf: EmbedderNodeConf, ref_vpack: VocabPackage,
                 comp_name: str, comp_dim: int, output_dim: int):
        super().__init__(pc, None, None)
        # =====
        self.ref_conf = ref_conf
        self.ref_vpack = ref_vpack
        self.add_root_token = ref_conf.add_root_token
        # modify conf
        _tmp_ec_conf = deepcopy(ref_conf.ec_word)
        _tmp_ec_conf.comp_dim = comp_dim
        _tmp_ec_conf.comp_init_from_pretrain = True
        # -----
        self.w_node = self.add_sub_node("nw", InputEmbedNode.get_input_embed_node(
            comp_name, pc, comp_name, _tmp_ec_conf, ref_conf, ref_vpack))
        self.c_node = self.add_sub_node("nc", InputEmbedNode.get_input_embed_node(
            "char", pc, "char", deepcopy(ref_conf.ec_char), ref_conf, ref_vpack))
        # final proj
        _dims = [self.w_node.get_output_dims()[0], self.c_node.get_output_dims()[0]]
        self.final_layer = self.add_sub_node("fl", Affine(
            self.pc, _dims, output_dim, act=ref_conf.emb_proj_act, init_scale=ref_conf.emb_proj_init_scale))
        self.output_dim = output_dim
        # =====
        # inputter
        self.w_helper = InputHelper.get_input_helper(comp_name, comp_name, ref_vpack)
        self.c_helper = InputHelper.get_input_helper("char", "char", ref_vpack)

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    def __call__(self, insts):
        # inputter
        input_w_arr = self.w_helper.prepare(insts)
        input_c_arr = self.c_helper.prepare(insts)
        # embedder
        output_w_t = self.w_node(input_w_arr, self.add_root_token)
        output_c_t = self.c_node(input_c_arr, self.add_root_token)
        final_t = self.final_layer([output_w_t, output_c_t])
        return final_t

# b tasks/zmlm/model/mods/embedder:342
