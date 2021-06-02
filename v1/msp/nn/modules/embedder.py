#

import numpy as np
from typing import Tuple, Iterable

from msp.utils import Conf, zcheck, zwarn
from msp.data import VocabPackage
from msp.nn import BK
from msp.nn.layers import BasicNode, Embedding, CnnLayer, PosiEmbedding, Affine, LayerNorm, Sequential, \
    DropoutLastN, Dropout

#
class EmbedConf(Conf):
    def __init__(self):
        # -- inputs (the first three are special, thus explicitly handle them)
        # for the inputs (all can be optional, but at least one should be provided)
        self.dim_word = 100
        self.init_words_from_pretrain = True
        self.word_freeze = False
        # cnn-char encoding
        self.dim_char = 0
        self.char_cnn_hidden = 30       # split by windows
        self.char_cnn_windows = [5, ]
        # using either trainable clipped distance or fixed sin-cos
        self.dim_posi = 0      # absolute positional embedding
        self.posi_fix_sincos = True
        self.posi_clip = 5000      # only used when fix==False
        self.posi_freeze = True
        # extra inputs, like POS, ...
        # self.dim_extras = [50, ]
        # self.extra_names = ["pos", ]
        self.dim_extras = []
        self.extra_names = []
        # aux direct features, like mbert ...
        self.dim_auxes = []
        self.fold_auxes = []  # how many folds per aux-repr, used to perform Elmo's styled linear combination
        #
        # -- project the concat before contextual encoders
        self.emb_proj_dim = 0
        self.emb_proj_norm = False       # cannot add&norm since possible dim mismatch

    def do_validate(self):
        # confirm int for dims
        self.dim_extras = [int(z) for z in self.dim_extras]
        self.dim_auxes = [int(z) for z in self.dim_auxes]
        self.fold_auxes = [int(z) for z in self.fold_auxes]
        # by default, fold=1
        fold_gap = max(0, len(self.dim_auxes) - len(self.fold_auxes))
        self.fold_auxes += [1] * fold_gap

# combining the inputs after projected to embeddings
# todo(0): names in vpack should be consistent: word, char, pos, ...
class MyEmbedder(BasicNode):
    def __init__(self, pc: BK.ParamCollection, econf: EmbedConf, vpack: VocabPackage):
        super().__init__(pc, None, None)
        self.conf = econf
        #
        repr_sizes = []
        # word
        self.has_word = (econf.dim_word>0)
        if self.has_word:
            npvec = vpack.get_emb("word") if econf.init_words_from_pretrain else None
            self.word_embed = self.add_sub_node("ew", Embedding(self.pc, len(vpack.get_voc("word")), econf.dim_word, npvec=npvec, name="word", freeze=econf.word_freeze))
            repr_sizes.append(econf.dim_word)
        # char
        self.has_char = (econf.dim_char>0)
        if self.has_char:
            # todo(warn): cnns will also use emb's drop?
            self.char_embed = self.add_sub_node("ec", Embedding(self.pc, len(vpack.get_voc("char")), econf.dim_char, name="char"))
            per_cnn_size = econf.char_cnn_hidden // len(econf.char_cnn_windows)
            self.char_cnns = [self.add_sub_node("cnnc", CnnLayer(self.pc, econf.dim_char, per_cnn_size, z, pooling="max", act="tanh")) for z in econf.char_cnn_windows]
            repr_sizes.append(econf.char_cnn_hidden)
        # posi: absolute positional embeddings
        self.has_posi = (econf.dim_posi>0)
        if self.has_posi:
            self.posi_embed = self.add_sub_node("ep", PosiEmbedding(self.pc, econf.dim_posi, econf.posi_clip, econf.posi_fix_sincos, econf.posi_freeze))
            repr_sizes.append(econf.dim_posi)
        # extras: like POS, ...
        self.dim_extras = econf.dim_extras
        self.extra_names = econf.extra_names
        zcheck(len(self.dim_extras) == len(self.extra_names), "Unmatched dims and names!")
        self.extra_embeds = []
        for one_extra_dim, one_name in zip(self.dim_extras, self.extra_names):
            self.extra_embeds.append(self.add_sub_node(
                "ext", Embedding(self.pc, len(vpack.get_voc(one_name)), one_extra_dim,
                                 npvec=vpack.get_emb(one_name, None), name="extra:"+one_name)))
            repr_sizes.append(one_extra_dim)
        # auxes
        self.dim_auxes = econf.dim_auxes
        self.fold_auxes = econf.fold_auxes
        self.aux_overall_gammas = []
        self.aux_fold_lambdas = []
        for one_aux_dim, one_aux_fold in zip(self.dim_auxes, self.fold_auxes):
            repr_sizes.append(one_aux_dim)
            # aux gamma and fold trainable lambdas
            self.aux_overall_gammas.append(self.add_param("AG", (), 1.))  # scalar
            self.aux_fold_lambdas.append(self.add_param("AL", (), [1./one_aux_fold for _ in range(one_aux_fold)]))  # [#fold]
        # =====
        # another projection layer? & set final dim
        if len(repr_sizes)<=0:
            zwarn("No inputs??")
        # zcheck(len(repr_sizes)>0, "No inputs?")
        self.repr_sizes = repr_sizes
        self.has_proj = (econf.emb_proj_dim>0)
        if self.has_proj:
            proj_layer = Affine(self.pc, sum(repr_sizes), econf.emb_proj_dim)
            if econf.emb_proj_norm:
                norm_layer = LayerNorm(self.pc, econf.emb_proj_dim)
                self.final_layer = self.add_sub_node("fl", Sequential(self.pc, [proj_layer, norm_layer]))
            else:
                self.final_layer = self.add_sub_node("fl", proj_layer)
            self.output_dim = econf.emb_proj_dim
        else:
            self.final_layer = None
            self.output_dim = sum(repr_sizes)
        # =====
        # special MdDropout: dropout the entire last dim (for word, char, extras, but not posi)
        self.dropmd_word = self.add_sub_node("md", DropoutLastN(pc, lastn=1))
        self.dropmd_char = self.add_sub_node("md", DropoutLastN(pc, lastn=1))
        self.dropmd_extras = [self.add_sub_node("md", DropoutLastN(pc, lastn=1)) for _ in self.extra_names]
        # dropouts for aux
        self.drop_auxes = [self.add_sub_node("aux", Dropout(pc, (one_aux_dim,))) for one_aux_dim in self.dim_auxes]

    #
    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    def __repr__(self):
        return "# MyEmbedder: %s -> %s" % (self.repr_sizes, self.output_dim)

    # word_arr: None or [*, seq-len], char_arr: None or [*, seq-len, word-len],
    # extra_arrs: list of [*, seq-len], aux_arrs: list of [*, seq-len, D]
    # todo(warn): no masks in this step?
    def __call__(self, word_arr:np.ndarray=None, char_arr:np.ndarray=None,
                 extra_arrs:Iterable[np.ndarray]=(), aux_arrs: Iterable[np.ndarray]=()):
        exprs = []
        # word/char/extras/posi
        seq_shape = None
        if self.has_word:
            # todo(warn): singleton-UNK-dropout should be done outside before
            seq_shape = word_arr.shape
            word_expr = self.dropmd_word(self.word_embed(word_arr))
            exprs.append(word_expr)
        if self.has_char:
            seq_shape = char_arr.shape[:-1]
            char_embeds = self.char_embed(char_arr)     # [*, seq-len, word-len, D]
            char_cat_expr = self.dropmd_char(BK.concat([z(char_embeds) for z in self.char_cnns]))
            exprs.append(char_cat_expr)
        zcheck(len(extra_arrs)==len(self.extra_embeds), "Unmatched extra fields.")
        for one_extra_arr, one_extra_embed, one_extra_dropmd in zip(extra_arrs, self.extra_embeds, self.dropmd_extras):
            seq_shape = one_extra_arr.shape
            exprs.append(one_extra_dropmd(one_extra_embed(one_extra_arr)))
        if self.has_posi:
            seq_len = seq_shape[-1]
            posi_idxes = BK.arange_idx(seq_len)
            posi_input0 = self.posi_embed(posi_idxes)
            for _ in range(len(seq_shape)-1):
                posi_input0 = BK.unsqueeze(posi_input0, 0)
            posi_input1 = BK.expand(posi_input0, tuple(seq_shape)+(-1,))
            exprs.append(posi_input1)
        #
        assert len(aux_arrs) == len(self.drop_auxes)
        for one_aux_arr, one_aux_dim, one_aux_drop, one_fold, one_gamma, one_lambdas in \
                zip(aux_arrs, self.dim_auxes, self.drop_auxes, self.fold_auxes, self.aux_overall_gammas, self.aux_fold_lambdas):
            # fold and apply trainable lambdas
            input_aux_repr = BK.input_real(one_aux_arr)
            input_shape = BK.get_shape(input_aux_repr)
            # todo(note): assume the original concat is [fold/layer, D]
            reshaped_aux_repr = input_aux_repr.view(input_shape[:-1]+[one_fold, one_aux_dim])  # [*, slen, fold, D]
            lambdas_softmax = BK.softmax(one_gamma, -1).unsqueeze(-1)  # [fold, 1]
            weighted_aux_repr = (reshaped_aux_repr * lambdas_softmax).sum(-2) * one_gamma  # [*, slen, D]
            one_aux_expr = one_aux_drop(weighted_aux_repr)
            exprs.append(one_aux_expr)
        #
        concated_exprs = BK.concat(exprs, dim=-1)
        # optional proj
        if self.has_proj:
            final_expr = self.final_layer(concated_exprs)
        else:
            final_expr = concated_exprs
        return final_expr
