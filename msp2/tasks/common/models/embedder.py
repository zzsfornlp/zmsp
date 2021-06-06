#

from typing import List, Dict, Set
from collections import OrderedDict
from msp2.nn import BK
from msp2.nn.layers import BasicConf, BasicNode, AffineConf, AffineNode, LayerNormNode, SequentialNode
from msp2.nn.modules.embedder import *
from msp2.nn.modules.berter import *
from msp2.data.vocab import VocabPackage

class MyEmbedderConf(BasicConf):
    def __init__(self):
        super().__init__()
        # embedders
        self.eg = EmbedderGroupConf()
        self.ec_word = PlainInputEmbedderConf()
        self.ec_lemma = PlainInputEmbedderConf()
        self.ec_upos = PlainInputEmbedderConf()
        self.ec_char = CharCnnInputEmbedderConf()
        self.ec_posi = PosiInputEmbedderConf()
        self.ec_bert = BertInputEmbedderConf()
        # final projection layer
        self.eproj_dim = 0  # 0 means no final proj
        self.eproj_norm = False  # further add layer norm after eproj
        self.eproj = AffineConf()

    def _do_validate(self):
        self.eproj.n_dim = self.eproj_dim

    @property
    def ec_dict(self):
        return OrderedDict([
            ('word', self.ec_word), ('lemma', self.ec_lemma), ('upos', self.ec_upos),
            ('char', self.ec_char), ('posi', self.ec_posi), ('bert', self.ec_bert),
        ])

class MyEmbdder(BasicNode):
    def __init__(self, conf: MyEmbedderConf, vpack: VocabPackage, berter: BertEncoder = None):
        super().__init__(conf)
        conf: MyEmbedderConf = self.conf
        # --
        self.vpakc = vpack
        self.ig = InputterGroup()  # inputter group
        self.eg = EmbedderGroup(conf.eg)  # embedder group
        # components
        self.comp_names = []
        self.comp_dims = []
        for one_name, one_conf in conf.ec_dict.items():
            if one_conf.dim > 0:  # also act as a switch!
                if isinstance(one_conf, CharCnnInputEmbedderConf):  # first check this one!!
                    one_voc, one_npvec = vpack.get_voc(one_name), vpack.get_emb(one_name)
                    one_inputter = CharInputHelper("word", one_voc)  # todo(note): char input comes from word
                    one_embedder = CharCnnInputEmbedderNode(one_conf, one_voc, npvec=one_npvec, name=one_name)
                elif isinstance(one_conf, PlainInputEmbedderConf):  # basic ones
                    one_voc, one_npvec = vpack.get_voc(one_name), vpack.get_emb(one_name)
                    one_inputter = PlainInputHelper(one_name, one_voc)
                    one_embedder = PlainInputEmbedderNode(one_conf, one_voc, npvec=one_npvec, name=one_name)
                elif isinstance(one_conf, PosiInputEmbedderConf):  # no need vocab
                    one_inputter = PosiInputHelper(one_name)
                    one_embedder = PosiInputEmbedderNode(one_conf, name=one_name)
                elif isinstance(one_conf, BertInputEmbedderConf):
                    one_inputter = BertInputHelper("word", berter)  # todo(note): bert's subword input comes from word
                    one_embedder = BertInputEmbedderNode(berter, one_conf, one_name)
                else:
                    raise NotImplementedError(f"UNK emb-conf of {type(one_conf)}")
                self.ig.register_inputter(one_name, one_inputter)
                self.eg.register_embedder(one_name, one_embedder, one_name)
                self.comp_names.append(one_name)
                self.comp_dims.append(one_embedder.get_output_dims()[0])
        # final output
        self.has_proj = (conf.eproj_dim > 0)
        if self.has_proj:
            proj_layer = AffineNode(conf.eproj, isize=sum(self.comp_dims), osize=conf.eproj_dim)
            if conf.eproj_norm:
                norm_layer = LayerNormNode(None, osize=conf.eproj_dim)
                self.final_layer = SequentialNode([proj_layer, norm_layer], None)
            else:
                self.final_layer = proj_layer
            self.output_dim = conf.eproj_dim
        else:
            self.final_layer = lambda x: x  # concatenate
            self.output_dim = sum(self.comp_dims)

    def extra_repr(self) -> str:
        nds = [f'{n}[{d}]' for n,d in zip(self.comp_names, self.comp_dims)]
        return f"MyEmbedder({','.join(nds)}->{self.output_dim})"

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # =====
    # specific ones

    def run_inputs(self, insts: List): return self.ig.prepare_inputs(insts)
    def run_mask_inputs(self, input_map: Dict, input_erase_mask: BK.Expr, nomask_names_set: Set = None):
        return self.ig.mask_inputs(input_map, input_erase_mask, nomask_names_set)
    def run_embeds(self, input_map: Dict): return self.forward(input_map)

    def forward(self, input_map: Dict):
        mask_expr, expr_map = self.eg.forward(input_map)
        exprs = list(expr_map.values())  # follow the order in OrderedDict
        # concat and final
        concat_expr = BK.concat(exprs, -1)  # [*, len, SUM]
        final_expr = self.final_layer(concat_expr)
        return mask_expr, final_expr
