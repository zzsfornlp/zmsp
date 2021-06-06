#

__all__ = [
    "PlainEncoderConf", "PlainEncoder",
]

from typing import Callable, List, Dict, Set, Tuple
from collections import OrderedDict
import numpy as np
from msp2.nn import BK
from msp2.nn.layers.base import *
from msp2.nn.layers.enc import *
from msp2.nn.layers import PosiEmbeddingNode, PosiEmbeddingConf, LayerNormNode, LayerNormConf

# PlainEncoder
class PlainEncoderConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.input_dim: int = -1  # to be filled
        # --
        self.enc_hidden = 512  # hidden dim (for all components)
        self.enc_ordering = ["rnn", "cnn", "att", "tatt"]  # check each one!
        # various encoders: by default no layers!
        self.enc_rnn = RnnConf().direct_update(n_layers=0)
        self.enc_cnn = CnnConf().direct_update(n_layers=0)
        self.enc_att = TransformerConf().direct_update(n_layers=0)
        self.enc_tatt = TTransformerConf().direct_update(n_layers=0)

@node_reg(PlainEncoderConf)
class PlainEncoder(MultiLayerEncNode):
    def __init__(self, conf: PlainEncoderConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PlainEncoderConf = self.conf
        # --
        output_dim = conf.input_dim
        self.layers = OrderedDict()  # ordered!!
        for name in conf.enc_ordering:
            assert name not in self.layers, "Repeated names!"
            if name == "rnn":
                if conf.enc_rnn.n_layers > 0:
                    node = RnnNode(conf.enc_rnn, isize=output_dim, osize=conf.enc_hidden)
                else:
                    node = None
            elif name == "cnn":
                if conf.enc_cnn.n_layers > 0:
                    node = CnnNode(conf.enc_cnn, isize=output_dim, osize=conf.enc_hidden)
                else:
                    node = None
            elif name == "att":
                if conf.enc_att.n_layers > 0:
                    assert output_dim == conf.enc_hidden
                    node = TransformerNode(conf.enc_att, isize=output_dim, osize=output_dim)
                else:
                    node = None
            elif name == "tatt":
                if conf.enc_tatt.n_layers > 0:
                    assert output_dim == conf.enc_hidden
                    node = TTransformerNode(conf.enc_tatt, isize=output_dim, osize=output_dim)
                else:
                    node = None
            else:
                raise NotImplementedError(f"Unknown enc-name {name}")
            if node is not None:  # if use this node!
                output_dim = node.get_output_dims((output_dim, ))[0]  # update
                self.layers[name] = node
                self.add_module(f"_M{name}", node)
        self.output_dim = output_dim

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    def extra_repr(self) -> str:
        descriptions = [f"{name}({node.conf.n_layers})" for name, node in self.layers.items()]
        return f"ENC({descriptions})"

    @property
    def num_layers(self):  # sum all
        return sum(s.num_layers for s in self.layers.values())

    def forward(self, input_expr, mask_expr=None, vstate: VrecSteppingState=None, **kwargs):
        cur_expr = input_expr
        for name, node in self.layers.items():
            cur_expr = node(cur_expr, mask_expr, vstate=vstate, **kwargs)
        return cur_expr

# =====
# borrow torch.nn.TransformerEncoderLayer
# todo(+N): has problem!!

class TTransformerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # input dim
        self.osize = -1  # hidden/output_dim
        # --
        self.n_layers = 6
        self.use_posi = False  # add posi embeddings at input?
        self.pconf = PosiEmbeddingConf().direct_update(min_val=0)
        self.norm_input = True

    @property
    def d_model(self):
        assert self.isize == self.osize
        return self.isize

@node_reg(TTransformerConf)
class TTransformerNode(BasicNode):
    def __init__(self, conf: TTransformerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: TTransformerConf = self.conf
        # --
        import torch.nn
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # --
        encoder_layers = TransformerEncoderLayer(conf.d_model, 8)
        transformer_encoder = TransformerEncoder(encoder_layers, conf.n_layers)
        self.enc = ModuleWrapper(transformer_encoder, None)
        self.enc.to(BK.DEFAULT_DEVICE)
        if conf.use_posi:
            self.PE = PosiEmbeddingNode(conf.pconf, osize=conf.d_model)
        if conf.norm_input:
            self.norm = LayerNormNode(None, osize=conf.d_model)

    def get_output_dims(self, *input_dims):
        return (self.conf.d_model, )

    def forward(self, input_expr, mask_expr=None, **kwargs):
        conf: TTransformerConf = self.conf
        # --
        if conf.n_layers == 0:
            return input_expr  # change nothing if no layers
        # --
        if conf.use_posi:
            ssize = BK.get_shape(input_expr, 1)  # step size
            posi_embed = self.PE(BK.arange_idx(ssize)).unsqueeze(0)  # [1, step, D]
            input_x = input_expr + posi_embed
        else:
            input_x = input_expr
        if conf.norm_input:
            input_x = self.norm(input_x)
        output = self.enc(input_x.transpose(0,1), src_key_padding_mask=(mask_expr>0)).transpose(0,1).contiguous()
        return output
