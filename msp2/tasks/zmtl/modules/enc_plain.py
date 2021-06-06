#

# plain emb+enc

__all__ = [
    "ZEncoderPlainConf", "ZEncoderPlain", "MyTransformerConf", "MyTransformer",
]

from typing import List
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.tasks.common.models.embedder import MyEmbdder, MyEmbedderConf
from .common import *
from .enc import *

# --

class ZEncoderPlainConf(ZEncoderConf):
    def __init__(self):
        super().__init__()
        # --
        self.econf = MyEmbedderConf()  # embedding
        self.tconf = MyTransformerConf()  # encoding
        self.inc_cls = False

@node_reg(ZEncoderPlainConf)
class ZEncoderPlain(ZEncoder):
    def __init__(self, conf: ZEncoderPlainConf, vpack, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        self.emb = MyEmbdder(conf.econf, vpack, berter=None)  # this one get no bert!!
        self.enc = MyTransformer(conf.tconf)
        # todo(+W): special change!!
        self.emb.eg.conf.add_bos = conf.inc_cls
        # --
        raise RuntimeError("Deprecated after MED's collecting of scores!!")

    def forward(self, insts: List, med: ZMediator):
        # first get embeddings
        input_map = self.emb.run_inputs([z.sent for z in insts])
        mask_expr, emb_expr = self.emb.run_embeds(input_map)  # [bs, slen, ?]
        # then simply run enc
        med.restart(insts=insts, mask_t=mask_expr)
        ret = self.enc.forward(emb_expr, mask_expr, med)
        # med.restart()  # not clear here!
        return ret

    # info
    def get_enc_dim(self) -> int: return self.enc.conf.enc_dim
    def get_head_dim(self) -> int: return self.enc.conf.aconf.nh_qk

# --
class MyTransformerConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.enc_dim = 512
        self.n_layers = 8
        self.use_posi = False  # add posi embeddings at input?
        self.pconf = PosiEmbeddingConf().direct_update(min_val=0)
        # for each layer
        self.aconf = AttentionPlainConf()
        self.d_ff = 1024  # dim of FF, 0 for skipping ff
        self.ff_act = "relu"

class _OneTNode(BasicNode):
    def __init__(self, conf: MyTransformerConf):
        super().__init__(None)  # simply borrowing conf
        # --
        d_model = conf.enc_dim
        self.att = AttentionPlainNode(conf.aconf, dim_q=d_model, dim_k=d_model, dim_v=d_model)
        self.norm0 = LayerNormNode(None, osize=d_model)
        if conf.d_ff > 0:
            ff_node = get_mlp(d_model, d_model, conf.d_ff, 1, AffineConf().direct_update(out_act=conf.ff_act), AffineConf())
            self.feed_forward = WrapperNode(ff_node, None, isize=d_model, strategy="addnorm")
        else:
            self.feed_forward = lambda x: x
        self.norm1 = LayerNormNode(None, osize=d_model)  # extra norm for adding extra features
        # --

    def forward(self, input_expr, mask_expr, med: ZMediator, **kwargs):
        # get attn scores
        scores_t = self.att.do_score(input_expr, input_expr)  # [*, Hin, lenq, lenk]
        # context
        context = self.att.do_output(scores_t, input_expr, mask_k=mask_expr)
        att_output = self.norm0(context + input_expr)  # layer norm
        # ff
        ff_output = self.feed_forward(att_output)
        return ff_output, scores_t

@node_reg(MyTransformerConf)
class MyTransformer(BasicNode):
    def __init__(self, conf: MyTransformerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MyTransformerConf = self.conf
        # --
        self.tnodes = []
        for i in range(conf.n_layers):
            one_node = _OneTNode(conf)
            self.add_module(f"T{i}", one_node)
            self.tnodes.append(one_node)
        # posi embeddings?
        if conf.use_posi:
            self.PE = PosiEmbeddingNode(conf.pconf, osize=conf.enc_dim)
        self.input_norm = LayerNormNode(None, osize=conf.enc_dim)

    def forward(self, input_expr, mask_expr, med: ZMediator):
        conf: MyTransformerConf = self.conf
        # for the L0 layer
        cur_expr = input_expr
        if conf.use_posi:
            ssize = BK.get_shape(input_expr, 1)  # step size
            posi_embed = self.PE(BK.arange_idx(ssize)).unsqueeze(0)  # [1, step, D]
            cur_expr = cur_expr + posi_embed
        add_expr, _ = med.layer_end(cur_expr, None)  # no checking at L0
        if add_expr is not None:
            cur_expr += add_expr  # norm right later!
        cur_expr = self.input_norm(cur_expr)
        # for later layers: L1+
        for ti, tnode in enumerate(self.tnodes):
            cur_expr, scores_t = tnode(cur_expr, mask_expr, med)
            add_expr, early_exit = med.layer_end(cur_expr, scores_t)  # check
            if add_expr is not None:
                cur_expr = tnode.norm1(cur_expr + add_expr)
            if early_exit:
                break
        return cur_expr, {}

# --
