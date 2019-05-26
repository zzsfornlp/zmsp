#
from typing import List

# shared encoder for various parsing methods
from msp.utils import Conf, zfatal, zcheck
from msp.nn import BK, layers
from msp.nn.layers import BasicNode, RnnLayer, RnnLayerBatchFirstWrapper, CnnLayer, TransformerEncoder

# conf
class EncConf(Conf):
    def __init__(self):
        self._input_dim = -1            # to be filled
        self.enc_hidden = 400           # concat-dimension
        self.enc_ordering = ["rnn","cnn","att"]     # default short-range to long-range
        # various encoders
        self.enc_rnn_type = "lstm"
        self.enc_rnn_layer = 1
        self.enc_rnn_bidirect = True
        self.enc_cnn_windows = [3, 5]   # split dim by windows
        self.enc_cnn_layer = 0
        self.enc_att_rel_clip = 0       # relative positional reprs if >0
        self.enc_att_rel_neg = True     # use neg for rel-posi
        self.enc_att_ff = 512
        self.enc_att_layer = 0
        self.enc_att_dropout = 0.1      # special attention dropout
        self.enc_att_add_wrapper = "addnorm"
        self.enc_att_type = "mh"        # multi-head
        self.enc_attf_selfw = -100.
        self.enc_att_use_ranges = False

# various kinds of encoders
class MyEncoder(BasicNode):
    def __init__(self, pc: BK.ParamCollection, econf: EncConf):
        super().__init__(pc, None, None)
        self.conf = econf
        #
        self.input_dim = econf._input_dim
        self.enc_hidden = econf.enc_hidden
        # add the sublayers
        self.layers = []
        # todo(0): allowing repeated names
        last_dim = self.input_dim
        for name in econf.enc_ordering:
            if name == "rnn":
                if econf.enc_rnn_layer > 0:
                    rnn_bidirect = econf.enc_rnn_bidirect
                    rnn_enc_size = self.enc_hidden//2 if rnn_bidirect else self.enc_hidden
                    rnn_layer = self.add_sub_node("rnn", RnnLayerBatchFirstWrapper(pc, RnnLayer(pc, last_dim, rnn_enc_size, econf.enc_rnn_layer, node_type=econf.enc_rnn_type, bidirection=rnn_bidirect)))
                    self.layers.append(rnn_layer)
            elif name == "cnn":
                if econf.enc_cnn_layer > 0:
                    per_cnn_size = self.enc_hidden // len(econf.enc_cnn_windows)
                    cnn_layer = self.add_sub_node("cnn", layers.Sequential(pc, [CnnLayer(pc, last_dim, per_cnn_size, econf.enc_cnn_windows, act="relu") for _ in range(econf.enc_cnn_layer)]))
                    self.layers.append(cnn_layer)
            elif name == "att":
                if econf.enc_att_layer > 0:
                    zcheck(last_dim == self.enc_hidden, "I/O should have same dim for Att-Enc")
                    att_layer = self.add_sub_node("att", TransformerEncoder(pc, econf.enc_att_layer, last_dim, econf.enc_att_ff, att_type=econf.enc_att_type, add_wrapper=econf.enc_att_add_wrapper, att_dropout=econf.enc_att_dropout, att_use_ranges=econf.enc_att_use_ranges, attf_selfw=econf.enc_attf_selfw, clip_dist=econf.enc_att_rel_clip, use_neg_dist=econf.enc_att_rel_neg))
                    self.layers.append(att_layer)
            else:
                zfatal("Unknown encoder name: "+name)
            if len(self.layers) > 0:
                last_dim = self.layers[-1].get_output_dims()[-1]
        self.output_dim = last_dim

    def __repr__(self):
        return "# MyEncoder: %s -> %s [%s]" % (self.input_dim, self.output_dim, ", ".join([str(z) for z in self.layers]))

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # T[*, seq-len, D], arr[*, seq-len] or None
    def __call__(self, embeds_expr, word_mask_arr):
        v = embeds_expr
        for one_node in self.layers:
            v = one_node(v, word_mask_arr)
        return v
