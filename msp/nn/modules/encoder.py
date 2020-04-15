#
from typing import List

# shared encoder for various parsing methods
from msp.utils import Conf, zfatal, zcheck, zwarn, zlog
from msp.nn import BK, layers
from msp.nn.layers import BasicNode, RnnLayer, RnnLayerBatchFirstWrapper, CnnLayer, TransformerEncoder, \
    Transformer2Encoder, AttConf, Sequential, Dropout

# conf
class EncConf(Conf):
    def __init__(self):
        self._input_dim = -1            # to be filled
        self.enc_hidden = 400           # concat-dimension
        self.enc_ordering = ["rnn","cnn","att","att2"]     # default short-range to long-range
        self.no_final_dropout = False  # disable dropout for the final layer of this module
        # various encoders
        # rnn
        self.enc_rnn_type = "lstm2"
        self.enc_rnn_layer = 1
        self.enc_rnn_bidirect = True
        self.enc_rnn_sep_bidirection = False
        # cnn
        self.enc_cnn_windows = [3, 5]   # split dim by windows
        self.enc_cnn_layer = 0
        # att(basic)
        self.enc_att_layer = 0
        self.enc_att_conf = AttConf()
        self.enc_att_add_wrapper = "addnorm"
        self.enc_att_ff = 512
        self.enc_att_fixed_ranges = []  # should sth like 2,4,8,16,32,64
        self.enc_att_final_act = "linear"
        # reuse some of the options in original att
        self.enc_att2_layer = 0
        self.enc_att2_conf = AttConf()
        self.enc_att2_short_range = 3
        self.enc_att2_long_ranges = []          # similar to enc_att_fixed_ranges

    def do_validate(self):
        def _res(rs, checked_length):
            if rs is None or len(rs) == 0:
                return None
            else:
                assert len(rs) == checked_length
                return [int(x) for x in rs]
        # make sure the ranges are ok!
        self.enc_att_fixed_ranges = _res(self.enc_att_fixed_ranges, self.enc_att_layer)
        self.enc_att2_long_ranges = _res(self.enc_att2_long_ranges, self.enc_att2_layer)

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
                    rnn_bidirect, rnn_sep_bidirection = econf.enc_rnn_bidirect, econf.enc_rnn_sep_bidirection
                    rnn_enc_size = self.enc_hidden//2 if rnn_bidirect else self.enc_hidden
                    rnn_layer = self.add_sub_node("rnn", RnnLayerBatchFirstWrapper(pc, RnnLayer(pc, last_dim, rnn_enc_size, econf.enc_rnn_layer, node_type=econf.enc_rnn_type, bidirection=rnn_bidirect, sep_bidirection=rnn_sep_bidirection)))
                    self.layers.append(rnn_layer)
            # todo(+2): different i/o sizes for cnn and att?
            elif name == "cnn":
                if econf.enc_cnn_layer > 0:
                    per_cnn_size = self.enc_hidden // len(econf.enc_cnn_windows)
                    cnn_layer = self.add_sub_node("cnn", Sequential(pc, [CnnLayer(pc, last_dim, per_cnn_size, econf.enc_cnn_windows, act="elu") for _ in range(econf.enc_cnn_layer)]))
                    self.layers.append(cnn_layer)
            elif name == "att":
                if econf.enc_att_layer > 0:
                    zcheck(last_dim == self.enc_hidden, "I/O should have same dim for Att-Enc")
                    att_layer = self.add_sub_node("att", TransformerEncoder(pc, econf.enc_att_layer, last_dim, econf.enc_att_ff, econf.enc_att_add_wrapper, econf.enc_att_conf, final_act=econf.enc_att_final_act, fixed_range_vals=econf.enc_att_fixed_ranges))
                    self.layers.append(att_layer)
            elif name == "att2":
                if econf.enc_att2_layer > 0:
                    zcheck(last_dim == self.enc_hidden, "I/O should have same dim for Att-Enc")
                    att2_layer = self.add_sub_node("att2", Transformer2Encoder(pc, econf.enc_att2_layer, last_dim, econf.enc_att2_conf, short_range=econf.enc_att2_short_range, long_ranges=econf.enc_att2_long_ranges))
                    self.layers.append(att2_layer)
            else:
                zfatal("Unknown encoder name: "+name)
            if len(self.layers) > 0:
                last_dim = self.layers[-1].get_output_dims()[-1]
        self.output_dim = last_dim
        #
        if econf.no_final_dropout:
            self.disable_final_dropout()

    def __repr__(self):
        return "# MyEncoder: %s -> %s [%s]" % (self.input_dim, self.output_dim, ", ".join([str(z) for z in self.layers]))

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # T[*, seq-len, D], arr[*, seq-len] or None
    def __call__(self, embeds_expr, word_mask_arr, return_all_layers=False):
        v = embeds_expr
        all_layers = []  # not including input!!!
        for one_node in self.layers:
            v = one_node(v, word_mask_arr)
            all_layers.append(v)
        if return_all_layers:
            return v, all_layers
        else:
            return v

    # todo(+2): specific for every type!
    def disable_final_dropout(self):
        if len(self.layers) < 1:
            zwarn("Cannot disable final dropout since this Enc layer is empty!!")
        else:
            # get the final one from sequential
            final_layer = self.layers[-1]
            while isinstance(final_layer, Sequential):
                final_layer = final_layer.ns_[-1] if len(final_layer.ns_) else None
            # get final dropout node
            final_drop_node: Dropout = None
            if isinstance(final_layer, RnnLayerBatchFirstWrapper):
                final_drop_nodes = final_layer.rnn_node.drop_nodes
                if final_drop_nodes is not None and len(final_drop_nodes)>0:
                    final_drop_node = final_drop_nodes[-1]
            elif isinstance(final_layer, CnnLayer):
                final_drop_node = final_layer.drop_node
            elif isinstance(final_layer, TransformerEncoder):
                pass  # todo(note): final is LayerNorm?
            if final_drop_node is None:
                zwarn(f"Failed at disabling final enc-layer dropout: type={type(final_layer)}: {final_layer}")
            else:
                final_drop_node.rop.add_fixed_value("hdrop", 0.)
                zlog(f"Ok at disabling final enc-layer dropout: type={type(final_layer)}: {final_layer}")
