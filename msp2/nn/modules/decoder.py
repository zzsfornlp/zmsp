#

__all__ = [
    "PlainDecoderConf", "PlainDecoder", "PlainDecoderCache",
]

from typing import List, Tuple, Union, Iterable
from collections import OrderedDict
from msp2.nn import BK
from msp2.nn.layers.base import *
from msp2.nn.layers.dec import *

# PlainDecoder
class PlainDecoderConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.input_dim: int = -1  # to be filled
        # --
        self.dec_hidden = 512  # hidden dim (for all components)
        self.dec_ordering = ["rnn", "att"]  # check each one
        # various decoders
        self.dec_rnn = RnnDecConf().direct_update(n_layers=0)
        self.dec_att = TransformerDecConf().direct_update(n_layers=0)

class PlainDecoderCache(DecCache):
    def __init__(self, sub_caches: Iterable[DecCache]):
        self.sub_caches: List[DecCache] = list(sub_caches)

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    # need init with store_last_state
    def get_last_state(self, layer_idx=-1):
        return self.sub_caches[-1].get_last_state(layer_idx)

    def _arrange_idxes(self, idxes: BK.Expr):
        for cache in self.sub_caches:  # simply arrange them all
            cache.arrange_idxes(idxes)

@node_reg(PlainDecoderConf)
class PlainDecoder(BasicNode):
    def __init__(self, conf: PlainDecoderConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PlainDecoderConf = self.conf
        # --
        output_dim = conf.input_dim
        self.layers = OrderedDict()  # ordered!!
        for name in conf.dec_ordering:
            assert name not in self.layers, "Repeated names!"
            if name == "rnn":
                node = RnnDecNode(conf.dec_rnn, isize=output_dim, osize=conf.dec_hidden) if conf.dec_rnn.n_layers>0 else None
            elif name == "att":
                node = TransformerDecNode(conf.dec_att, isize=output_dim, osize=conf.dec_hidden) if conf.dec_att.n_layers>0 else None
            else:
                raise NotImplementedError(f"Unknown dec-name {name}")
            # --
            if node is None:
                continue
            # --
            output_dim = node.get_output_dims((output_dim, ))[0]  # update
            self.layers[name] = node
            self.add_module(f"_M{name}", node)
        self.output_dim = output_dim

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    def extra_repr(self) -> str:
        descriptions = [f"{name}({node.conf.n_layers})" for name, node in self.layers.items()]
        return f"DEC({descriptions})"

    @property
    def num_layers(self):  # sum all
        return sum(s.num_layers for s in self.layers.values())

    # =====
    # init cache with starting ones or None for empty one!
    def go_init(self, init_hs: Union[List[BK.Expr], BK.Expr], init_mask: BK.Expr = None, **kwargs):
        sub_caches = [layer.go_init(init_hs, init_mask, **kwargs) for layer in self.layers.values()]
        return PlainDecoderCache(sub_caches)

    # continue on cache, the inputs/outputs both have a step dim, even for one step!
    def go_feed(self, cache: PlainDecoderCache, input_expr: BK.Expr, mask_expr: BK.Expr = None):
        cur_expr = input_expr
        layer_idx = 0
        for name, node in self.layers.items():
            cur_expr = node.go_feed(cache.sub_caches[layer_idx], cur_expr, mask_expr)
            layer_idx += 1
        return cur_expr
