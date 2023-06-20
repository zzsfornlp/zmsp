#

# for embeddings

__all__ = [
    "EmbedConf", "EmbedLayer", "LmheadConf", "LmheadLayer", "ExtraEmbedConf", "ExtraEmbedLayer",
]

from typing import Union
from math import log as math_log
from mspx.utils import zlog
from ..backends import BK
from .base import *

@NnConf.rd('emb')
class EmbedConf(NnConf):
    def __init__(self):
        super().__init__()
        self.mdim = -1
        self.vocab_size = -1
        self.dropout = 0.1
        self.use_ln = True  # use layernorm?
        self.max_abs_posi = 0  # use absolute position, <=0 if not-use
        self.posi_offset0 = 0  # special offset (NOT add to posi-emb-shape)!
        self.posi_offset1 = 0  # special offset (add to posi-emb-shape)!
        self.type_size = 0  # type_vocab, <=0 if not-use
        self.layer_norm_eps = 1e-5  # same as pytorch's default

@EmbedConf.conf_rd()
class EmbedLayer(NnLayer):
    def __init__(self, conf: EmbedConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: EmbedConf = self.conf
        # --
        self.word_embeddings = BK.nn.Embedding(conf.vocab_size, conf.mdim)
        self.position_embeddings = BK.nn.Embedding(conf.max_abs_posi+conf.posi_offset1, conf.mdim) if conf.max_abs_posi>0 else None
        self.token_type_embeddings = BK.nn.Embedding(conf.type_size, conf.mdim) if conf.type_size>0 else None
        self.layer_norm = BK.nn.LayerNorm(conf.mdim, eps=conf.layer_norm_eps) if conf.use_ln else (lambda x: x)
        self.dropout = BK.nn.Dropout(conf.dropout)
        self.posi_offset = conf.posi_offset0 + conf.posi_offset1
        # --

    def extra_repr(self) -> str:
        return f"Embed[{self.word_embeddings}]"

    def get_output_dims(self, *input_dims):
        return (self.conf.mdim, )

    def forward_emb(self, t_ids):  # only forward word embeddings
        return self.word_embeddings(t_ids)

    def forward(self, t_ids, t_posi_ids=None, t_ttype_ids=None, t_emb=None, cache=None):
        # first tok-id embeddings!
        if t_emb is not None:
            assert t_ids is None
            cur_embeds = t_emb  # [*, L, D]
        else:
            cur_embeds = self.word_embeddings(t_ids)  # [*, L, D]
        emb_shape = BK.get_shape(cur_embeds)
        # add position embeddings
        if self.position_embeddings is not None:
            _cur_size = emb_shape[-2]
            _offset = self.posi_offset
            _step = _offset if cache is None else cache.get('step', _offset)
            if t_posi_ids is None:  # by default, simply range(len)
                t_posi_ids = BK.arange_idx(_cur_size) + _step
            if cache is not None:
                cache['step'] = _step + _cur_size
            position_embeddings = self.position_embeddings(t_posi_ids)
            cur_embeds = cur_embeds + position_embeddings
        # add type embeddings
        if self.token_type_embeddings is not None:
            if t_ttype_ids is None:
                t_ttype_ids = BK.constants_idx(emb_shape[:-1], 0)  # by default zeros
            token_type_embeddings = self.token_type_embeddings(t_ttype_ids)
            cur_embeds = cur_embeds + token_type_embeddings
        # output
        cur_embeds = self.layer_norm(cur_embeds)
        cur_embeds = self.dropout(cur_embeds)
        return cur_embeds

    def tie_embeddings(self, emb: Union['EmbedLayer', BK.nn.Parameter]):
        if isinstance(emb, EmbedLayer):
            emb = emb.word_embeddings.weight
        assert isinstance(emb, BK.nn.Parameter)
        self.word_embeddings.weight = emb  # note: directly set it!

# --
# note: from "transformers.RobertaLMHead"

@NnConf.rd('lmhead')
class LmheadConf(NnConf):
    def __init__(self):
        super().__init__()
        self.mdim = -1  # input
        self.odim = -1  # output
        self.use_hid = True
        self.hid_act = 'gelu'
        self.use_ln = True
        self.layer_norm_eps = 1e-5  # same as pytorch's default
        self.final_bias = True

@LmheadConf.conf_rd()
class LmheadLayer(NnLayer):
    def __init__(self, conf: LmheadConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: LmheadConf = self.conf
        # --
        mdim = conf.mdim
        self.dense = self.layer_norm = None
        self.hid_act = ActivationHelper.get_act(conf.hid_act)
        if conf.use_hid:
            self.dense = BK.nn.Linear(mdim, mdim)
            if conf.use_ln:
                self.layer_norm = BK.nn.LayerNorm(mdim, eps=conf.layer_norm_eps)
        self.decoder = BK.nn.Linear(mdim, conf.odim, bias=conf.final_bias)
        if self.decoder.bias is not None:
            with BK.no_grad_env():
                self.decoder.bias.zero_()  # note: by default set it zero!
        # --

    def extra_repr(self) -> str:
        conf: LmheadConf = self.conf
        return f"Lmhead[{conf.mdim}, {conf.odim}]"

    def get_output_dims(self, *input_dims):
        return (self.conf.odim, )

    def forward(self, x):
        if self.dense is not None:
            x = self.dense(x)
            x = self.hid_act(x)
            if self.layer_norm is not None:
                x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def tie_embeddings(self, emb: Union[EmbedLayer, BK.nn.Parameter]):
        if isinstance(emb, EmbedLayer):
            emb = emb.word_embeddings.weight
        assert isinstance(emb, BK.nn.Parameter)
        self.decoder.weight = emb  # note: directly set it!

# --
# a layer for extra mixing embs

@NnConf.rd('ememb')
class ExtraEmbedConf(NnConf):
    def __init__(self):
        super().__init__()
        self.mdim = -1
        self.osizes = []  # vocab sizes
        # self.mixing_init = []  # init mixing weight
        # self.mixing_tune = True  # make it tunable?
        self.init_scale = 1.  # scale for init embeddings
        self.mixing_weight = 1.  # mixing weight
        self.ues_ln = True  # use layer norm?
        self.drop0 = 0.  # "drop" ids to 0 (which usually indicate NIL!)

@ExtraEmbedConf.conf_rd()
class ExtraEmbedLayer(NnLayer):
    def __init__(self, conf: ExtraEmbedConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ExtraEmbedConf = self.conf
        # --
        _mdim = conf.mdim
        _osizes = [int(z) for z in conf.osizes]
        zlog(f"Init ExtraEmb with {_osizes}")
        self.mixing_weight = conf.mixing_weight
        self.emb_names = []
        for idx, _osize in enumerate(_osizes):
            self.add_module(f"E{idx}", BK.get_emb_with_initscale(_osize, _mdim, initscale=conf.init_scale))
            self.emb_names.append(f"E{idx}")
        if conf.ues_ln and len(self.emb_names) > 0:  # nothing to do if no inputs
            self.layer_norm = BK.nn.LayerNorm(_mdim)  # final layer-norm
        else:
            self.layer_norm = lambda x: x
        # --

    # @staticmethod
    # def inverse_sigmoid(x):
    #     x = min(max(x, 1e-5), (1 - 1e-5))
    #     ret = math_log(x/(1-x))
    #     return ret

    @property
    def embs(self):
        return [getattr(self, k) for k in self.emb_names]

    def forward(self, t_base, extra_ids):
        _drop0 = self.conf.drop0
        # --
        cur_t = t_base
        embs = self.embs
        all_extras = []
        for ii, t_ids in enumerate(extra_ids):
            if self.training and _drop0 > 0.:
                _mask = BK.random_bernoulli(t_ids.shape, p=(1-_drop0), mul=1).to(t_ids)
                t_ids = t_ids * _mask
            all_extras.append(embs[ii](t_ids))
        if len(all_extras) > 0:
            t_extra = BK.stack(all_extras, -1).sum(-1)  # [...]
            cur_t = cur_t + t_extra * self.mixing_weight
        ret = self.layer_norm(cur_t)
        return ret
