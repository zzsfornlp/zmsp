#

# my own version of bmod (emb+enc+dec+lmhead)

__all__ = [
    "ZBmod2Conf", "ZBmod2Mod",
]

import os
from collections import OrderedDict
from typing import List, Dict
from mspx.utils import zglob1, zlog, ConfEntryCallback, zwarn
from mspx.data.vocab import Vocab, Toker, TokerPretrained
from mspx.nn import BK
from ...layers import *
from .base import *
from .bmod1 import ZBmod1Helper

@NnConf.rd('bmod2')
class ZBmod2Conf(ZBmodBaseConf):
    def __init__(self):
        super().__init__()
        # -- vocab
        self.b_vpath = ""  # file-path or bert's name
        self.b_vpath_trg = None  # file-path or bert's name (for trg) (same as enc if empty)
        # -- components
        self.b_mdim = -1  # overall dim
        self.emb = EmbedConf()  # src (or default)
        self.emb_trg = EmbedConf()  # trg
        self.enc = ConfEntryCallback((lambda s: self.callback_entry(s, T=NnConf)), default_s='tf')
        self.dec = ConfEntryCallback((lambda s: self.callback_entry(s, T=NnConf, use_cross=True)), default_s='tf')
        self.lmhead = LmheadConf()
        # -- special init
        self.sconf = ConfEntryCallback(lambda s: ZBmod2Helper.ff_sconf(s, conf=self))  # special conf setting!
        self.init_with_bmodel = ''  # load weights from pre-trained model?
        self.tie_embs2dec = False  # enc -> dec
        self.tie_embs2lmhead = False  # dec(or enc) -> lmhead
        self.b_init_range = -1.  # valid if >0

@ZBmod2Conf.conf_rd()
class ZBmod2Mod(ZBmodBaseMod):
    def __init__(self, conf: ZBmod2Conf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZBmod2Conf = self.conf
        # --
        self.toker, self.toker_trg = ZBmod2Mod.load_tokers(conf)
        _mdim = conf.b_mdim
        self.emb = self.emb_trg = self.enc = self.dec = self.lmhead = None
        if conf.b_inc_enc:
            self.emb = conf.emb.make_node(mdim=_mdim, vocab_size=len(self.toker)) if conf.b_inc_emb else None
            self.enc = conf.enc.make_node(mdim=_mdim)
        if conf.b_inc_dec:
            self.emb_trg = conf.emb.make_node(mdim=_mdim, vocab_size=len(self.toker_trg)) if conf.b_inc_emb else None
            self.dec = conf.dec.make_node(mdim=_mdim)
        if conf.b_inc_lmhead:
            _odim = len(self.toker_trg) if conf.b_inc_dec else len(self.toker)
            self.lmhead = conf.lmhead.make_node(mdim=_mdim, odim=_odim)
        # --
        # init?
        if conf.b_init_range > 0:
            with BK.no_grad_env():
                self.apply(self._init_weights)
        if conf.init_with_bmodel:
            ZBmod2Helper.init_with_bmodel(conf.init_with_bmodel, self, conf)
        if conf.tie_embs2dec and self.emb_trg is not None:
            self.emb_trg.tie_embeddings(self.emb)
        if conf.tie_embs2lmhead and self.lmhead is not None:  # first consider emb_trg
            self.lmhead.tie_embeddings(self.emb if self.emb_trg is None else self.emb_trg)
        # --

    # --
    # note: from transformers
    def _init_weights(self, module):
        _r = float(self.conf.b_init_range)
        """Initialize the weights"""
        if isinstance(module, BK.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=_r)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BK.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=_r)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, BK.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    # --

    # [*, L], [*, L], L([*, L], [*, L, D])
    # typical encoders: BertModel, BartEncoder, T5Stack
    def forward_enc(self, t_ids, t_mask=None, t_emb=None, t_ihid=None):
        if t_ihid is not None:
            cur_hid = t_ihid  # directly using it as input-hid
            if t_ids is not None:
                zwarn(f"Input strange: both t_ids and t_ihid provided!")
        else:
            cur_hid = self.emb(t_ids, t_emb=t_emb)  # emb
        ret = self.enc(cur_hid, t_mask=t_mask, return_dict=True)  # enc
        return ret

    # [*, L], [*, L];; [*, S, D], [*, S];; L([*, L], [*, L, D])
    # typical decoders: GPT2Model, BartDecoder, T5Stack
    def forward_dec(self, t_ids, t_mask=None, t_emb=None, t_ihid=None, t_cross=None, t_cross_mask=None, cache=None):
        if t_ihid is not None:
            cur_hid = t_ihid  # directly using it as input-hid
            if t_ids is not None:
                zwarn(f"Input strange: both t_ids and t_ihid provided!")
        else:
            cur_hid = self.emb_trg(t_ids, t_emb=t_emb, cache=cache)  # emb
        t_mask_qk = make_causual_mask(t_ids)  # note: causal mask!!
        ret = self.dec(cur_hid, t_mask=t_mask, t_mask_qk=t_mask_qk, t_cross=t_cross, t_cross_mask=t_cross_mask,
                       cache=cache, return_dict=True)
        return ret

    def forward_emb(self, t_ids, mixes=None, forw_full=False):
        t_emb = self.emb.forward_emb(t_ids)  # note: use "forward_emb"!
        ret = ZBmodHelper.mix_embs(t_emb, mixes)
        if forw_full:
            ret = self.emb(None, t_emb=ret)
        return ret

    def forward_emb_trg(self, t_ids, mixes=None, forw_full=False, cache=None):
        t_emb = self.emb_trg.forward_emb(t_ids)  # note: use "forward_emb"!
        ret = ZBmodHelper.mix_embs(t_emb, mixes)
        if forw_full:
            ret = self.emb_trg(None, t_emb=t_emb, cache=cache)
        return ret

    def get_mdim(self) -> int: return self.conf.b_mdim
    def get_head_num(self) -> int: return self.conf.enc.tf.aconf.head_count  # todo(+2): could err if not using att!

    # --
    # helpers

    @staticmethod
    def load_toker(vpath: str, extra_tokens, conf: ZBmod2Conf):
        if not vpath:
            return None
        _vpath = zglob1(vpath)
        if os.path.isfile(_vpath):  # first try to load a Vocab from file
            vv = Vocab.create_from_file(_vpath)
            zlog(f"Load {vv} from {_vpath}")
            vv.add_tokens(extra_tokens)
            ret = Toker(vv)
        else:  # otherwise trying to load pretrained one!
            ret = TokerPretrained(vpath, conf.cache_dir_or_none, extra_tokens)
        return ret

    @staticmethod
    def load_tokers(conf: ZBmod2Conf):
        toker = ZBmod2Mod.load_toker(conf.b_vpath, conf.b_extra_tokens, conf)
        if not conf.b_vpath_trg:  # empty then the same
            toker_trg = toker
        else:
            toker_trg = ZBmod2Mod.load_toker(conf.b_vpath_trg, conf.b_extra_tokens_trg, conf)
        return toker, toker_trg

# mainly for helping init with pre-defined ones
class ZBmod2Helper:
    INFO_TABLE = {
        # pretrained ones
        'bert': {
            # conf (directly) set
            'confS': [('b_vpath', None), ('init_with_bmodel', None), ('enc', 'tf'), ('dec', ''), ('b_inc_dec', False), ('tie_embs2lmhead', True)],
            # conf translate
            'confT': [('vocab_size', 'vocab_size'), ('hidden_size', 'b_mdim'), ('num_hidden_layers', 'n_layers'), ('num_attention_heads', 'head_count'), ('hidden_act', ['ff_act', 'hid_act']), ('intermediate_size', 'd_ff'), ('hidden_dropout_prob', 'dropout'), ('attention_probs_dropout_prob', 'att_drop'), ('max_position_embeddings', 'max_abs_posi'), ('type_vocab_size', 'type_size'), ('layer_norm_eps', 'layer_norm_eps'), ('initializer_range', 'b_init_range')],
            # param-name translate (for loading params)
            'paramT': [
                # emb
                ('bert.embeddings.position_ids', ''), ('bert.embeddings', 'emb'), ('LayerNorm', 'layer_norm'),
                # enc
                ('bert.encoder', 'enc'), ('enc.layer.', 'enc.T'), ('self.query', 'affine_q.linear'),
                ('self.key', 'affine_k.linear'), ('self.value', 'affine_v.linear'),
                ('attention.output.dense', 'attention.final_linear.linear'), ('attention.output.layer_norm', 'att_ln'),
                ('intermediate.dense', 'ffn_d1'), ('output.dense', 'ffn_d2'), ('output.layer_norm', 'ffn_ln'),
                # lmhead
                ('cls.predictions.bias', ''), ('cls.predictions.transform', 'lmhead'), ('cls.predictions', 'lmhead'),
            ],
        },
        'roberta': {
            # note: posi start from pad_idx+1!!
            'confS': [('b_vpath', None), ('init_with_bmodel', None), ('enc', 'tf'), ('dec', ''), ('b_inc_dec', False), ('tie_embs2lmhead', True), ('emb.posi_offset0', 2)],
            'confT': 'bert',
            'paramT': [
                # emb
                ('roberta.embeddings.position_ids', ''), ('roberta.embeddings', 'emb'), ('LayerNorm', 'layer_norm'),
                # enc
                ('roberta.encoder', 'enc'), ('enc.layer.', 'enc.T'), ('self.query', 'affine_q.linear'),
                ('self.key', 'affine_k.linear'), ('self.value', 'affine_v.linear'),
                ('attention.output.dense', 'attention.final_linear.linear'), ('attention.output.layer_norm', 'att_ln'),
                ('intermediate.dense', 'ffn_d1'), ('output.dense', 'ffn_d2'), ('output.layer_norm', 'ffn_ln'),
                # lmhead
                ('lm_head', 'lmhead'), ('lmhead.bias', ''),
            ],
        },
        'xlm-roberta': 'roberta',  # same as roberta!
        'matbert': 'bert',
        'scibert': 'bert',
        'bart': {
            # note: bart still has the hack of posi += 2
            'confS': [('b_vpath', None), ('init_with_bmodel', None), ('enc', 'tf'), ('dec', 'tf'), ('lmhead.use_hid', False), ('final_bias', False), ('tie_embs2dec', True), ('tie_embs2lmhead', True), ('emb.posi_offset1', 2), ('emb_trg.posi_offset1', 2)],
            'confT': [('vocab_size', 'vocab_size'), ('max_position_embeddings', 'max_abs_posi'), ('encoder_layers', 'enc.n_layers'), ('encoder_ffn_dim', 'enc.d_ff'), ('encoder_attention_heads', 'enc.head_count'), ('decoder_layers', 'dec.n_layers'), ('decoder_ffn_dim', 'dec.d_ff'), ('decoder_attention_heads', 'dec.head_count'), ('activation_function', ['ff_act', 'hid_act']), ('d_model', 'b_mdim'), ('dropout', 'dropout'), ('attention_dropout', 'att_drop'), ('initializer_range', 'b_init_range')],
            'paramT': [
                # enc.emb
                ('model.encoder.embed_tokens', 'emb.word_embeddings'),
                ('model.encoder.embed_positions', 'emb.position_embeddings'),
                ('model.encoder.layernorm_embedding', 'emb.layer_norm'),
                # enc
                ('model.encoder.layers.', 'enc.T'), ('self_attn.k_proj', 'attention.affine_k.linear'),
                ('self_attn.v_proj', 'attention.affine_v.linear'), ('self_attn.q_proj', 'attention.affine_q.linear'),
                ('self_attn.out_proj', 'attention.final_linear.linear'), ('self_attn_layer_norm', 'att_ln'),
                ('fc1', 'ffn_d1'), ('fc2', 'ffn_d2'), ('final_layer_norm', 'ffn_ln'),
                # dec.emb
                ('model.decoder.embed_tokens', 'emb_trg.word_embeddings'),
                ('model.decoder.embed_positions', 'emb_trg.position_embeddings'),
                ('model.decoder.layernorm_embedding', 'emb_trg.layer_norm'),
                # dec
                ('model.decoder.layers.', 'dec.T'), ('encoder_attn.k_proj', 'crossattention.affine_k.linear'),
                ('encoder_attn.v_proj', 'crossattention.affine_v.linear'),
                ('encoder_attn.q_proj', 'crossattention.affine_q.linear'),
                ('encoder_attn.out_proj', 'crossattention.final_linear.linear'), ('encoder_attn_layer_norm', 'catt_ln'),
                # lmhead & extra
                ('lm_head.weight', 'lmhead.decoder.weight'), ('final_logits_bias', ''), ('model.shared.weight', ''),
            ],
        },
        # standard ones
        "baseT": {
            "confS": [('enc', 'tf'), ('dec', 'tf'),
                      ('n_layers', 6), ('mdim', 512), ('d_ff', 2048), ('head_count', 8), ('dropout', 0.1)],
        },
        "largeT": {
            "confS": [('enc', 'tf'), ('dec', 'tf'),
                      ('n_layers', 6), ('mdim', 1024), ('d_ff', 4096), ('head_count', 16), ('dropout', 0.3)],
        },
        "smallT": {
            "confS": [('enc', 'tf'), ('dec', 'tf'),
                      ('n_layers', 3), ('mdim', 512), ('d_ff', 1024), ('head_count', 8), ('dropout', 0.1)],
        },
        "tinyT": {
            "confS": [('enc', 'tf'), ('dec', 'tf'),
                      ('n_layers', 3), ('mdim', 256), ('d_ff', 1024), ('head_count', 4), ('dropout', 0.1)],
        },
    }

    @staticmethod
    def get_info(tname: str, key=None, df=None):
        table = ZBmod2Helper.INFO_TABLE
        info = tname
        while isinstance(info, str):  # note: allow links!
            info = table[info]
        if key is None:
            return info
        if key not in info:
            return df
        ret = info[key]
        if isinstance(ret, str):
            return ZBmod2Helper.get_info(ret, key, df)
        else:
            return ret

    @staticmethod
    def ff_sconf(x: str, conf: ZBmod2Conf):
        # get info_table entry
        if x.startswith("__"):  # note: allow loading from local dir
            x = zglob1(x)
        tname = ZBmod1Helper.get_tname(x)
        # specify
        update_kwargs = OrderedDict()
        confS = ZBmod2Helper.get_info(tname, 'confS', [])
        if len(confS) > 0:
            for k, v in confS:
                if k in ['b_vpath', 'init_with_bmodel']:
                    assert v is None
                    v = x  # update with bnames!
                update_kwargs[k] = v
        confT = ZBmod2Helper.get_info(tname, 'confT', [])
        if len(confT) > 0:
            from transformers import AutoConfig
            b_config = AutoConfig.from_pretrained(x)
            for k, vs in confT:
                if isinstance(vs, str):
                    vs = [vs]
                for v in vs:
                    update_kwargs[v] = getattr(b_config, k)
        # update
        zlog(f"Update with ff_sconf: {update_kwargs}")
        conf.update_from_dict(update_kwargs, _update_all=True)
        # --
        return x

    @staticmethod
    def init_with_bmodel(x: str, mod: ZBmod2Mod, conf: ZBmod2Conf):
        if x.startswith("__"):  # note: allow loading from local dir
            x = zglob1(x)
        tname = ZBmod1Helper.get_tname(x)
        tinfo = ZBmod1Helper.get_tinfo(tname)
        m = ZBmod1Helper.get_model(tinfo.auto_type, x, conf.cache_dir_or_none, extra_token_num=mod.toker.extra_num)
        paramT = ZBmod2Helper.get_info(tname, 'paramT', [])
        md = BK.change_state_dict(m.state_dict(), sub_mods=paramT)
        BK.load_mod(mod, md)
        zlog(f"Load for {mod} from pretrained {x}")
        # --

# --
# b mspx/nn/layers2/bmod/bmod2:
