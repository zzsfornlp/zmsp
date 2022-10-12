#

# a unified wrapper module for pretrained models (enc+dec+lmhead)

__all__ = [
    "ZBmodConf", "ZBmodMod",
]

from typing import List, Type
from msp2.nn import BK
from msp2.nn.l3 import *
from msp2.utils import zlog, zwarn, Conf, ZObject, zglob1z
from .common import *

# --

class ZBmodConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        # enc-only: "bert-base-multilingual-cased", "bert-base-cased", "bert-large-cased", "bert-base-chinese", "roberta-base", "roberta-large", "xlm-roberta-base", "xlm-roberta-large", ...
        # dec-only: "gpt2", ...
        # enc-dec: "facebook/bart-base", "facebook/bart-large", ...
        self.b_model = "bert-base-multilingual-cased"  # model name
        self.b_cache_dir = ""  # dir for downloading
        self.b_extra_tokens = []  # add extra tokens?
        self.b_no_pretrain = False  # no init from pretrained ones
        self.b_ft = True  # whether fine-tune the model
        self.b_kwargs = {}  # extra changes to bert's config (when no_pretrain)
        self.b_inc_enc = True  # whether include enc if there is?
        self.b_inc_dec = True  # whether include dec if there is?
        self.b_inc_lmhead = False  # whether include lmhead if there is?
        # --

    @property
    def cache_dir_or_none(self):
        return self.b_cache_dir if self.b_cache_dir else None

@node_reg(ZBmodConf)
class ZBmodMod(Zlayer):
    def __init__(self, conf: ZBmodConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf = self.conf
        # --
        # note: allow loading from local dir
        if conf.b_model.startswith("__"):
            conf.b_model = zglob1z(conf.b_model)
        # --
        # create
        self.sub_toker = PretrainedSubwordTokenizer(
            bert_name=conf.b_model, cache_dir=conf.cache_dir_or_none, extra_tokens=conf.b_extra_tokens)
        self.mconf, self.minfo, (mod_emb, mod_enc, mod_dec, mod_lmhead) = ZBmodHelper.create(conf, self.sub_toker.extra_num)
        # --
        if conf.b_ft:
            self.emb, self.enc, self.dec, self.lmhead = mod_emb, mod_enc, mod_dec, mod_lmhead
        else:  # no adding into the "modules"!
            zlog("No fine-tune for the modules!")
            self.setattr_borrow('emb', mod_emb)
            self.setattr_borrow('enc', mod_enc)
            self.setattr_borrow('dec', mod_dec)
            self.setattr_borrow('lmhead', mod_lmhead)
        # --

    @property
    def tokenizer(self):
        return self.sub_toker.tokenizer

    # --
    # forwards
    def _input_with_mixes(self, m_emb, t_ids, mixes):
        t_input = m_emb(t_ids)  # [*, L, D]
        for t_mix_w, t_mix_emb in mixes:  # note: order matters!
            _w = t_mix_w.unsqueeze(-1)  # [*, L, 1]
            t_input = _w * t_mix_emb + (1.-_w) * t_input
        return t_input

    # [*, L], [*, L], L([*, L], [*, L, D])
    # typical encoders: BertModel, BartEncoder, T5Stack
    def forward_enc(self, t_ids, t_ids_mask, mixes: List = None):
        kwargs = {
            'input_ids': t_ids, 'attention_mask': t_ids_mask,
            'output_attentions': True, 'output_hidden_states': True, 'return_dict': True,
        }
        if mixes:
            t_input = self._input_with_mixes(self.emb, t_ids, mixes)
            kwargs['input_ids'] = None
            kwargs['inputs_embeds'] = t_input
        ret = self.enc(**kwargs)
        return ret

    # [*, L], [*, L];; [*, S, D], [*, S];; L([*, L], [*, L, D])
    # typical decoders: GPT2Model, BartDecoder, T5Stack
    # todo(+N): decoder usually enables causal mask inside itself, therefore mostly we do-not need decoder_mask!
    def forward_dec(self, t_ids, t_ids_mask=None, cache_past=None, t_cross=None, t_cross_mask=None, mixes: List = None):
        kwargs = {
            'input_ids': t_ids, 'attention_mask': t_ids_mask,
            'output_attentions': True, 'output_hidden_states': True, 'return_dict': True,
            'use_cache': True, 'past_key_values': cache_past,
            'encoder_hidden_states': t_cross, 'encoder_attention_mask': t_cross_mask,
        }
        if mixes:
            t_input = self._input_with_mixes(self.emb, t_ids, mixes)
            kwargs['input_ids'] = None
            kwargs['inputs_embeds'] = t_input
        ret = self.dec(**kwargs)
        return ret

    def forward_lmhead(self, t_hid):
        return self.lmhead(t_hid)

    def forward_emb(self, t_ids):
        return self.emb(t_ids)

    def dec_reorder_cache(self, cache, t_idxes):
        return self.dec.reorder_cache(cache, t_idxes)

    # info
    def get_mdim(self) -> int: return self.mconf.hidden_size
    def get_head_num(self) -> int: return self.mconf.num_attention_heads

# --
# helper

# --
# specific helpers
def _get_bert_enc(m):
    mod_enc = None
    for _bname in ['bert', 'roberta']:
        if hasattr(m, _bname):
            mod_enc = getattr(m, _bname)
            if hasattr(mod_enc, "pooler"):  # note: delete unused part!
                zlog("Delete 'pooler' for mod_enc")
                mod_enc.__delattr__("pooler")
                mod_enc.__setattr__('pooler', (lambda x: x))  # note: no need for pooler!
            break
    assert mod_enc is not None
    return mod_enc
def _get_bert_lmhead(m):
    mod_lmhead = None
    for _hname in ['cls', 'lm_head']:
        if hasattr(m, _hname):
            mod_lmhead = getattr(m, _hname)
            break
    assert mod_lmhead is not None
    return mod_lmhead
# --

class ZBmodHelper:
    def __init__(self, conf: ZBmodConf, extra_token_num=0):
        self.conf = conf
        self.extra_token_num = extra_token_num

    @staticmethod
    def create(conf: ZBmodConf, extra_token_num=0):
        helper = ZBmodHelper(conf, extra_token_num)
        conf = helper.conf
        # --
        # parse model name
        tname = helper.get_tname(conf.b_model)
        # get model info
        tinfo = helper.get_tinfo(tname)
        # get full model & modules
        m = helper.get_model(tinfo.auto_type)
        rets = m.get_input_embeddings(), tinfo.enc_getter(m), tinfo.dec_getter(m), tinfo.lmhead_getter(m)
        rets = [z if inc else None for z,inc in zip(rets, [True, conf.b_inc_enc, conf.b_inc_dec, conf.b_inc_lmhead])]
        zlog(f"Create modules for {tname}: {[{z.__class__.__name__} if z is not None else 'None' for z in rets]}")
        return m.config, tinfo, rets

    # get type name
    def get_tname(self, bname: str):
        mname = bname.split("/")[-1]  # strip away '/'s
        name_list = ['bert', 'roberta', 'xlm-roberta', 'bart', 'gpt2', 'spanbert', 'mbart', 'matbert']
        tname = None
        for t in name_list:
            if mname.startswith(t):
                tname = t
                break
        assert tname is not None, f"UNK bname: {bname} {mname}"
        return tname

    # get the model itself
    def get_model(self, auto_type: Type):
        conf, extra_token_num = self.conf, self.extra_token_num
        # --
        from transformers import AutoConfig, AutoModel
        if auto_type is None:
            auto_type = AutoModel
        # --
        b_name, cache_dir = conf.b_model, conf.cache_dir_or_none
        if conf.b_no_pretrain:
            b_config = AutoConfig.from_pretrained(conf.b_model)
            for k, v in conf.b_kwargs.items():
                assert hasattr(b_config, k)
                setattr(b_config, k, v)
            zwarn(f"No pretrain-loading with {b_config}, really want this?")
            m = auto_type.from_config(b_config)
        else:
            m = auto_type.from_pretrained(conf.b_model, cache_dir=cache_dir)
        if extra_token_num > 0:
            m.resize_token_embeddings(len(m.get_input_embeddings().weight) + extra_token_num)
        zlog(f"Creating *model of {b_name} from {cache_dir}")
        m.eval()
        return m
        # --

    # get type info
    def get_tinfo(self, tname: str):
        from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM
        from .wrapper_bart import MyBartModel
        # --
        INFO = {
            "bert": ZObject(
                auto_type=AutoModelForMaskedLM,
                enc_getter=_get_bert_enc,
                dec_getter=(lambda x: None),  # no decoder
                lmhead_getter=_get_bert_lmhead,
            ),
            "roberta": ZObject(
                auto_type=AutoModelForMaskedLM,
                enc_getter=_get_bert_enc,
                dec_getter=(lambda x: None),  # no decoder
                lmhead_getter=_get_bert_lmhead,
            ),
            "xlm-roberta": ZObject(
                auto_type=AutoModelForMaskedLM,
                enc_getter=_get_bert_enc,
                dec_getter=(lambda x: None),  # no decoder
                lmhead_getter=_get_bert_lmhead,
            ),
            "gpt2": ZObject(
                auto_type=AutoModelForMaskedLM,
                enc_getter=(lambda x: None),  # no encoder
                dec_getter=(lambda x: x.transformer),  # note: specific one!
                lmhead_getter=(lambda x: x.get_output_embeddings()),
            ),
            "bart": ZObject(
                auto_type=MyBartModel,  # note: use this modified one!
                enc_getter=(lambda x: x.get_encoder()),
                dec_getter=(lambda x: x.get_decoder()),  # no decoder
                lmhead_getter=(lambda x: x.get_output_embeddings()),
            ),
            "mbart": ZObject(
                auto_type=MyBartModel,  # note: use this modified one!
                enc_getter=(lambda x: x.get_encoder()),
                dec_getter=(lambda x: x.get_decoder()),  # no decoder
                lmhead_getter=(lambda x: x.get_output_embeddings()),
            ),
        }
        INFO['spanbert'] = INFO['bert']
        INFO['matbert'] = INFO['bert']
        # --
        ret = INFO[tname]
        return ret

# --
# b msp2/tasks/zmtl3/mod/pretrained/bmod:71
