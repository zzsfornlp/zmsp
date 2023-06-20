#

# a unified wrapper module for pretrained models (emb+enc+dec+lmhead)

__all__ = [
    "ZBmod1Conf", "ZBmod1Mod", "ZBmod1Helper",
]

from typing import List, Type, Dict
from mspx.nn import BK
from mspx.utils import zlog, zwarn, Conf, ZObject, zglob1
from mspx.data.vocab import TokerPretrained
from ...layers import *
from .base import *

# --

@NnConf.rd('bmod')
class ZBmod1Conf(ZBmodBaseConf):
    def __init__(self):
        super().__init__()
        # --
        # enc-only: "bert-base-multilingual-cased", "bert-base-cased", "bert-large-cased", "bert-base-chinese", "roberta-base", "roberta-large", "xlm-roberta-base", "xlm-roberta-large", ...
        # dec-only: "gpt2", ...
        # enc-dec: "facebook/bart-base", "facebook/bart-large", ...
        self.b_model = "bert-base-multilingual-cased"  # model name
        self.b_no_pretrain = False  # no init from pretrained ones
        self.b_ft = True  # whether fine-tune the model
        self.b_kwargs = {}  # extra changes to bert's config (when no_pretrain)
        # --

@ZBmod1Conf.conf_rd()
class ZBmod1Mod(ZBmodBaseMod):
    def __init__(self, conf: ZBmod1Conf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZBmod1Conf = self.conf
        # --
        # note: allow loading from local dir
        if conf.b_model.startswith("__"):
            conf.b_model = zglob1(conf.b_model)
        # --
        # create
        self.toker = TokerPretrained(
            bert_name=conf.b_model, cache_dir=conf.cache_dir_or_none, extra_tokens=conf.b_extra_tokens)
        assert conf.b_inc_emb
        self.mconf, self.minfo, (mod_enc, mod_dec, mod_lmhead) = ZBmod1Helper.create(conf, self.toker.extra_num)
        # --
        if conf.b_ft:
            self.enc, self.dec, self.lmhead = mod_enc, mod_dec, mod_lmhead
        else:  # no adding into the "modules"!
            zlog("No fine-tune for the modules!")
            self.setattr_borrow('enc', mod_enc)
            self.setattr_borrow('dec', mod_dec)
            self.setattr_borrow('lmhead', mod_lmhead)
        # --

    # --
    # forwards

    # [*, L], [*, L], L([*, L], [*, L, D])
    # typical encoders: BertModel, BartEncoder, T5Stack
    def forward_enc(self, t_ids, t_mask=None, t_emb=None, t_ihid=None):
        assert t_ihid is None, "Not implemented for input-hid!"
        kwargs = {
            'input_ids': t_ids, 'attention_mask': t_mask,
            'output_attentions': True, 'output_hidden_states': True, 'return_dict': True,
        }
        if t_emb is not None:
            assert t_ids is None
            kwargs['input_ids'] = None
            kwargs['inputs_embeds'] = t_emb
        ret = self.enc(**kwargs)
        return ret

    # [*, L], [*, L];; [*, S, D], [*, S];; L([*, L], [*, L, D])
    # typical decoders: GPT2Model, BartDecoder, T5Stack
    def forward_dec(self, t_ids, t_mask=None, t_emb=None, t_ihid=None, t_cross=None, t_cross_mask=None, cache=None):
        assert t_ihid is None, "Not implemented for input-hid!"
        kwargs = {
            'input_ids': t_ids, 'attention_mask': t_mask,
            'output_attentions': True, 'output_hidden_states': True, 'return_dict': True,
            'use_cache': True, 'past_key_values': cache,
            'encoder_hidden_states': t_cross, 'encoder_attention_mask': t_cross_mask,
        }
        if t_emb is not None:
            assert t_ids is None
            kwargs['input_ids'] = None
            kwargs['inputs_embeds'] = t_emb
        # note: decoder usually enables causal mask inside itself, therefore mostly we do-not need decoder_mask!
        ret = self.dec(**kwargs)
        ret['cache'] = ret.past_key_values  # name it as cache!
        return ret

    def forward_emb(self, t_ids, mixes=None, forw_full=False):
        assert not forw_full, "Not supported!"
        t_emb = self.enc.get_input_embeddings()(t_ids)
        return ZBmodHelper.mix_embs(t_emb, mixes)

    def forward_emb_trg(self, t_ids, mixes=None, forw_full=False, cache=None):
        assert (not forw_full) and (cache is None), "Not supported!"
        t_emb = self.dec.get_input_embeddings()(t_ids)
        return ZBmodHelper.mix_embs(t_emb, mixes)

    # info
    def get_mdim(self) -> int: return self.mconf.hidden_size
    def get_head_num(self) -> int: return self.mconf.num_attention_heads
    @property
    def toker_trg(self): return self.toker

# --
# helper

# --
# specific helpers
def _get_bmod_enc(m):
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
def _get_bmod_lmhead(m):
    mod_lmhead = None
    for _hname in ['cls', 'lm_head']:
        if hasattr(m, _hname):
            mod_lmhead = getattr(m, _hname)
            break
    assert mod_lmhead is not None
    return mod_lmhead
# --

# helping for loading!
class ZBmod1Helper:
    @staticmethod
    def create(conf: ZBmod1Conf, extra_token_num=0):
        # parse model name
        tname = ZBmod1Helper.get_tname(conf.b_model)
        # get model info
        tinfo = ZBmod1Helper.get_tinfo(tname)
        # get full model & modules
        m = ZBmod1Helper.get_model(tinfo.auto_type, conf.b_model, conf.cache_dir_or_none, extra_token_num,
                                   conf.b_no_pretrain, conf.b_kwargs)
        rets = tinfo.enc_getter(m), tinfo.dec_getter(m), tinfo.lmhead_getter(m)
        rets = [z if inc else None for z,inc in zip(rets, [conf.b_inc_enc, conf.b_inc_dec, conf.b_inc_lmhead])]
        zlog(f"Create modules for {tname}: {[{z.__class__.__name__} if z is not None else 'None' for z in rets]}")
        return m.config, tinfo, rets

    # get type name from bname
    @staticmethod
    def get_tname(bname: str):
        mname = bname.split("/")[-1]  # strip away '/'s
        name_list = ['bert', 'roberta', 'xlm-roberta', 'scibert']
        tname = None
        for t in name_list:
            if mname.startswith(t):
                tname = t
                break
        if tname is None:
            tname = mname.split('-')[0]  # simply use the first one!
        return tname

    # get the model itself
    @staticmethod
    def get_model(auto_type: Type, b_name: str, cache_dir: str = None, extra_token_num=0,
                  b_no_pretrain=False, b_kwargs: Dict = None):
        from transformers import AutoConfig, AutoModel
        if auto_type is None:
            auto_type = AutoModel
        # --
        # note: sometimes this may be different on different machines (at least make init the same)
        with BK.no_change_rand_env():
            if b_no_pretrain:
                b_config = AutoConfig.from_pretrained(b_name)
                if b_kwargs:
                    for k, v in b_kwargs.items():
                        assert hasattr(b_config, k)
                        setattr(b_config, k, v)
                zwarn(f"No pretrain-loading with {b_config}, really want this?")
                m = auto_type.from_config(b_config)
            else:
                m = auto_type.from_pretrained(b_name, cache_dir=cache_dir)
        if extra_token_num > 0:
            m.resize_token_embeddings(len(m.get_input_embeddings().weight) + extra_token_num)
        zlog(f"Creating *model of {b_name} from {cache_dir}")
        m.eval()
        return m
        # --

    # get type info
    @staticmethod
    def get_tinfo(tname: str):
        from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        entry_enc = ZObject(
            auto_type=AutoModelForMaskedLM,
            enc_getter=_get_bmod_enc,
            dec_getter=(lambda x: None),  # no decoder
            lmhead_getter=_get_bmod_lmhead,
        )
        entry_gpt2 = ZObject(
            auto_type=AutoModelForCausalLM,
            enc_getter=(lambda x: None),  # no encoder
            dec_getter=(lambda x: x.transformer),  # note: specific one!
            lmhead_getter=_get_bmod_lmhead,
        )
        entry_s2s = ZObject(
            auto_type=AutoModelForSeq2SeqLM,
            enc_getter=(lambda x: x.get_encoder()),
            dec_getter=(lambda x: x.get_decoder()),
            lmhead_getter=_get_bmod_lmhead,
        )
        INFO = {
            'bert': entry_enc, 'spanbert': entry_enc, 'matbert': entry_enc, 'scibert': entry_enc,
            'roberta': entry_enc, 'xlm-roberta': entry_enc,
            'gpt2': entry_gpt2,
            'bart': entry_s2s, 'mbart': entry_s2s, 't5': entry_s2s,
        }
        # --
        ret = INFO[tname]
        return ret

# --
# b mspx/nn/layers2/bmod/bmod1:
