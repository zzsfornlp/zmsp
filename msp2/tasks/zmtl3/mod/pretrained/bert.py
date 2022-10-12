#

# wrapper for pretrained bert-like (encoder-only) modules

__all__ = [
    "ZBertConf", "ZBertMod",
]

from typing import List
from msp2.nn import BK
from msp2.nn.l3 import *
from msp2.utils import zlog, zwarn, Conf
from .common import *

# --

class ZBertConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        self.b_model = "bert-base-multilingual-cased"  # or "bert-base-multilingual-cased", "bert-base-cased", "bert-large-cased", "bert-base-chinese", "roberta-base", "roberta-large", "xlm-roberta-base", "xlm-roberta-large", ...
        self.b_cache_dir = ""  # dir for downloading
        self.b_extra_tokens = []  # add extra tokens?
        self.b_no_pretrain = False  # no init from pretrained ones
        self.b_inc_lmhead = False  # whether include lmhead?
        self.b_ft = True  # whether fine-tune the model
        self.b_kwargs = {}  # extra changes to bert's config (when no_pretrain)
        # --

    @property
    def cache_dir_or_none(self):
        return self.b_cache_dir if self.b_cache_dir else None

@node_reg(ZBertConf)
class ZBertMod(Zlayer):
    def __init__(self, conf: ZBertConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf = self.conf
        self.b_type = 'bert'
        # --
        # create
        self.sub_toker = PretrainedSubwordTokenizer(
            bert_name=conf.b_model, cache_dir=conf.cache_dir_or_none, extra_tokens=conf.b_extra_tokens)
        mod_bert, mod_lmhead = ZBertMod.create_mods(conf, self.sub_toker.extra_num)
        if conf.b_ft:
            self.bert, self.lmhead = mod_bert, mod_lmhead
        else:  # no adding modules in!
            zlog("No fine-tune for the bert/lmhead modules!")
            self.setattr_borrow('bert', mod_bert)
            self.setattr_borrow('lmhead', mod_lmhead)
        # --

    @property
    def tokenizer(self):
        return self.sub_toker.tokenizer

    @staticmethod
    def create_mods(conf: ZBertConf, extra_num=0):
        bert_name, cache_dir = conf.b_model, conf.cache_dir_or_none
        zlog(f"Creating bert model of {bert_name} from {cache_dir}")
        # --
        # get the MLM model
        from transformers import AutoConfig, AutoModelForMaskedLM
        if conf.b_no_pretrain:
            bert_config = AutoConfig.from_pretrained(bert_name)
            for k, v in conf.b_kwargs.items():
                assert hasattr(bert_config, k)
                setattr(bert_config, k, v)
            zwarn(f"No pretrain-loading with {conf.b_kwargs}, really want this?")
            m = AutoModelForMaskedLM.from_config(bert_config)
        else:
            m = AutoModelForMaskedLM.from_pretrained(bert_name, cache_dir=cache_dir)
        if extra_num > 0:
            m.resize_token_embeddings(len(m.get_input_embeddings().weight) + extra_num)
        # --
        # get the two modules
        mod_bert, mod_lmhead = None, None
        for _bname in ['bert', 'roberta']:
            if hasattr(m, _bname):
                mod_bert = getattr(m, _bname)
                if hasattr(mod_bert, "pooler"):  # note: delete unused part!
                    zlog("Delete 'pooler' for mod_bert")
                    mod_bert.__delattr__("pooler")
                    mod_bert.__setattr__('pooler', (lambda x: x))  # note: no need for pooler!
                break
        assert mod_bert is not None
        mod_bert.eval()
        if conf.b_inc_lmhead:
            for _hname in ['cls', 'lm_head']:
                if hasattr(m, _hname):
                    mod_lmhead = getattr(m, _hname)
            assert mod_lmhead is not None
            mod_lmhead.eval()
        # --
        zlog(f"Creating ok.")
        return mod_bert, mod_lmhead

    # forwards
    def forward_bert(self, *args, **kwargs):
        return self.bert(*args, **kwargs)

    def forward_lmhead(self, hid_t):
        return self.lmhead(hid_t)

    # shortcuts
    # [*, L], [*, L], L([*, L], [*, L, D])
    def forward_enc(self, ids_t, ids_mask_t, mixes: List = None):
        if mixes:
            embs = self.bert.get_input_embeddings()
            input_t = embs(ids_t)  # [*, L, D]
            for mix_w_t, mix_emb_t in mixes:  # note: order matters!
                _w = mix_w_t.unsqueeze(-1)  # [*, L, 1]
                input_t = _w * mix_emb_t + (1.-_w) * input_t
            return self.bert(inputs_embeds=input_t, attention_mask=ids_mask_t,
                             output_attentions=True, output_hidden_states=True, return_dict=True)
        else:
            return self.bert(input_ids=ids_t, attention_mask=ids_mask_t,
                             output_attentions=True, output_hidden_states=True, return_dict=True)
        # --

    # info
    def get_mdim(self) -> int: return self.bert.config.hidden_size
    def get_head_num(self) -> int: return self.bert.config.num_attention_heads
