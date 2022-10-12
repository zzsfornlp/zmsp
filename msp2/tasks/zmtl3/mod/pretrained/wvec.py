#

# pre-trained word vectors

__all__ = [
    "ZWvecConf", "ZWvecMod",
]

import os.path
from typing import List
import numpy as np
from msp2.nn import BK
from msp2.nn.l3 import *
from msp2.utils import zlog, zwarn, Conf, zglob1, ZObject
from msp2.data.inst import SplitAlignInfo
from msp2.data.vocab import WordVectors
from .common import *

# --

class ZWvecConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        # note: reuse some of ZBert's names
        self.b_model = ""  # pretrained file name
        self.b_cache_dir = ""  # dir for downloading
        self.b_no_pretrain = False  # no init from pretrained ones
        self.b_ft = True  # whether fine-tune the model
        # --

@node_reg(ZWvecConf)
class ZWvecMod(Zlayer):
    def __init__(self, conf: ZWvecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZWvecConf = self.conf
        self.b_type = 'wvec'
        # --
        # load word vecs
        dir_path = zglob1(conf.b_cache_dir)
        emb_path = os.path.join(dir_path, conf.b_model)
        _words, _vecs = WordVectors.load(emb_path, return_raw=True)
        # append a NIL one!
        self.wvec_shape = (1+len(_words), len(_vecs[0]))
        _words = ['<NIL>'] + _words
        _vecs = [np.zeros_like(_vecs[0])] + _vecs  # note: all 0!
        self.wv = WordVectors(_words, [None] * len(_vecs))  # no store there!
        # --
        self.emb = BK.nn.Embedding.from_pretrained(BK.input_real(np.asarray(_vecs)), freeze=(not conf.b_ft))  # [w,d]
        self.tokenizer = self.sub_toker = WvecTokenizer(self.wv)
        zlog(f"Creating wvec ok.")

    # forwards
    def forward_bert(self, input_ids, attention_mask, return_dict=True, **kwargs):
        ret = self.emb(input_ids)  # [***, D]
        ret = ZObject(last_hidden_state=ret, pooler_output=None, hidden_states=[ret], attentions=[])
        if return_dict:
            return ret
        else:
            return ret, None, [ret], []
        # --

    def forward_lmhead(self, hid_t):
        raise NotImplementedError("Not for this module!")

    # info
    def get_enc_dim(self) -> int: return self.wvec_shape[-1]
    def get_head_num(self) -> int: return 0
    # special one to get input embed!
    def get_embed_w(self): return self.emb.weight  # [nword, dim]

# note: a specific one!!
class WvecTokenizer:
    def __init__(self, wv: WordVectors):
        self.wv = wv
        self.vocab = wv.vocab
        # --
        # put special ids!
        special_names = [['[CLS]', '<s>', '</s>', '<bos>'], ['[SEP]', '</s>', '<s>', '<eos>'],
                         ['[MASK]', '<mask>'], ['[PAD]', '<pad>'], ['[UNK]', '<unk>']]
        special_ids = []
        for ns in special_names:
            _name = None
            for n in ns:
                n2 = wv.find_key(n)
                if n2 is not None:
                    _name = n2
                    break
            _id = self.vocab.get(_name, 0)  # note: default 0!
            special_ids.append(_id)
        self.cls_token_id, self.sep_token_id, self.mask_token_id, self.pad_token_id, self.unk_token_id = special_ids
        self.key = f"wvec{len(self.vocab)}"
        zlog(f"Create WvecTokenizer with special_ids = {special_ids}")
        # --

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def sub_vals(self, vals: List[str]):
        sub_idxes = []
        _unk_id = self.unk_token_id
        for v in vals:
            v2 = self.wv.find_key(v)
            i2 = self.vocab.get(v2, _unk_id)
            sub_idxes.append(i2)
        return vals, sub_idxes, SplitAlignInfo([1]*len(vals))
