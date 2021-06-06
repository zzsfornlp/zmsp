#

# encoders with pretrained bert!
# use this one: (pip install transformers==3.1.0)

__all__ = [
    "ZEncoderBertConf", "ZEncoderBert",
]

from typing import List
import time
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import zlog, ZObject
from msp2.data.inst import InputSubwordSeqField, DataPadder
from ..common import *
from ..enc import *

# --

class ZEncoderBertConf(ZEncoderConf):
    def __init__(self):
        super().__init__()
        # --
        # basic
        self.bert_model = "bert-base-cased"  # or "bert-base-multilingual-cased", "bert-base-cased", "bert-large-cased", "bert-base-chinese", ...
        self.bert_cache_dir = ""
        self.bert_ft = True  # whether fine-tuning (add to self?)
        self.inc_cls = False
        # --
        # special: detach and skip
        self.bert_detach_layers = []  # input-detach at which layers
        self.bert_detach_scales = []  # graident scaling at detaching, 0 means full detach!
        self.bert_detach_skips = []  # use which one as input, <=0 mean no skipping!
        # special: TeeGP
        self.bert_tee_layers = []  # tee output at which layers?
        self.bert_tee_splits = []  # 2*len(layers), for mask0 and mask1
        self.bert_tee_conf = TeeGPConf()
        # --
        # final output
        # self.bert_output_layers = [-1]  # which layers to extract for final output (0 means embedding-layer)
        # self.bert_combiner = CombinerConf()

    @property
    def cache_dir_or_none(self):
        return self.bert_cache_dir if self.bert_cache_dir else None

@node_reg(ZEncoderBertConf)
class ZEncoderBert(ZEncoder):
    def __init__(self, conf: ZEncoderBertConf, vpack, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZEncoderBertConf = self.conf
        # --
        # make a impl and add module
        zlog(f"Loading pre-trained bert model for ZEBert of {conf.bert_model}")
        self.impl = ZBImpl.create(conf.bert_model, cache_dir=conf.cache_dir_or_none)
        if conf.bert_ft:  # fine-tune it!
            self.add_module("M", ModuleWrapper(self.impl.model, None))  # reg it!
        zlog(f"Load ok, move to default device {BK.DEFAULT_DEVICE}")
        self.impl.model.to(BK.DEFAULT_DEVICE)
        # --
        self.extra_info = self._get_extra_info()
        # --

    def _get_extra_info(self):
        conf: ZEncoderBertConf = self.conf
        # --
        # embedding as layer0
        _MAX_LAYER = 25  # note: this should be enough!
        detach_scales = [1.] * _MAX_LAYER  # by default no detach
        skip_prevs = [0] * _MAX_LAYER  # by default no skip
        for i, _lidx in enumerate(conf.bert_detach_layers):
            _lidx = int(_lidx)
            _scale = float(conf.bert_detach_scales[i]) if (len(conf.bert_detach_scales)>i) else 1.
            _skip = int(conf.bert_detach_skips[i]) if (len(conf.bert_detach_skips)>i) else 0
            detach_scales[_lidx] = _scale
            assert _skip>=0, "Cannot feed future as input!!"
            skip_prevs[_lidx] = _skip
        # --
        # tee layers
        tee_layers = [None] * _MAX_LAYER  # by default nope
        _isize = self.get_enc_dim()
        _bert_tee_splits = [float(z) for z in conf.bert_tee_splits]
        for _ii, lidx in enumerate(conf.bert_tee_layers):
            lidx = int(lidx)
            _tee_layer = TeeGPNode(conf.bert_tee_conf, _isize=_isize,
                                   split0=_bert_tee_splits[2*_ii], split1=_bert_tee_splits[2*_ii+1])
            self.add_module(f"Tee_{lidx}", _tee_layer)  # reg it!
            tee_layers[lidx] = _tee_layer
        # --
        return detach_scales, skip_prevs, tee_layers

    # --
    def _rev_f(self, orig_v, ib, inc_cls: bool):
        # note: cls is not included back!
        tmp_v = orig_v[ib.arange2_t, int(inc_cls)+ib.batched_rev_idxes]  # [bsize, sub_len, D]
        tmp_slice_size = BK.get_shape(tmp_v)
        tmp_slice_size[1] = 1  # [bsize, 1, D]
        tmp_slice_zero = BK.zeros(tmp_slice_size)
        aug_v = BK.concat([tmp_slice_zero, tmp_v, tmp_slice_zero], 1)  # [bsize, 1+sub_len+1, D]
        return aug_v
    # --

    # common procedure of preparing inputs
    def prepare_inputs(self, insts: List):
        inc_cls = self.conf.inc_cls
        # --
        # first get sub toks
        sub_toker = self.impl.sub_toker
        # --
        time0 = time.time()
        seq_subs = [s.sent.seq_word.get_subword_seq(sub_toker) for s in insts]
        time1 = time.time()
        # --
        from msp2.nn.modules.berter import BerterInputBatch
        ib = BerterInputBatch(self.impl, seq_subs)
        input_ids, input_masks = ib.get_basic_inputs(0.)  # [bsize, 1+sub_len+1]
        # --
        batched_first_idxes, batched_first_mask = ib.batched_first_idxes, ib.batched_first_mask
        batched_first_idxes_p1 = (batched_first_mask.long() + batched_first_idxes)  # +1 for CLS offset!
        if inc_cls:  # [bsize, 1+orig_len]
            idxes_sub2orig = BK.concat([BK.constants_idx([ib.bsize, 1], 0), batched_first_idxes_p1], 1)
            idxes_origmask = BK.concat([BK.constants_idx([ib.bsize, 1], 1.), batched_first_mask], 1)
        else:
            idxes_sub2orig, idxes_origmask = batched_first_idxes_p1, batched_first_mask
        # --
        cached_input = ZObject(insts=insts, ib=ib, idxes_origmask=idxes_origmask, idxes_sub2orig=idxes_sub2orig,
                               input_ids=input_ids, input_masks=input_masks, subtok_time=time1-time0)
        return cached_input

    # actual forward
    def forward(self, insts: List, med: ZMediator, cached_input: ZObject = None):
        inc_cls = self.conf.inc_cls
        if cached_input is None:
            cached_input = self.prepare_inputs(insts)
        # --
        med.restart(insts=cached_input.insts, mask_t=cached_input.idxes_origmask, valid_idxes_t=cached_input.idxes_sub2orig,
                    rev_f=(lambda t: self._rev_f(t, cached_input.ib, inc_cls)))  # [bsize, 1?+orig_len]
        ret = self.impl.model.forward(cached_input.input_ids, cached_input.input_masks, med=med, extra_info=self.extra_info)
        # med.restart()  # not clear here!
        return ret[0], {"subtok_time": cached_input.subtok_time}

    # info
    def get_enc_dim(self) -> int: return self.impl.model.config.hidden_size
    def get_head_dim(self) -> int: return self.impl.model.config.num_attention_heads

    # special
    def get_layered_params(self):
        m = self.impl.model
        return [list(z.parameters()) for z in m.encoder.layer]

# =====
class ZBImpl:
    @staticmethod
    def get_type(model_name):
        last_name = model_name.split("/")[-1]
        model_type = last_name.split("-")[0].lower()
        return model_type

    @staticmethod
    def name2cls(name: str):  # can either be full model_name or model_type
        model_type = ZBImpl.get_type(name)
        return {"bert": ZBImplBert}[model_type]

    @staticmethod
    def create(model_name: str, **kwargs):
        model_cls = ZBImpl.name2cls(model_name)
        return model_cls(model_name, **kwargs)

# =====
# bert
class ZBImplBert(ZBImpl):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        # --
        from transformers import BertTokenizer
        from .modeling_bert import BertModel
        self.tokenizer = BertTokenizer.from_pretrained(model_name, **kwargs)
        self.model = BertModel.from_pretrained(model_name, **kwargs)
        self.model.eval()
        from msp2.nn.modules.berter_impl import BertSubwordTokenizer
        self.sub_toker = BertSubwordTokenizer(self.tokenizer)
        # --
