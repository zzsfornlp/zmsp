#

# utilizing bert encoder

__all__ = [
    "ZTaskEncBertConf", "ZTaskEncBert", "ZEncoderBertConf", "ZEncoderBert"
]

from typing import List
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.data.inst import SubwordTokenizer, SplitAlignInfo
from msp2.utils import zlog, zwarn, ConfEntryChoices
from .modeling_bert import BertModel
from .modeling_roberta import RobertaModel, XLMRobertaModel
from ...common import ZMediator, ZGcn0Conf, ZGcn0Node
from ..base import *

# --
# extra helper

class ZBertSubwordTokenizer(SubwordTokenizer):
    def __init__(self, bert_name: str, tokenizer):
        self.tokenizer = tokenizer
        # --
        # used for tokenizer
        from string import punctuation
        self.punct_set = set(punctuation)
        self.is_roberta = (bert_name.startswith("roberta-"))  # special treating for roberta
        # --

    # sub-tokenize List[str] with align-info, note: no use of sub_tok, since we may need seq info!!
    def sub_vals(self, vals: List[str]):
        toker = self.tokenizer
        # --
        split_sizes = []
        sub_vals: List[str] = []  # flattened ones
        for ii, tok in enumerate(vals):
            # simple judge of whether need to add space before
            add_space = (ii>0 and self.is_roberta and not all((c in self.punct_set) for c in tok))
            # tokenize it
            cur_toks = toker.tokenize((" "+tok) if add_space else tok)
            # delete special ones!!
            if len(cur_toks) > 0 and cur_toks[0] in ['▁', 'Ġ']:  # for xlmr and roberta
                cur_toks = cur_toks[1:]
            # in some cases, there can be empty strings -> put the original word
            if len(cur_toks) == 0:
                cur_toks = [tok]
            # add
            sub_vals.extend(cur_toks)
            split_sizes.append(len(cur_toks))
        # --
        sub_idxes = toker.convert_tokens_to_ids(sub_vals)  # simply change to idxes here!
        return sub_vals, sub_idxes, SplitAlignInfo(split_sizes)
    # --

# --

class ZTaskEncBertConf(ZTaskEncConf):
    def __init__(self):
        super().__init__()
        # --
        self.bert_conf = ZEncoderBertConf()

    def build_task(self):
        return ZTaskEncBert(self)

class ZTaskEncBert(ZTaskEnc):
    def __init__(self, conf: ZTaskEncBertConf):
        super().__init__(conf)
        # --
        self._bert_mod = None

    # allow lazy load
    @property
    def bert_mod(self):
        if self._bert_mod is None:
            self._bert_mod = ZEncoderBert(self.conf.bert_conf, self)  # directly init here, since no other deps!
        return self._bert_mod

    # no need of vocab
    def build_vocab(self, datasets: List):
        return None

    # prepare one instance
    def prep_inst(self, inst, dataset):
        pass  # simply pass, let "prep_item" handle!!

    # prepare one input_item
    def prep_item(self, item, dataset):
        from ....core.run import InputSeqInfo
        tokenizer = self.bert_mod.tokenizer
        sub_toker = self.bert_mod.sub_toker
        subs = [s.seq_word.get_subword_seq(sub_toker) for s in item.sents]
        IDX_CLS, IDX_SEP = tokenizer.cls_token_id, tokenizer.sep_token_id
        item.seq_info = InputSeqInfo.create_from_subtoks(item, subs, IDX_CLS, IDX_SEP)
        item.assign_batch_len(len(item.seq_info.enc_input_ids))  # note: use this len for batching
        # --

    # already pre-built, simply return it!!
    def build_mod(self, model):
        return self.bert_mod

class ZEncoderBertConf(ZEncoderConf):
    def __init__(self):
        super().__init__()
        # --
        # basic
        self.bert_model = "bert-base-multilingual-cased"  # or "bert-base-multilingual-cased", "bert-base-cased", "bert-large-cased", "bert-base-chinese", "roberta-base", "roberta-large", "xlm-roberta-base", "xlm-roberta-large", ...
        self.bert_cache_dir = ""  # dir for downloading
        self.bert_no_pretrain = False
        self.bert_ft = True  # whether fine-tune the model
        # --
        # extra layers
        self.gcn = ConfEntryChoices({'yes': ZGcn0Conf(), 'no': None}, 'no')

    @property
    def cache_dir_or_none(self):
        return self.bert_cache_dir if self.bert_cache_dir else None

@node_reg(ZEncoderBertConf)
class ZEncoderBert(ZEncoder):
    def __init__(self, conf: ZEncoderBertConf, ztask, **kwargs):
        super().__init__(conf, ztask, **kwargs)
        # --
        conf: ZEncoderBertConf = self.conf
        # make and load bert
        self.tokenizer, self.sub_toker, raw_bert = ZEncoderBert.from_pretrained(conf)
        self.bert = ModuleWrapper(raw_bert, None, no_reg=(not conf.bert_ft))
        # --
        self.gcn = None
        if conf.gcn is not None:
            self.gcn = ZGcn0Node(conf.gcn, _isize=self.get_enc_dim())
        # --

    @staticmethod
    def from_pretrained(conf: ZEncoderBertConf):
        bert_name, cache_dir = conf.bert_model, conf.cache_dir_or_none
        zlog(f"Loading pre-trained bert model for ZBert of {bert_name} from {cache_dir}")
        # --
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=cache_dir)
        sub_toker = ZBertSubwordTokenizer(bert_name, tokenizer)
        mtype = {"bert": BertModel, "roberta": RobertaModel, "xlm": XLMRobertaModel}[bert_name.split("/")[-1].split("-")[0]]
        if conf.bert_no_pretrain:
            from transformers import AutoConfig
            bert_config = AutoConfig.from_pretrained(bert_name)
            model = mtype(bert_config)
            zwarn("No pretrain-loading for bert, really want this?")
        else:
            model = mtype.from_pretrained(bert_name, cache_dir=cache_dir)
        # --
        if hasattr(model, "pooler"):  # note: delete unused part!
            model.__delattr__("pooler")
        # --
        model.eval()  # note: by default set eval!!
        # --
        zlog(f"Load ok, move to default device {BK.DEFAULT_DEVICE}")
        model.to(BK.DEFAULT_DEVICE)
        zlog("Move ok!")
        return tokenizer, sub_toker, model

    @property
    def raw_bert(self):
        return self.bert.node

    # info
    def get_enc_dim(self) -> int: return self.raw_bert.config.hidden_size
    def get_head_dim(self) -> int: return self.raw_bert.config.num_attention_heads
    # special one to get input embed!
    def get_embed_w(self): return self.raw_bert.embeddings.word_embeddings.weight  # [nword, dim]

    # step0!
    def restart(self, ibatch, med):
        # prepare input
        ibatch.set_seq_info(IDX_PAD=self.tokenizer.pad_token_id)
        med.restart(ibatch)

    # forward
    def forward(self, med: ZMediator):
        ibatch_seq_info = med.ibatch.seq_info
        # prepare input, truncate if too long
        _input_ids, _input_masks, _input_segids = \
            ibatch_seq_info.enc_input_ids, ibatch_seq_info.enc_input_masks, ibatch_seq_info.enc_input_segids
        _eff_input_ids = med.get_cache('eff_input_ids')  # note: special name!!
        if _eff_input_ids is not None:
            _input_ids = _eff_input_ids
        # --
        if BK.get_shape(_input_ids,-1) > self.tokenizer.model_max_length:
            _full_len = BK.get_shape(_input_ids, -1)
            _max_len = self.tokenizer.model_max_length
            zwarn(f"Input too long for bert, truncate it: {BK.get_shape(_input_ids)} => {_max_len}")
            _input_ids, _input_masks, _input_segids = \
                _input_ids[:,:_max_len], _input_masks[:,:_max_len], _input_segids[:,:_max_len]
            # todo(+W+N): how to handle decoders for these cases?
        # forward
        ret = self.bert.forward(_input_ids, _input_masks, _input_segids, med=med)
        # extra
        if self.gcn:
            ret = self.gcn.forward(med)
        # --
        return ret
