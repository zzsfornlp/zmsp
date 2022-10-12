#

# common helpers

__all__ = [
    "PretrainedSubwordTokenizer", "BertOuterConf", "BertOuterLayer",
]

from typing import List, Type
from msp2.data.inst import SubwordTokenizer, SplitAlignInfo
from msp2.utils import zlog, zwarn
from msp2.nn import BK
from msp2.nn.l3 import *

# tokenizer helper
class PretrainedSubwordTokenizer(SubwordTokenizer):
    def __init__(self, bert_name: str, cache_dir=None, extra_tokens=None):
        # --
        from transformers import AutoTokenizer
        t_kwargs = {}
        # note: specific setting here since not in the lib's entries!
        if bert_name.split('/')[-1] == "matbert-base-cased":
            t_kwargs['do_lower_case'] = False
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=cache_dir, **t_kwargs)
        # --
        zlog(f"Load tokenizer {bert_name} from {cache_dir}: {self.tokenizer}")
        self.extra_num = 0
        if extra_tokens:
            self.extra_num = self.tokenizer.add_tokens(extra_tokens)
            zlog(f"Try to add extra_tokens ({self.extra_num}) {extra_tokens}")
        # --
        # used for tokenizer
        from string import punctuation
        self.punct_set = set(punctuation)
        self.is_roberta = ("/bart-" in bert_name) or bert_name.startswith("roberta-")  # special treating for roberta
        self.key = bert_name
        # --

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    # sub-tokenize List[str] with align-info, note: no use of sub_tok, since we may need seq info!!
    def sub_vals(self, vals: List[str]):
        toker = self.tokenizer
        # --
        split_sizes = []
        sub_vals: List[str] = []  # flattened ones
        for ii, tok in enumerate(vals):
            # simple judge of whether need to add space before
            add_space = (self.is_roberta and not all((c in self.punct_set) for c in tok))
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

# output helper
class BertOuterConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        # for hidden layer
        self.bert_dim = -1  # dimension of bert
        self.bert_lidx = [-1]  # output
        self.bert_comb = CombinerConf.direct_conf(comb_method='concat')  # combining method
        # for att
        self.att_num = -1  # number of attention heads (att_dim)
        self.att_lidx = [-1]  # which attention layer?
        self.att_comb = CombinerConf.direct_conf(comb_method='concat')  # combining the attention heads
        # --

@node_reg(BertOuterConf)
class BertOuterLayer(Zlayer):
    def __init__(self, conf: BertOuterConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: BertOuterConf = self.conf
        # --
        self.bert_comb = CombinerLayer(conf.bert_comb, isizes=[conf.bert_dim] * len(conf.bert_lidx))
        self.att_comb = CombinerLayer(conf.att_comb, isizes=[conf.att_num] * len(conf.att_lidx))
        # --

    def forward_hid(self, bert_out):
        conf: BertOuterConf = self.conf
        # --
        hids = [bert_out.hidden_states[z] for z in conf.bert_lidx]  # *[bs, len, D]
        ret = self.bert_comb(hids)  # [bs, Q, K, ??]
        return ret

    def forward_att(self, bert_out):
        conf: BertOuterConf = self.conf
        # --
        atts = [bert_out.attentions[z].transpose(-2,-3).transpose(-1,-2) for z in conf.att_lidx]  # *[bs, Q, K, H]
        ret = self.att_comb(atts)  # [bs, Q, K, ??]
        return ret

    def dim_out_hid(self):
        return self.bert_comb.output_dim

    def dim_out_att(self):
        return self.att_comb.output_dim
