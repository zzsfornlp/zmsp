#

# Tokenizer
# -- make it similar to transformers.*Tokenizer

__all__ = [
    "Toker", "TokerChar", "TokerPretrained",
]

import time
from typing import List
from mspx.utils import zlog, zwarn, zglob1
from .vocab import Vocab

class Toker:
    def __init__(self, base, df_idx: int = None):
        self.base = base
        if df_idx is None:
            if isinstance(base, Vocab):
                try:
                    df_idx = base.unk
                except:
                    zwarn(f"Cannot find UNK-idx for {base}, simply use 0!")
                    df_idx = 0
        assert df_idx is not None
        self.df_idx = df_idx

    def __repr__(self):
        return f"Toker({type(self.base)})"

    def __len__(self):
        return len(self.base)

    def __getattr__(self, item):  # delegate to base!
        return getattr(self.base, item)

    def get_sig(self):  # signature
        if self.base is None:
            return None
        for key in ["name_or_path", "name"]:
            ret = getattr(self.base, key, None)
            if ret is not None:
                return f"T_{ret}"
        return None

    # sub-tokenize one token
    def sub_tok(self, tok: str) -> List[str]:
        return [tok]

    # sub-tokenize List[str] with align-info: by default sub_tok each one!
    def sub_vals(self, vals: List[str]):
        from mspx.data.inst import SeqMAlignInfo
        _v, _di = self.base, self.df_idx
        split_sizes = []
        sub_vals: List[str] = []  # flattened ones
        sub_idxes = None if _v is None else []
        for v in vals:
            one_sub_vals = self.sub_tok(v)
            split_sizes.append(len(one_sub_vals))
            sub_vals.extend(one_sub_vals)
            if _v is not None:
                sub_idxes.extend([_v.get(z, _di) for z in one_sub_vals])
        return sub_vals, sub_idxes, SeqMAlignInfo(split_sizes)

# simple splitting into chars
class TokerChar(Toker):
    def get_sig(self): return 'C_' + super().get_sig()
    def sub_tok(self, w: str): return [" "] + list(w)  # todo(+N): simply put a space before!

# pretrained ones!
class TokerPretrained(Toker):
    def __init__(self, bert_name: str, cache_dir=None, extra_tokens=None):
        # --
        t_kwargs = {}
        # note: specific setting here since not in the lib's entries!
        if bert_name.startswith("__"):
            bert_name = zglob1(bert_name)
        if bert_name.split('/')[-1] == "matbert-base-cased":
            t_kwargs['do_lower_case'] = False
        _tokenizer = self.load_toker_from_pretrained(bert_name, cache_dir=cache_dir, **t_kwargs)
        zlog(f"Load tokenizer {bert_name} from {cache_dir}: {_tokenizer}")
        super().__init__(_tokenizer, _tokenizer.unk_token_id)
        self.extra_num = 0
        if extra_tokens:
            self.extra_num = self.tokenizer.add_tokens(extra_tokens)
            zlog(f"Try to add extra_tokens ({self.extra_num}) {extra_tokens}")
        # --
        # used for tokenizer
        from string import punctuation
        self.punct_set = set(punctuation)
        self.is_roberta = ("/bart-" in bert_name) or bert_name.startswith("roberta-")  # special treating for roberta
        # --

    @staticmethod
    def load_toker_from_pretrained(*args, **kwargs):
        from transformers import AutoTokenizer
        for _ in range(10):  # try many times
            try:
                ret = AutoTokenizer.from_pretrained(*args, **kwargs)
                return ret
            except ValueError as err:
                zwarn(f"Loading-pretrain_toker error: {err}")
                time.sleep(10)  # wait for some time!
        # --
        # final trying
        return AutoTokenizer.from_pretrained(*args, **kwargs)

    # sub-tokenize one token
    def sub_tok(self, tok: str):
        raise NotImplementedError("No sub for single tok for this class!")

    # sub-tokenize List[str] with align-info, note: no use of sub_tok, since we may need seq info!!
    def sub_vals(self, vals: List[str]):
        from mspx.data.inst import SeqMAlignInfo
        _tokenizer = self.base
        # --
        split_sizes = []
        sub_vals: List[str] = []  # flattened ones
        for ii, tok in enumerate(vals):
            # simple judge of whether need to add space before
            add_space = (self.is_roberta and not all((c in self.punct_set) for c in tok))
            # tokenize it
            cur_toks = _tokenizer.tokenize((" "+tok) if add_space else tok)
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
        sub_idxes = _tokenizer.convert_tokens_to_ids(sub_vals)  # simply change to idxes here!
        return sub_vals, sub_idxes, SeqMAlignInfo(split_sizes)
    # --
