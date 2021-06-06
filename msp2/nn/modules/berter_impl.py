#

# BerterImpl part for berter
# todo(note): specific to the "transformers" lib's implementation

from typing import List
try:
    from transformers import AutoTokenizer, AutoModel
except:
    AutoTokenizer = AutoModel = None
from msp2.data.inst import SubwordTokenizer
from msp2.nn import BK

class BerterImpl:
    @staticmethod
    def get_type(model_name):
        last_name = model_name.split("/")[-1]
        model_type = last_name.split("-")[0].lower()
        return model_type

    @staticmethod
    def name2cls(name: str):  # can either be full model_name or model_type
        model_type = BerterImpl.get_type(name)
        return {"bert": (BerterBertImpl, BertSubwordTokenizer)}[model_type]

    @staticmethod
    def create(model_name: str, **kwargs):
        model_cls, _ = BerterBertImpl.name2cls(model_name)
        return model_cls(model_name, **kwargs)

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        # double check!
        model_cls, subtok_cls = BerterBertImpl.name2cls(model_name)
        assert model_cls == self.__class__, f"Init with wrong type: {model_cls} vs {self.__class__}"
        # common for most
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.eval()  # set it eval when loading
        self.sub_toker: SubwordTokenizer = subtok_cls(self.tokenizer)

    def __repr__(self):
        return f"BerterImpl({self.model_name})"

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def num_hidden_layers(self):
        return self.model.config.num_hidden_layers

    def forward_embedding(self, input_ids, attention_mask, token_type_ids, position_ids, other_embeds): raise NotImplementedError()
    def forward_hidden(self, i: int, cur_hidden, extended_attention_mask): raise NotImplementedError()

# --

# bert
class BerterBertImpl(BerterImpl):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)

    # forward embeddings and prepare for later hiddens
    def forward_embedding(self, input_ids, attention_mask, token_type_ids, position_ids, other_embeds):
        input_shape = input_ids.size()  # [bsize, len]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = BK.arange_idx(seq_length)  # [len]
            position_ids = position_ids.unsqueeze(0).expand(input_shape)  # [bsize, len]
        if token_type_ids is None:
            token_type_ids = BK.zeros(input_shape).long()  # [bsize, len]
        # BertEmbeddings.forward
        _embeddings = self.model.embeddings
        inputs_embeds = _embeddings.word_embeddings(input_ids)  # [bsize, len, D]
        position_embeddings = _embeddings.position_embeddings(position_ids)  # [bsize, len, D]
        token_type_embeddings = _embeddings.token_type_embeddings(token_type_ids)  # [bsize, len, D]
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        if other_embeds is not None:
            embeddings += other_embeds
        embeddings = _embeddings.LayerNorm(embeddings)
        embeddings = _embeddings.dropout(embeddings)
        # prepare attention_mask
        if attention_mask is None:
            attention_mask = BK.constants(input_shape, value=1.)
        assert attention_mask.dim() == 2
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return embeddings, extended_attention_mask

    def forward_hidden(self, i: int, cur_hidden, extended_attention_mask):
        _layer = self.model.encoder.layer[i]
        # todo(note): we can also extract other info like attentions
        new_hidden = _layer(cur_hidden, extended_attention_mask)[0]
        return new_hidden

# using bert tokenizer
# todo(note): this works well with bert if subword-tok each word individually,
#  but have to be careful with others, things may be tricky with whitespaces!!
class BertSubwordTokenizer(SubwordTokenizer):
    def __init__(self, toker):
        self.toker = toker

    def sub_tok(self, tok: str) -> List[str]:
        cur_toks = self.toker.tokenize(tok)
        # note: delete empty special ones!!
        if len(cur_toks) > 1 and cur_toks[0] == '▁':
            cur_toks = cur_toks[1:]
        # note: in some cases, there can be empty strings -> put the original word
        if len(cur_toks) == 0:
            cur_toks = [tok]
        return cur_toks

    def sub_vals(self, vals: List[str]):
        toker = self.toker
        sub_vals, _, align_info = super().sub_vals(vals)  # simply use the default one!!
        sub_idxes = toker.convert_tokens_to_ids(sub_vals)  # simply change to idxes here!
        return sub_vals, sub_idxes, align_info
