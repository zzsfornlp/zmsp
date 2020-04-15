#

# pretrained bert as part of the model (trainable)

try:
    from transformers import BertModel, BertTokenizer, BertForMaskedLM, RobertaModel
except:
    transformers = None

from typing import List
import numpy as np

from msp.utils import Conf, zcheck, zlog, zwarn, Helper, Random
from msp.nn import BK
from msp.nn.layers import BasicNode, Embedding
from msp.nn.modules.berter import Berter

#
class Berter2Conf(Conf):
    def __init__(self):
        # basic
        self.bert2_model = "bert-base-multilingual-cased"  # or "bert-base-cased", "bert-large-cased", "bert-base-chinese", ...
        self.bert2_lower_case = False
        self.bert2_output_layers = [-1]  # which layers to extract, use concat for final output
        self.bert2_trainable_layers = 0  # how many (upper) layers are trainable: all the way down to the last
        # cache dir for downloading bert models
        self.bert2_cache_dir = ""
        # extra special one, zero padding embedding
        self.bert2_zero_pademb = False  # whether make padding embedding zero vector
        # whether include CLS for return expr
        self.bert2_retinc_cls = False
        # dropout (mask) rate for training
        self.bert2_training_mask_rate = 0.1
        # output mode
        self.bert2_output_mode = "layered"  # layered, concat, weighted (layered by default with extra dim)
        # =====
        # other inputs (these two should match)
        self.bert2_other_input_names = []
        self.bert2_other_input_vsizes = []

# weighted (mixing) bert features
class BertFeaturesWeightLayer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, bert_fold: int):
        super().__init__(pc, None, None)
        self.bert_fold = bert_fold
        self.bert_gamma = self.add_param("AG", (), 1.)  # scalar
        self.bert_lambdas = self.add_param("AL", (), [1./bert_fold] * bert_fold)  # [fold]

    def __call__(self, bert_t):
        lambdas_softmax = BK.softmax(self.bert_lambdas, -1).unsqueeze(-1)  # [fold, 1]
        weighted_bert_t = (bert_t * lambdas_softmax).sum(-2) * self.bert_gamma  # [*, D]
        return weighted_bert_t

#
class Berter2Seq:
    def __init__(self, all_toks, all_is_starts, all_ids, all_typeids):
        self.orig_toks = all_toks
        self.orig_is_starts = all_is_starts
        self.orig_ids = all_ids
        self.orig_typeids = all_typeids
        self.cur_toks = self.orig_toks
        self.cur_is_starts = self.orig_is_starts
        self.cur_ids = self.orig_ids
        self.cur_typeids = self.orig_typeids
        self._share_obj = True

    def reset(self):
        self.cur_toks = self.orig_toks
        self.cur_is_starts = self.orig_is_starts
        self.cur_ids = self.orig_ids
        self.cur_typeids = self.orig_typeids
        self._share_obj = True

    # =====
    # each time we are applying on the orig

    # always apply mask (0 keeps origin, 1 to mask&repl) on the original ones
    # todo(note): only change ids and correponding the shapes of starts and typeids
    def apply_mask_inp(self, mask_repl_arr, mask_repl_id):
        # todo(note): first shallow copy self
        # self.cur_toks = self.orig_toks.copy()  # no need for toks
        self.cur_is_starts = self.orig_is_starts.copy()
        self.cur_ids = self.orig_ids.copy()
        if self.cur_typeids:
            self.cur_typeids = self.orig_typeids.copy()
        # -----
        length = len(self.cur_ids)
        for i, m in enumerate(mask_repl_arr):
            if i >= length:
                break
            if m:
                self.cur_ids[i] = [mask_repl_id]
                self.cur_is_starts[i] = [1]
                if self.cur_typeids:
                    self.cur_typeids[i] = [self.cur_typeids[i][0]]

    def apply_mask_new(self, mask_repl_arr, mask_repl_id):
        ret = Berter2Seq(self.orig_toks, self.orig_is_starts, self.orig_ids, self.orig_typeids)
        ret.apply_mask_inp(mask_repl_arr, mask_repl_id)
        return ret

    # one time process
    def mask_and_return(self, mask_arr, mask_id):
        self.apply_mask_inp(mask_arr, mask_id)
        ret_ids, ret_is_starts, ret_typeids = self.subword_ids, self.subword_is_start, self.subword_typeids
        self.reset()
        return ret_ids, ret_is_starts, ret_typeids

    @property
    def subword_ids(self):
        return Helper.join_list(self.cur_ids)

    @property
    def subword_is_start(self):
        return Helper.join_list(self.cur_is_starts)

    @property
    def subword_typeids(self):
        if self.cur_typeids is None:
            return None
        else:
            return Helper.join_list(self.cur_typeids)

    @staticmethod
    def create(bert_tokenizer, tokens: List[str], typeids: List[int]=None):
        all_toks, all_is_starts, all_ids, all_typeids = [], [], [], None
        if typeids is not None:
            assert len(typeids) == len(tokens)
            all_typeids = []
        # -----
        for i, t in enumerate(tokens):
            cur_toks = bert_tokenizer.tokenize(t)
            # in some cases, there can be empty strings -> put the original word
            if len(cur_toks) == 0:
                cur_toks = [t]
            cur_is_start = [0] * len(cur_toks)
            cur_is_start[0] = 1
            all_toks.append(cur_toks)
            all_is_starts.append(cur_is_start)
            all_ids.append(bert_tokenizer.convert_tokens_to_ids(cur_toks))
            if all_typeids is not None:
                all_typeids.append([typeids[i]] * len(cur_toks))
        # -----
        return Berter2Seq(all_toks, all_is_starts, all_ids, all_typeids)

#
class Berter2(BasicNode):
    def __init__(self, pc: BK.ParamCollection, bconf: Berter2Conf):
        super().__init__(pc, None, None)
        self.bconf = bconf
        self.model_name = bconf.bert2_model
        zlog(f"Loading pre-trained bert model for Berter2 of {self.model_name}")
        # Load pretrained model/tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=bconf.bert2_lower_case,
                                                       cache_dir=None if (not bconf.bert2_cache_dir) else bconf.bert2_cache_dir)
        self.model = BertModel.from_pretrained(self.model_name, output_hidden_states=True,
                                               cache_dir=None if (not bconf.bert2_cache_dir) else bconf.bert2_cache_dir)
        zlog(f"Load done, move to default device {BK.DEFAULT_DEVICE}")
        BK.to_device(self.model)
        # =====
        # zero padding embeddings?
        if bconf.bert2_zero_pademb:
            with BK.no_grad_env():
                # todo(warn): specific!!
                zlog(f"Unusual operation: make bert's padding embedding (idx0) zero!!")
                self.model.embeddings.word_embeddings.weight[0].fill_(0.)
        # =====
        # check trainable ones and add parameters
        # todo(+N): this part is specific and looking into the lib, can break in further versions!!
        # the idx of layer is [1(embed)] + [N(enc)], that is, layer0 is the output of embeddings
        self.hidden_size = self.model.config.hidden_size
        self.num_bert_layers = len(self.model.encoder.layer) + 1  # +1 for embeddings
        self.output_layers = [i if i>=0 else (self.num_bert_layers+i) for i in bconf.bert2_output_layers]
        self.layer_is_output = [False] * self.num_bert_layers
        for i in self.output_layers:
            self.layer_is_output[i] = True
        # the highest used layer
        self.output_max_layer = max(self.output_layers) if len(self.output_layers)>0 else -1
        # from max-layer down
        self.trainable_layers = list(range(self.output_max_layer, -1, -1))[:bconf.bert2_trainable_layers]
        # the lowest trainable layer
        self.trainable_min_layer = min(self.trainable_layers) if len(self.trainable_layers)>0 else (self.output_max_layer+1)
        zlog(f"Build Berter2: {self}")
        # add parameters
        prefix_name = self.pc.nnc_name(self.name, True) + "/"
        for layer_idx in self.trainable_layers:
            if layer_idx == 0:  # add the embedding layer
                infix_name = "embed"
                named_params = self.pc.param_add_external(prefix_name+infix_name, self.model.embeddings)
            else:
                # here we should use the original (-1) index
                infix_name = "enc"+str(layer_idx)
                named_params = self.pc.param_add_external(prefix_name+infix_name, self.model.encoder.layer[layer_idx-1])
            # add to self.params
            for one_name, one_param in named_params:
                assert f"{infix_name}_{one_name}" not in self.params
                self.params[f"{infix_name}_{one_name}"] = one_param
        # for dropout/mask input
        self.random_sample_stream = Random.stream(Random.random_sample)
        # =====
        # for other inputs; todo(note): still, 0 means all-zero embedding
        self.other_embeds = [self.add_sub_node("OE", Embedding(self.pc, vsize, self.hidden_size, fix_row0=True))
                             for vsize in bconf.bert2_other_input_vsizes]
        # =====
        # for output
        if bconf.bert2_output_mode == "layered":
            self.output_f = lambda x: x
            self.output_dims = (self.hidden_size, len(self.output_layers), )
        elif bconf.bert2_output_mode == "concat":
            self.output_f = lambda x: x.view(BK.get_shape(x)[:-2]+[-1])  # combine the last two dims
            self.output_dims = (self.hidden_size*len(self.output_layers), )
        elif bconf.bert2_output_mode == "weighted":
            self.output_f = self.add_sub_node("wb", BertFeaturesWeightLayer(pc, len(self.output_layers)))
            self.output_dims = (self.hidden_size, )
        else:
            raise NotImplementedError(f"UNK mode for bert2 output: {bconf.bert2_output_mode}")

    def __repr__(self):
        return f"Berter2({self.model_name}): output={self.output_layers}, trainable={self.trainable_layers}"

    # the same as Berter for preparing
    def subword_tokenize(self, tokens: List[str], no_special_root: bool, mask_idx=-1, mask_mode="all", mask_repl="",
                         typeids: List[int]=None):
        assert no_special_root
        return Berter._subword_tokenize(self.tokenizer, tokens, mask_idx, mask_mode, mask_repl, typeids=typeids)

    def subword_tokenize2(self, tokens: List[str], no_special_root: bool, typeids: List[int]=None):
        assert no_special_root
        return Berter2Seq.create(self.tokenizer, tokens, typeids)

    # =====
    # actual forward: take advantage of fixed layers
    # todo(+N): specific: BertModel.forward
    def forward_features(self, ids_expr, mask_expr, typeids_expr, other_embed_exprs: List):
        bmodel = self.model
        bmodel_embedding = bmodel.embeddings
        bmodel_encoder = bmodel.encoder
        # prepare
        attention_mask = mask_expr
        token_type_ids = BK.zeros(BK.get_shape(ids_expr)).long() if typeids_expr is None else typeids_expr
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(bmodel.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # embeddings
        cur_layer = 0
        if self.trainable_min_layer <= 0:
            last_output = bmodel_embedding(ids_expr, position_ids=None, token_type_ids=token_type_ids)
        else:
            with BK.no_grad_env():
                last_output = bmodel_embedding(ids_expr, position_ids=None, token_type_ids=token_type_ids)
        # extra embeddings (this implies overall graident requirements!!)
        for one_eidx, one_embed in enumerate(self.other_embeds):
            last_output += one_embed(other_embed_exprs[one_eidx])  # [bs, slen, D]
        # =====
        all_outputs = []
        if self.layer_is_output[cur_layer]:
            all_outputs.append(last_output)
        cur_layer += 1
        # todo(note): be careful about the indexes!
        # not-trainable encoders
        trainable_min_layer_idx = max(0, self.trainable_min_layer-1)
        with BK.no_grad_env():
            for layer_module in bmodel_encoder.layer[:trainable_min_layer_idx]:
                last_output = layer_module(last_output, extended_attention_mask, None)[0]
                if self.layer_is_output[cur_layer]:
                    all_outputs.append(last_output)
                cur_layer += 1
        # trainable encoders
        for layer_module in bmodel_encoder.layer[trainable_min_layer_idx:self.output_max_layer]:
            last_output = layer_module(last_output, extended_attention_mask, None)[0]
            if self.layer_is_output[cur_layer]:
                all_outputs.append(last_output)
            cur_layer += 1
        assert cur_layer == self.output_max_layer + 1
        # stack
        if len(all_outputs) == 1:
            ret_expr = all_outputs[0].unsqueeze(-2)
        else:
            ret_expr = BK.stack(all_outputs, -2)  # [BS, SLEN, LAYER, D]
        final_ret_exp = self.output_f(ret_expr)
        return final_ret_exp

    # calculation: split for too long sentences (input List of Iterable)
    def forward_batch(self, batched_ids: List, batched_starts: List, batched_typeids: List,
                      training: bool, other_inputs: List[List]=None):
        conf = self.bconf
        tokenizer = self.tokenizer
        PAD_IDX = tokenizer.pad_token_id
        MASK_IDX = tokenizer.mask_token_id
        CLS_IDX = tokenizer.cls_token_id
        SEP_IDX = tokenizer.sep_token_id
        if other_inputs is None:
            other_inputs = []
        # =====
        # batch: here add CLS and SEP
        bsize = len(batched_ids)
        max_len = max(len(z) for z in batched_ids) + 2  # plus [CLS] and [SEP]
        input_shape = (bsize, max_len)
        # first collect on CPU
        input_ids_arr = np.full(input_shape, PAD_IDX, dtype=np.int64)
        input_ids_arr[:, 0] = CLS_IDX
        input_mask_arr = np.full(input_shape, 0, dtype=np.float32)
        input_is_start_arr = np.full(input_shape, 0, dtype=np.int64)
        input_typeids = None if batched_typeids is None else np.full(input_shape, 0, dtype=np.int64)
        other_input_arrs = [np.full(input_shape, 0, dtype=np.int64) for _ in other_inputs]
        if conf.bert2_retinc_cls:  # act as the ROOT word
            input_is_start_arr[:, 0] = 1
        training_mask_rate = conf.bert2_training_mask_rate if training else 0.
        self_sample_stream = self.random_sample_stream
        for bidx in range(bsize):
            cur_ids, cur_starts = batched_ids[bidx], batched_starts[bidx]
            cur_end = len(cur_ids) + 2  # plus CLS and SEP
            if training_mask_rate>0.:
                # input dropout
                input_ids_arr[bidx, 1:cur_end] = [(MASK_IDX if next(self_sample_stream)<training_mask_rate else z)
                                                  for z in cur_ids] + [SEP_IDX]
            else:
                input_ids_arr[bidx, 1:cur_end] = cur_ids + [SEP_IDX]
            input_is_start_arr[bidx, 1:cur_end-1] = cur_starts
            input_mask_arr[bidx, :cur_end] = 1.
            if batched_typeids is not None and batched_typeids[bidx] is not None:
                input_typeids[bidx, 1:cur_end-1] = batched_typeids[bidx]
            for one_other_input_arr, one_other_input_list in zip(other_input_arrs, other_inputs):
                one_other_input_arr[bidx, 1:cur_end-1] = one_other_input_list[bidx]
        # arr to tensor
        input_ids_t = BK.input_idx(input_ids_arr)
        input_mask_t = BK.input_real(input_mask_arr)
        input_is_start_t = BK.input_idx(input_is_start_arr)
        input_typeid_t = None if input_typeids is None else BK.input_idx(input_typeids)
        other_input_ts = [BK.input_idx(z) for z in other_input_arrs]
        # =====
        # forward (maybe need multiple times to fit maxlen constraint)
        MAX_LEN = 510  # save two for [CLS] and [SEP]
        BACK_LEN = 100  # for splitting cases, still remaining some of previous sub-tokens for context
        if max_len <= MAX_LEN:
            # directly once
            final_outputs = self.forward_features(input_ids_t, input_mask_t, input_typeid_t, other_input_ts)  # [bs, slen, *...]
            start_idxes, start_masks = BK.mask2idx(input_is_start_t.float())  # [bsize, ?]
        else:
            all_outputs = []
            cur_sub_idx = 0
            slice_size = [bsize, 1]
            slice_cls, slice_sep = BK.constants(slice_size, CLS_IDX, dtype=BK.int64), BK.constants(slice_size, SEP_IDX, dtype=BK.int64)
            while cur_sub_idx < max_len-1:  # minus 1 to ignore ending SEP
                cur_slice_start = max(1, cur_sub_idx - BACK_LEN)
                cur_slice_end = min(cur_slice_start + MAX_LEN, max_len - 1)
                cur_input_ids_t = BK.concat([slice_cls, input_ids_t[:, cur_slice_start:cur_slice_end], slice_sep], 1)
                # here we simply extend extra original masks
                cur_input_mask_t = input_mask_t[:, cur_slice_start-1:cur_slice_end+1]
                cur_input_typeid_t = None if input_typeid_t is None else input_typeid_t[:, cur_slice_start-1:cur_slice_end+1]
                cur_other_input_ts = [z[:, cur_slice_start-1:cur_slice_end+1] for z in other_input_ts]
                cur_outputs = self.forward_features(cur_input_ids_t, cur_input_mask_t, cur_input_typeid_t, cur_other_input_ts)
                # only include CLS in the first run, no SEP included
                if cur_sub_idx == 0:
                    # include CLS, exclude SEP
                    all_outputs.append(cur_outputs[:, :-1])
                else:
                    # include only new ones, discard BACK ones, exclude CLS, SEP
                    all_outputs.append(cur_outputs[:, cur_sub_idx - cur_slice_start + 1:-1])
                    zwarn(f"Add multiple-seg range: [{cur_slice_start}, {cur_sub_idx}, {cur_slice_end})] "
                          f"for all-len={max_len}")
                cur_sub_idx = cur_slice_end
            final_outputs = BK.concat(all_outputs, 1)  # [bs, max_len-1, *...]
            start_idxes, start_masks = BK.mask2idx(input_is_start_t[:,:-1].float())  # [bsize, ?]
        start_expr = BK.gather_first_dims(final_outputs, start_idxes, 1)  # [bsize, ?, *...]
        return start_expr, start_masks  # [bsize, ?, ...], [bsize, ?]

    # wrapper calling
    def __call__(self, inputs: List[Berter2Seq]):
        training = self.rop.training
        batched_ids, batched_starts, batched_typeids = \
            [z.subword_ids for z in inputs], [z.subword_is_start for z in inputs], [z.cur_typeids for z in inputs]
        ret, _ = self.forward_batch(batched_ids, batched_starts, batched_typeids, training)
        return ret

    # actually [BS, SEQLEN, LAYER, HS]
    def get_output_dims(self, *input_dims):
        return self.output_dims

    def refresh(self, rop=None):
        # todo(note): not following convention here, simply on the model
        super().refresh(rop)
        if self.rop.training:
            self.model.train()
        else:
            self.model.eval()

# b msp/nn/modules/berter2.py:163
