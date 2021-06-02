#

# the module for masked lm

from typing import List, Dict, Tuple
import numpy as np

from msp.utils import Conf, Random, zlog, JsonRW, zfatal, zwarn
from msp.data import VocabPackage
from msp.zext.seq_helper import DataPadder
from msp.nn import BK
from msp.nn.layers import Affine, NoDropRop

from ..base import BaseModuleConf, BaseModule, LossHelper
from .embedder import Inputter

# -----
class MaskLMNodeConf(BaseModuleConf):
    def __init__(self):
        super().__init__()
        self.loss_lambda.val = 0.  # by default no such loss
        # ----
        # hidden layer
        self.hid_dim = 0
        self.hid_act = "elu"
        # mask/pred options
        self.mask_rate = 0.
        self.min_mask_rank = 1  # min word idx to mask
        self.min_pred_rank = 1  # min word idx to pred for the masked ones
        self.max_pred_rank = 1  # max word idx to pred for the masked ones
        self.tie_input_embeddings = False  # tie all preds with input embeddings
        self.init_pred_from_pretrain = False
        # lambdas for word/pos
        self.lambda_word = 1.
        self.lambda_pos = 0.
        # sometimes maybe we don't want to mask some fields, for example, predicting words with full pos
        self.nomask_names = ["pos"]
        # =====
        # special loss considering multiple layers
        self.loss_layers = [-1]  # by default use only the last layer
        self.loss_weights = [1.]  # weights for each layer?
        self.loss_comb_method = "min"  # sum/avg/min/max
        self.score_comb_method = "sum"  # how to combine scores (logprobs)

class MaskLMNode(BaseModule):
    def __init__(self, pc: BK.ParamCollection, input_dim: int, conf: MaskLMNodeConf, inputter: Inputter):
        super().__init__(pc, conf, name="MLM")
        self.conf = conf
        self.inputter = inputter
        self.input_dim = input_dim
        # this step is performed at the embedder, thus still does not influence the inputter
        self.add_root_token = self.inputter.embedder.add_root_token
        # vocab and padder
        vpack = inputter.vpack
        vocab_word, vocab_pos = vpack.get_voc("word"), vpack.get_voc("pos")
        # no mask fields
        self.nomask_names_set = set(conf.nomask_names)
        # models
        if conf.hid_dim <= 0:  # no hidden layer
            self.hid_layer = None
            self.pred_input_dim = input_dim
        else:
            self.hid_layer = self.add_sub_node("hid", Affine(pc, input_dim, conf.hid_dim, act=conf.hid_act))
            self.pred_input_dim = conf.hid_dim
        # todo(note): unk is the first one above real words
        self.pred_word_size = min(conf.max_pred_rank+1, vocab_word.unk)
        self.pred_pos_size = vocab_pos.unk
        if conf.tie_input_embeddings:
            zwarn("Tie all preds in mlm with input embeddings!!")
            self.pred_word_layer = self.pred_pos_layer = None
            self.inputter_word_node = self.inputter.embedder.get_node("word")
            self.inputter_pos_node = self.inputter.embedder.get_node("pos")
        else:
            self.inputter_word_node, self.inputter_pos_node = None, None
            self.pred_word_layer = self.add_sub_node("pw", Affine(pc, self.pred_input_dim, self.pred_word_size, init_rop=NoDropRop()))
            self.pred_pos_layer = self.add_sub_node("pp", Affine(pc, self.pred_input_dim, self.pred_pos_size, init_rop=NoDropRop()))
            if conf.init_pred_from_pretrain:
                npvec = vpack.get_emb("word")
                if npvec is None:
                    zwarn("Pretrained vector not provided, skip init pred embeddings!!")
                else:
                    with BK.no_grad_env():
                        self.pred_word_layer.ws[0].copy_(BK.input_real(npvec[:self.pred_word_size].T))
                    zlog(f"Init pred embeddings from pretrained vectors (size={self.pred_word_size}).")
        # =====
        COMBINE_METHOD_FS = {
            "sum": lambda xs: BK.stack(xs, -1).sum(-1), "avg": lambda xs: BK.stack(xs, -1).mean(-1),
            "min": lambda xs: BK.stack(xs, -1).min(-1)[0], "max": lambda xs: BK.stack(xs, -1).max(-1)[0],
        }
        self.loss_comb_f = COMBINE_METHOD_FS[conf.loss_comb_method]
        self.score_comb_f = COMBINE_METHOD_FS[conf.score_comb_method]

    # ----
    # mask randomly certain part for each inst
    def mask_input(self, input_map: Dict, rand_gen=None, extra_mask_arr=None):
        if rand_gen is None:
            rand_gen = Random
        # -----
        conf = self.conf
        # original valid mask
        input_mask = input_map["mask"]  # [*, len]
        # create mask for input words
        input_word = input_map.get("word")  # [*, len]
        # prepare 'input_erase_mask' and 'output_pred_mask'
        input_erase_mask = (rand_gen.random_sample(input_mask.shape) < conf.mask_rate) & (input_mask>0.)  # by mask-rate
        if input_word is not None:  # min word to mask
            input_erase_mask &= (input_word >= conf.min_mask_rank)
        if extra_mask_arr is not None:  # extra mask for valid points
            input_erase_mask &= (extra_mask_arr>0.)
        # get the new masked inputs (mask for all input components)
        new_map = self.inputter.mask_input(input_map, input_erase_mask.astype(np.int), self.nomask_names_set)
        return new_map, input_erase_mask.astype(np.float32)

    # helper method: p self.arr2words(seq_idx_t); p self.arr2words(argmax_idxes)
    def arr2words(self, arr):
        vocab_word = self.inputter.vpack.get_voc("word")
        flattened_arr = arr.reshape(-1)
        words_list = [vocab_word.idx2word(i) for i in flattened_arr]
        words_arr = np.asarray(words_list).reshape(arr.shape)
        return words_arr

    # [bsize, slen, *]
    # todo(note): be careful that repr_t can be root-appended!!
    def loss(self, repr_ts, input_erase_mask_arr, orig_map: Dict, active_hid=True, **kwargs):
        conf = self.conf
        _tie_input_embeddings = conf.tie_input_embeddings
        # prepare idxes for the masked ones
        if self.add_root_token:  # offset for the special root added in embedder
            mask_idxes, mask_valids = BK.mask2idx(BK.input_real(input_erase_mask_arr), padding_idx=-1)  # [bsize, ?]
            repr_mask_idxes = mask_idxes + 1
            mask_idxes.clamp_(min=0)
        else:
            mask_idxes, mask_valids = BK.mask2idx(BK.input_real(input_erase_mask_arr))  # [bsize, ?]
            repr_mask_idxes = mask_idxes
        # get the losses
        if BK.get_shape(mask_idxes, -1) == 0:  # no loss
            return self._compile_component_loss("mlm", [])
        else:
            if not isinstance(repr_ts, (List, Tuple)):
                repr_ts = [repr_ts]
            target_word_scores, target_pos_scores = [], []
            target_pos_scores = None  # todo(+N): for simplicity, currently ignore this one!!
            for layer_idx in conf.loss_layers:
                # calculate scores
                target_reprs = BK.gather_first_dims(repr_ts[layer_idx], repr_mask_idxes, 1)  # [bsize, ?, *]
                if self.hid_layer and active_hid:  # todo(+N): sometimes, we only want last softmax, need to ensure dim at outside!
                    target_hids = self.hid_layer(target_reprs)
                else:
                    target_hids = target_reprs
                if _tie_input_embeddings:
                    pred_W = self.inputter_word_node.E.E[:self.pred_word_size]  # [PSize, Dim]
                    target_word_scores.append(BK.matmul(target_hids, pred_W.T))  # List[bsize, ?, Vw]
                else:
                    target_word_scores.append(self.pred_word_layer(target_hids))  # List[bsize, ?, Vw]
            # gather the losses
            all_losses = []
            for pred_name, target_scores, loss_lambda, range_min, range_max in \
                    zip(["word", "pos"], [target_word_scores, target_pos_scores], [conf.lambda_word, conf.lambda_pos],
                        [conf.min_pred_rank, 0], [min(conf.max_pred_rank, self.pred_word_size-1), self.pred_pos_size-1]):
                if loss_lambda > 0.:
                    seq_idx_t = BK.input_idx(orig_map[pred_name])  # [bsize, slen]
                    target_idx_t = seq_idx_t.gather(-1, mask_idxes)  # [bsize, ?]
                    ranged_mask_valids = mask_valids * (target_idx_t>=range_min).float() * (target_idx_t<=range_max).float()
                    target_idx_t[(ranged_mask_valids < 1.)] = 0  # make sure invalid ones in range
                    # calculate for each layer
                    all_layer_losses, all_layer_scores = [], []
                    for one_layer_idx, one_target_scores in enumerate(target_scores):
                        # get loss: [bsize, ?]
                        one_pred_losses = BK.loss_nll(one_target_scores, target_idx_t) * conf.loss_weights[one_layer_idx]
                        all_layer_losses.append(one_pred_losses)
                        # get scores
                        one_pred_scores = BK.log_softmax(one_target_scores, -1) * conf.loss_weights[one_layer_idx]
                        all_layer_scores.append(one_pred_scores)
                    # combine all layers
                    pred_losses = self.loss_comb_f(all_layer_losses)
                    pred_loss_sum = (pred_losses * ranged_mask_valids).sum()
                    pred_loss_count = ranged_mask_valids.sum()
                    # argmax
                    _, argmax_idxes = self.score_comb_f(all_layer_scores).max(-1)
                    pred_corrs = (argmax_idxes == target_idx_t).float() * ranged_mask_valids
                    pred_corr_count = pred_corrs.sum()
                    # compile leaf loss
                    r_loss = LossHelper.compile_leaf_info(pred_name, pred_loss_sum, pred_loss_count,
                                                          loss_lambda=loss_lambda, corr=pred_corr_count)
                    all_losses.append(r_loss)
            return self._compile_component_loss("mlm", all_losses)

# b tasks/zmlm/model/mods/masklm:170
