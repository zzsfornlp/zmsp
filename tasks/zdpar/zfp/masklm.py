#

# the module for masked lm

from typing import List
import numpy as np

from msp.utils import Conf, Random, zlog, JsonRW, zfatal, zwarn
from msp.data import VocabPackage, MultiHelper
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, RefreshOptions, NoDropRop
from msp.zext.seq_helper import DataPadder

from ..common.data import ParseInstance

class MaskLMNodeConf(Conf):
    def __init__(self):
        self._input_dim = -1
        # ----
        # hidden layer
        self.hid_dim = 300
        self.hid_act = "elu"
        # mask/pred options
        self.mask_rate = 0.
        self.min_mask_rank = 2
        self.max_pred_rank = 2000
        self.init_pred_from_pretrain = False

class MaskLMNode(BasicNode):
    def __init__(self, pc: BK.ParamCollection, conf: MaskLMNodeConf, vpack: VocabPackage):
        super().__init__(pc, None, None)
        self.conf = conf
        # vocab and padder
        self.word_vocab = vpack.get_voc("word")
        self.padder = DataPadder(2, pad_vals=self.word_vocab.pad, mask_range=2)  # todo(note): <pad>-id is very large
        # models
        self.hid_layer = self.add_sub_node("hid", Affine(pc, conf._input_dim, conf.hid_dim, act=conf.hid_act))
        self.pred_layer = self.add_sub_node("pred", Affine(pc, conf.hid_dim, conf.max_pred_rank+1, init_rop=NoDropRop()))
        if conf.init_pred_from_pretrain:
            npvec = vpack.get_emb("word")
            if npvec is None:
                zwarn("Pretrained vector not provided, skip init pred embeddings!!")
            else:
                with BK.no_grad_env():
                    self.pred_layer.ws[0].copy_(BK.input_real(npvec[:conf.max_pred_rank+1].T))
                zlog(f"Init pred embeddings from pretrained vectors (size={conf.max_pred_rank+1}).")

    # return (input_word_mask_repl, output_pred_mask_repl, ouput_pred_idx)
    def prepare(self, insts: List[ParseInstance], training):
        conf = self.conf
        word_idxes = [z.words.idxes for z in insts]
        word_arr, input_mask = self.padder.pad(word_idxes)  # [bsize, slen]
        # prepare for the masks
        input_word_mask = (Random.random_sample(word_arr.shape) < conf.mask_rate) & (input_mask>0.)
        input_word_mask &= (word_arr >= conf.min_mask_rank)
        input_word_mask[:, 0] = False  # no masking for special ROOT
        output_pred_mask = (input_word_mask & (word_arr <= conf.max_pred_rank))
        return input_word_mask.astype(np.float32), output_pred_mask.astype(np.float32), word_arr

    # [bsize, slen, *]
    def loss(self, repr_t, pred_mask_repl_arr, pred_idx_arr):
        mask_idxes, mask_valids = BK.mask2idx(BK.input_real(pred_mask_repl_arr))  # [bsize, ?]
        if BK.get_shape(mask_idxes, -1) == 0:  # no loss
            zzz = BK.zeros([])
            return [[zzz, zzz, zzz]]
        else:
            target_reprs = BK.gather_first_dims(repr_t, mask_idxes, 1)  # [bsize, ?, *]
            target_hids = self.hid_layer(target_reprs)
            target_scores = self.pred_layer(target_hids)  # [bsize, ?, V]
            pred_idx_t = BK.input_idx(pred_idx_arr)  # [bsize, slen]
            target_idx_t = pred_idx_t.gather(-1, mask_idxes)  # [bsize, ?]
            target_idx_t[(mask_valids<1.)] = 0  # make sure invalid ones in range
            # get loss
            pred_losses = BK.loss_nll(target_scores, target_idx_t)  # [bsize, ?]
            pred_loss_sum = (pred_losses * mask_valids).sum()
            pred_loss_count = mask_valids.sum()
            # argmax
            _, argmax_idxes = target_scores.max(-1)
            pred_corrs = (argmax_idxes == target_idx_t).float() * mask_valids
            pred_corr_count = pred_corrs.sum()
            return [[pred_loss_sum, pred_loss_count, pred_corr_count]]
