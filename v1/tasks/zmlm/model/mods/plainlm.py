#

# the module for plain lm
# todo(note): require directional encoders, like RNN with enc_rnn_sep_bidirection=True!

from typing import List, Dict
from msp.utils import Conf, zwarn, zlog
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, NoDropRop
from ..base import BaseModuleConf, BaseModule, LossHelper
from .embedder import Inputter

# -----
class PlainLMNodeConf(BaseModuleConf):
    def __init__(self):
        super().__init__()
        self.loss_lambda.val = 0.  # by default no such loss
        # ----
        # hidden layer
        self.hid_dim = 0
        self.hid_act = "elu"
        # pred options
        self.split_input_blm = True  # split the inputs into half for (l2r+r2l) for blm, otherwise input list or all l2r
        self.min_pred_rank = 1  # >= min word idx to pred for the masked ones
        self.max_pred_rank = 1  # <= max word idx to pred for the masked ones
        # tie weights?
        self.tie_input_embeddings = False  # tie all preds with input embeddings
        self.tie_bidirect_pred = False  # tie params for bidirection preds
        self.init_pred_from_pretrain = False

class PlainLMNode(BaseModule):
    def __init__(self, pc: BK.ParamCollection, input_dim: int, conf: PlainLMNodeConf, inputter: Inputter):
        super().__init__(pc, conf, name="PLM")
        self.conf = conf
        self.inputter = inputter
        self.input_dim = input_dim
        self.split_input_blm = conf.split_input_blm
        # this step is performed at the embedder, thus still does not influence the inputter
        self.add_root_token = self.inputter.embedder.add_root_token
        # vocab and padder
        vpack = inputter.vpack
        vocab_word = vpack.get_voc("word")
        # models
        real_input_dim = input_dim//2 if self.split_input_blm else input_dim
        if conf.hid_dim <= 0:  # no hidden layer
            self.l2r_hid_layer = self.r2l_hid_layer = None
            self.pred_input_dim = real_input_dim
        else:
            self.l2r_hid_layer = self.add_sub_node("l2r_h", Affine(pc, real_input_dim, conf.hid_dim, act=conf.hid_act))
            self.r2l_hid_layer = self.add_sub_node("r2l_h", Affine(pc, real_input_dim, conf.hid_dim, act=conf.hid_act))
            self.pred_input_dim = conf.hid_dim
        # todo(note): unk is the first one above real words
        self.pred_size = min(conf.max_pred_rank+1, vocab_word.unk)
        if conf.tie_input_embeddings:
            zwarn("Tie all preds in plm with input embeddings!!")
            self.l2r_pred = self.r2l_pred = None
            self.inputter_embed_node = self.inputter.embedder.get_node("word")
        else:
            self.l2r_pred = self.add_sub_node("l2r_p", Affine(pc, self.pred_input_dim, self.pred_size, init_rop=NoDropRop()))
            if conf.tie_bidirect_pred:
                self.r2l_pred = self.l2r_pred
            else:
                self.r2l_pred = self.add_sub_node("r2l_p", Affine(pc, self.pred_input_dim, self.pred_size, init_rop=NoDropRop()))
            self.inputter_embed_node = None
            if conf.init_pred_from_pretrain:
                npvec = vpack.get_emb("word")
                if npvec is None:
                    zwarn("Pretrained vector not provided, skip init pred embeddings!!")
                else:
                    with BK.no_grad_env():
                        self.l2r_pred.ws[0].copy_(BK.input_real(npvec[:self.pred_size].T))
                        self.r2l_pred.ws[0].copy_(BK.input_real(npvec[:self.pred_size].T))
                    zlog(f"Init pred embeddings from pretrained vectors (size={self.pred_size}).")

    # [bsize, slen, *]
    # todo(note): be careful that repr_t can be root-appended!!
    def loss(self, repr_t, orig_map: Dict, **kwargs):
        conf = self.conf
        _tie_input_embeddings = conf.tie_input_embeddings
        # --
        # specify input
        add_root_token = self.add_root_token
        # get from inputs
        if isinstance(repr_t, (list, tuple)):
            l2r_repr_t, r2l_repr_t = repr_t
        elif self.split_input_blm:
            l2r_repr_t, r2l_repr_t = BK.chunk(repr_t, 2, -1)
        else:
            l2r_repr_t, r2l_repr_t = repr_t, None
        # l2r and r2l
        word_t = BK.input_idx(orig_map["word"])  # [bs, rlen]
        slice_zero_t = BK.zeros([BK.get_shape(word_t, 0), 1]).long()  # [bs, 1]
        if add_root_token:
            l2r_trg_t = BK.concat([word_t, slice_zero_t], -1)  # pad one extra 0, [bs, rlen+1]
            r2l_trg_t = BK.concat([slice_zero_t, slice_zero_t, word_t[:,:-1]], -1)  # pad two extra 0 at front, [bs, 2+rlen-1]
        else:
            l2r_trg_t = BK.concat([word_t[:,1:], slice_zero_t], -1)  # pad one extra 0, but remove the first one, [bs, -1+rlen+1]
            r2l_trg_t = BK.concat([slice_zero_t, word_t[:,:-1]], -1)  # pad one extra 0 at front, [bs, 1+rlen-1]
        # gather the losses
        all_losses = []
        pred_range_min, pred_range_max = max(1, conf.min_pred_rank), self.pred_size-1
        if _tie_input_embeddings:
            pred_W = self.inputter_embed_node.E.E[:self.pred_size]  # [PSize, Dim]
        else:
            pred_W = None
        # get input embeddings for output
        for pred_name, hid_node, pred_node, input_t, trg_t in \
                    zip(["l2r", "r2l"], [self.l2r_hid_layer, self.r2l_hid_layer], [self.l2r_pred, self.r2l_pred],
                        [l2r_repr_t, r2l_repr_t], [l2r_trg_t, r2l_trg_t]):
            if input_t is None:
                continue
            # hidden
            hid_t = hid_node(input_t) if hid_node else input_t  # [bs, slen, hid]
            # pred: [bs, slen, Vsize]
            if _tie_input_embeddings:
                scores_t = BK.matmul(hid_t, pred_W.T)
            else:
                scores_t = pred_node(hid_t)
            # loss
            mask_t = ((trg_t >= pred_range_min) & (trg_t <= pred_range_max)).float()  # [bs, slen]
            trg_t.clamp_(max=pred_range_max)  # make it in range
            losses_t = BK.loss_nll(scores_t, trg_t) * mask_t  # [bs, slen]
            _, argmax_idxes = scores_t.max(-1)  # [bs, slen]
            corrs_t = (argmax_idxes == trg_t).float() * mask_t  # [bs, slen]
            # compile leaf loss
            one_loss = LossHelper.compile_leaf_info(pred_name, losses_t.sum(), mask_t.sum(), loss_lambda=1., corr=corrs_t.sum())
            all_losses.append(one_loss)
        return self._compile_component_loss("plm", all_losses)

# b zmlm/model/mods/plainlm:45
