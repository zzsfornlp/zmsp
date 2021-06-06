#

# UDEP

__all__ = [
    "ZDecoderUDEPConf", "ZDecoderUDEPNode", "ZDecoderUDEPHelper",
]

from typing import List
import numpy as np
import time
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab, SeqVocab
from msp2.utils import Constants, zlog, ZObject, zwarn, ConfEntryChoices
from ..common import *
from ..enc import *
from ..dec import *

# =====

class ZDecoderUDEPConf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        # for depth (distance to root)
        self.depth_conf = IdecSingleConf()  # one score for depth!
        self.depth_label_smoothing = 0.
        self.loss_depth = 1.0
        self.depth_nonroot_space = 0.  # how much space left for nonroot ones? 0 means binary!
        # for dep
        self.udep_conf: IdecConf = \
            ConfEntryChoices({"pair_emb": IdecPairwiseConf(), "pair_att": IdecAttConf()}, "pair_emb")  # argument
        self.loss_udep = 1.0
        self.udep_label_smoothing = 0.
        self.udep_loss_sample_neg = 0.05  # how much of neg to include: for each token, (n-1)/n is neg examples
        # special
        self.udep_pred_root_penalty = -0.  # penalty for root edges
        self.udep_train_use_cache = True

@node_reg(ZDecoderUDEPConf)
class ZDecoderUDEPNode(ZDecoder):
    def __init__(self, conf: ZDecoderUDEPConf, name: str, vocab_udep: SimpleVocab, ref_enc: ZEncoder, **kwargs):
        super().__init__(conf, name, **kwargs)
        conf: ZDecoderUDEPConf = self.conf
        self.vocab_udep = vocab_udep
        _enc_dim, _head_dim = ref_enc.get_enc_dim(), ref_enc.get_head_dim()
        # --
        self.helper = ZDecoderUDEPHelper(conf, self.vocab_udep)
        self._label_idx_root = vocab_udep.get("root")  # get root's index for decoding
        # --
        # nodes
        self.depth_node: IdecNode = conf.depth_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=1)
        self.udep_node: IdecNode = conf.udep_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=len(vocab_udep))
        # --

    def get_idec_nodes(self):
        return [self.depth_node, self.udep_node]

    # return List[tensor], bool
    def layer_end(self, med: ZMediator):
        _name = self.name
        scores = {}
        lidx = med.lidx
        rets = []
        if self.depth_node.need_app_layer(lidx):
            depth_score, depth_rets = self.depth_node.forward(med)
            scores[(_name, 'depth')] = depth_score
            rets.append(depth_rets)
        if self.udep_node.need_app_layer(lidx):
            udep_score, udep_rets = self.udep_node.forward(med)
            scores[(_name, 'udep')] = udep_score
            rets.append(udep_rets)
        return scores, rets, (lidx >= self.max_app_lidx)

    # --
    # depth score
    def _loss_depth(self, med: ZMediator, mask_expr, expr_depth):
        conf: ZDecoderUDEPConf = self.conf
        # --
        all_depth_scores = med.main_scores.get((self.name, "depth"))  # [*, slen]
        all_depth_losses = []
        for one_depth_scores in all_depth_scores:
            one_losses = BK.loss_binary(one_depth_scores.squeeze(-1), expr_depth, label_smoothing=conf.depth_label_smoothing)
            all_depth_losses.append(one_losses)
        depth_loss_results = self.depth_node.helper.loss(all_losses=all_depth_losses)
        loss_items = []
        for loss_t, loss_alpha, loss_name in depth_loss_results:
            one_depth_item = LossHelper.compile_leaf_loss(
                "depth"+loss_name, (loss_t*mask_expr).sum(), mask_expr.sum(), loss_lambda=(loss_alpha*conf.loss_depth))
            loss_items.append(one_depth_item)
        return loss_items

    # udep score
    def _loss_udep(self, med: ZMediator, mask_expr, expr_udep):
        conf: ZDecoderUDEPConf = self.conf
        # --
        all_udep_scores = med.main_scores.get((self.name, "udep"))  # [*, slen, slen, L]
        all_udep_losses = []
        for one_udep_scores in all_udep_scores:
            one_losses = BK.loss_nll(one_udep_scores, expr_udep, label_smoothing=conf.udep_label_smoothing)
            all_udep_losses.append(one_losses)
        udep_loss_results = self.udep_node.helper.loss(all_losses=all_udep_losses)
        # --
        _loss_weights = ((BK.rand(expr_udep.shape) < conf.udep_loss_sample_neg) | (expr_udep>0)).float() \
                        * mask_expr.unsqueeze(-1) * mask_expr.unsqueeze(-2)  # [*, slen, slen]
        # --
        loss_items = []
        for loss_t, loss_alpha, loss_name in udep_loss_results:
            one_udep_item = LossHelper.compile_leaf_loss(
                "udep"+loss_name, (loss_t*_loss_weights).sum(), _loss_weights.sum(), loss_lambda=(loss_alpha*conf.loss_udep))
            loss_items.append(one_udep_item)
        return loss_items

    # finally loss
    def loss(self, med: ZMediator):
        conf: ZDecoderUDEPConf = self.conf
        insts, mask_expr = med.insts, med.get_mask_t()
        # --
        # first prepare golds
        expr_depth, expr_udep = self.helper.prepare(insts, conf.udep_train_use_cache)
        loss_items = []
        if conf.loss_depth > 0.:
            loss_items.extend(self._loss_depth(med, mask_expr, expr_depth))
        if conf.loss_udep > 0.:
            loss_items.extend(self._loss_udep(med, mask_expr, expr_udep))
        # --
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss

    # --
    # predict
    def predict(self, med: ZMediator):
        conf: ZDecoderUDEPConf = self.conf
        # --
        # depth scores
        all_depth_raw_scores = med.main_scores.get((self.name, "depth"), [])  # [*, slen]
        all_depth_logprobs = [BK.logsigmoid(z.squeeze(-1)) for z in all_depth_raw_scores]
        if len(all_depth_logprobs) > 0:
            final_depth_logprobs = self.depth_node.helper.pred(all_logprobs=all_depth_logprobs)  # [*, slen]
        else:
            final_depth_logprobs = -99.
        # udep scores
        all_udep_raw_scores = med.main_scores.get((self.name, "udep"), [])  # [*, slen, slen, L]
        all_udep_logprobs = [z.log_softmax(-1) for z in all_udep_raw_scores]
        final_udep_logprobs = self.udep_node.helper.pred(all_logprobs=all_udep_logprobs)  # [*, slen, slen, L]
        # prepare final scores
        final_scores = BK.pad(final_udep_logprobs, [0,0,1,0,1,0], value=Constants.REAL_PRAC_MIN)  # [*, 1+slen, 1+slen, L]
        final_scores[:, :, :, 0] = Constants.REAL_PRAC_MIN  # force no 0!!
        final_scores[:, 1:, 0, self._label_idx_root] = (final_depth_logprobs + conf.udep_pred_root_penalty)  # assign root score
        # decode
        from msp2.tools.algo.nmst import mst_unproj  # decoding algorithm
        insts = med.insts
        arr_lengths = np.asarray([len(z.sent)+1 for z in insts])  # +1 for arti-root
        arr_scores = BK.get_value(final_scores)  # [*, 1+slen, 1+slen, L]
        arr_ret_heads, arr_ret_labels, arr_ret_scores = mst_unproj(arr_scores, arr_lengths, labeled=True)  # [*, 1+slen]
        self.helper.put_results(insts, [arr_ret_heads, arr_ret_labels, arr_ret_scores])
        # --
        return {}

# helper
class ZDecoderUDEPHelper(ZDecoderHelper):
    def __init__(self, conf: ZDecoderUDEPConf, vocab_udep: SimpleVocab):
        self.conf = conf
        self.vocab_udep = vocab_udep
        # --

    # prepare for one inst
    def _prep_inst(self, inst):
        sent = inst.sent
        slen = len(sent)
        # --
        tree = sent.tree_dep
        # first on depth (normalized by max-depth)
        tree_depths = tree.depths
        max_depth = max(tree_depths)
        assert max_depth >= 1
        # note: real-root got 1, leaf got 0, others in between
        _tmp_depth = np.asarray(tree.depths)
        depth_arr = 1. - (_tmp_depth-1) / max(1, max_depth-1)  # real-root's depth==1
        depth_arr *= self.conf.depth_nonroot_space  # squeeze non-root!
        depth_arr[_tmp_depth==1] = 1.  # reset the root!
        # then on labels
        _lab_arr = tree.label_matrix  # (m,h) [slen, slen+1]
        udep_arr = _lab_arr[:, 1:]  # (m,h) [slen, slen], arti-root not included!
        return ZObject(slen=slen, depth_arr=depth_arr, udep_arr=udep_arr)

    # prepare inputs
    def prepare(self, insts: List, use_cache: bool):
        # get info
        zobjs = ZDecoderHelper.get_zobjs(insts, self._prep_inst, use_cache, f"_cache_udep")
        # then
        bsize, mlen = len(insts), max(z.slen for z in zobjs) if len(zobjs)>0 else 1
        batched_shape = (bsize, mlen)
        arr_depth = np.full(batched_shape, 0., dtype=np.float)  # [*, slen]
        arr_udep = np.full(batched_shape+(mlen,), 0, dtype=np.int)  # [*, slen_m, slen_h]
        for zidx, zobj in enumerate(zobjs):
            zlen = zobj.slen
            arr_depth[zidx, :zlen] = zobj.depth_arr
            arr_udep[zidx, :zlen, :zlen] = zobj.udep_arr
        expr_depth = BK.input_real(arr_depth)  # [*, slen]
        expr_udep = BK.input_idx(arr_udep)  # [*, slen_m, slen_h]
        return expr_depth, expr_udep

    # heads[*, slen], labels[*, slen], scores[*, slen]
    def put_results(self, insts: List[Sent], all_arrs):
        vocab_udep = self.vocab_udep
        arr_ret_heads, arr_ret_labels, arr_ret_scores = all_arrs
        for bidx, inst in enumerate(insts):
            # todo(+W): if there are repeated sentences, may overwrite previous ones, but maybe does not matter!
            cur_len_p1 = 1 + len(inst.sent)
            list_dep_heads = arr_ret_heads[bidx][1:cur_len_p1].tolist()
            list_dep_lidxes = arr_ret_labels[bidx][1:cur_len_p1].tolist()
            list_dep_labels = vocab_udep.seq_idx2word(list_dep_lidxes)
            inst.sent.build_dep_tree(list_dep_heads, list_dep_labels)
            inst.sent.tree_dep.seq_label.set_idxes(list_dep_lidxes)
        # --

# --
# b msp2/tasks/zmtl/modules/dec_comps/udep:197
