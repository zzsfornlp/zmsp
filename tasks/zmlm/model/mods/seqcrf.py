#

# sequence labeling module with linear CRF
# adopted from NCRF++: https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py (17 Dec 2018)

from typing import List, Dict, Tuple
import numpy as np

from msp.utils import Conf, Random, zlog, JsonRW, zfatal, zwarn
from msp.nn import BK
from msp.nn.layers import Affine, NoDropRop

from ..base import BaseModuleConf, BaseModule, LossHelper
from .embedder import Inputter
from ...data.insts import GeneralSentence, SeqField

import torch

# extra ones
START_TAG = -2
STOP_TAG = -1

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M

# --
class SeqCrfNodeConf(BaseModuleConf):
    def __init__(self):
        super().__init__()
        self.loss_lambda.val = 0.  # by default no such loss
        # --
        # hidden layer
        self.hid_dim = 0
        self.hid_act = "elu"
        # for loss function
        self.div_by_tok = True

class SeqCrfNode(BaseModule):
    def __init__(self, pc: BK.ParamCollection, pname: str, input_dim: int, conf: SeqCrfNodeConf, inputter: Inputter):
        super().__init__(pc, conf, name="CRF")
        self.conf = conf
        self.inputter = inputter
        self.input_dim = input_dim
        # this step is performed at the embedder, thus still does not influence the inputter
        self.add_root_token = self.inputter.embedder.add_root_token
        # --
        self.pname = pname
        self.attr_name = pname + "_seq"  # attribute name in Instance
        self.vocab = inputter.vpack.get_voc(pname)
        # todo(note): we must make sure that 0 means NAN
        assert self.vocab.non == 0
        # models
        if conf.hid_dim <= 0:  # no hidden layer
            self.hid_layer = None
            self.pred_input_dim = input_dim
        else:
            self.hid_layer = self.add_sub_node("hid", Affine(pc, input_dim, conf.hid_dim, act=conf.hid_act))
            self.pred_input_dim = conf.hid_dim
        self.tagset_size = self.vocab.unk  # todo(note): UNK is the prediction boundary
        self.pred_layer = self.add_sub_node("pr", Affine(pc, self.pred_input_dim, self.tagset_size+2, init_rop=NoDropRop()))
        # transition matrix
        init_transitions = np.zeros([self.tagset_size+2, self.tagset_size+2])
        init_transitions[:, START_TAG] = -10000.0
        init_transitions[STOP_TAG, :] = -10000.0
        init_transitions[:, 0] = -10000.0
        init_transitions[0, :] = -10000.0
        self.transitions = self.add_param("T", (self.tagset_size+2, self.tagset_size+2), init=init_transitions)

    # score
    def _score(self, repr_t):
        if self.hid_layer is None:
            hid_t = repr_t.contiguous()
        else:
            hid_t = self.hid_layer(repr_t.contiguous())
        out_t = self.pred_layer(hid_t)
        return out_t

    # loss
    def loss(self, insts: List[GeneralSentence], repr_t, mask_t, **kwargs):
        conf = self.conf
        if self.add_root_token:
            repr_t = repr_t[:,1:]
            mask_t = mask_t[:,1:]
        # score
        scores_t = self._score(repr_t)  # [bs, rlen, D]
        # get gold
        gold_pidxes = np.zeros(BK.get_shape(mask_t), dtype=np.long)  # [bs, ?+rlen]
        for bidx, inst in enumerate(insts):
            cur_seq_idxes = getattr(inst, self.attr_name).idxes
            gold_pidxes[bidx, :len(cur_seq_idxes)] = cur_seq_idxes
        # get loss
        gold_pidxes_t = BK.input_idx(gold_pidxes)
        nll_loss_sum = self.neg_log_likelihood_loss(scores_t, mask_t.bool(), gold_pidxes_t)
        if not conf.div_by_tok:  # otherwise div by sent
            nll_loss_sum *= (mask_t.sum() / len(insts))
        # compile loss
        crf_loss = LossHelper.compile_leaf_info("crf", nll_loss_sum.sum(), mask_t.sum())
        return self._compile_component_loss(self.pname, [crf_loss])

    # predict
    def predict(self, insts: List[GeneralSentence], repr_t, mask_t, **kwargs):
        conf = self.conf
        if self.add_root_token:
            repr_t = repr_t[:,1:]
            mask_t = mask_t[:,1:]
        # score
        scores_t = self._score(repr_t)  # [bs, rlen, D]
        # decode
        _, decode_idx = self._viterbi_decode(scores_t, mask_t.bool())
        decode_idx_arr = BK.get_value(decode_idx)  # [bs, rlen]
        for one_bidx, one_inst in enumerate(insts):
            one_pidxes = decode_idx_arr[one_bidx].tolist()[:len(one_inst)]
            one_pseq = SeqField(None)
            one_pseq.build_vals(one_pidxes, self.vocab)
            one_inst.add_item("pred_"+self.attr_name, one_pseq, assert_non_exist=False)
        return

    # =====

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        # print feats.view(seq_len, tag_size)
        assert (tag_size == self.tagset_size+2)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            # print cur_partition.data

                # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            ## effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        # mask =  (1 - mask.long()).byte()
        mask =  (1 - mask.long()).bool()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        # print "init part:",partition.size()
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            ## forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            # print "cur value:", cur_values.size()
            partition, cur_bp = torch.max(cur_values, 1)
            # print "partsize:",partition.size()
            # exit(0)
            # print partition
            # print cur_bp
            # print "one best, ",idx
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        # exit(0)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1,0).contiguous() ## (batch_size, seq_len. tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        # pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        # if self.gpu:
        #     pad_zero = pad_zero.cuda()
        pad_zero = BK.zeros((batch_size, tag_size)).long()
        back_points.append(pad_zero)
        back_points  =  torch.cat(back_points).view(seq_len, batch_size, tag_size)

        ## select end ids in STOP_TAG
        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size)
        back_points = back_points.transpose(1,0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last
        back_points.scatter_(1, last_position, insert_last)
        # print "bp:",back_points
        # exit(0)
        back_points = back_points.transpose(1,0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        # decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        # if self.gpu:
        #     decode_idx = decode_idx.cuda()
        decode_idx = BK.zeros((seq_len, batch_size)).long()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1,0)
        return path_score, decode_idx

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        ## convert tag value into a new format, recorded label bigram information to index
        # new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        # if self.gpu:
        #     new_tags = new_tags.cuda()
        new_tags = BK.zeros((batch_size, seq_len)).long()
        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:,0] =  (tag_size - 2)*tag_size + tags[:,0]

            else:
                new_tags[:,idx] =  tags[:,idx-1]*tag_size + tags[:,idx]

        ## transition for label to STOP_TAG
        end_transition = self.transitions[:,STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        ## length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ## index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1,0).contiguous().view(seq_len, batch_size, 1)
        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1,0))

        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        # nonegative log likelihood
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        # print "batch, f:", forward_score.data[0], " g:", gold_score.data[0], " dis:", forward_score.data[0] - gold_score.data[0]
        # exit(0)
        return forward_score - gold_score
