#

from typing import List
import numpy as np
from copy import deepcopy

from msp.utils import Conf, Random, Constants, MathHelper
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, BiAffineScorer, Embedding, NoDropRop, FreezeRop
from msp.zext.seq_helper import DataPadder
from msp.zext.ie import HLabelVocab, HLabelIdx

from ..common.data import DocInstance, Sentence, Mention, HardSpan, EntityFiller, Event
from ..common.data_helper import ERE_ARG_BUDGETS, RAMS_ARG_BUDGETS

from ..models2.labeler import HLabelNodeConf, HLabelNode

# =====
# task-specific Adaptor layer on encoder outputs

#
class TaskSpecAdp(BasicNode):
    def __init__(self, pc, input_dims, extra_input_dims: List, output_dim, act="elu"):
        super().__init__(pc, None, None)
        # check inputs
        bert_input_dims, basic_dim = input_dims
        bert_dim, bert_fold = bert_input_dims
        #
        final_input_dims = [bert_dim, ]
        self.has_basic = (basic_dim is not None)
        if self.has_basic:
            final_input_dims.append(basic_dim)
        final_input_dims.extend(extra_input_dims)
        self.bert_gamma = self.add_param("AG", (), 1.)  # scalar
        self.bert_lambdas = self.add_param("AL", (), [1./bert_fold] * bert_fold)  # [fold]
        self.fnode = self.add_sub_node("f", Affine(pc, final_input_dims, output_dim, act=act))

    # [*, fold, D], [*, D'] or None, *[*, ?]
    def __call__(self, bert_t, basic_t, extra_t_list):
        lambdas_softmax = BK.softmax(self.bert_lambdas, -1).unsqueeze(-1)  # [fold, 1]
        weighted_bert_t = (bert_t * lambdas_softmax).sum(-2) * self.bert_gamma  # [*, D]
        input_list = [weighted_bert_t]
        if self.has_basic:
            input_list.append(basic_t)
        input_list.extend(extra_t_list)
        output = self.fnode(input_list)
        return output  # [*, hid]

    def get_output_dims(self, *input_dims):
        return self.fnode.get_output_dims(*input_dims)

# =====
# main components: mainly in the form of multitask
# todo(note): make it simple -> use gold in loss(*), pred in predict(*)

# helper for preparation
class PrepHelper:
    # some helpers
    random_sample_stream = Random.stream(Random.random_sample)
    padder_idxes = DataPadder(2, pad_vals=0, mask_range=2)
    padder_idxes_nomask = DataPadder(2, pad_vals=0)
    padder_items = DataPadder(2, pad_vals=None)
    padder_roles = DataPadder(3, pad_vals=0)

    # =====
    # prepare for mention extraction

    # add target idxes and items for one sent: being able to add multiple positive instances for one position
    @staticmethod
    def add_targets(sent, sent_target_f, offset: int, neg_rate: float):
        target_items = sent_target_f(sent)
        if target_items is None:
            return [], []
        else:  # only add if there are annotations
            out_idxes, out_items = [], []
            final_offset = offset - 1  # exclude ROOT
            slen = sent.length
            is_positive = [False] * slen
            # first positive examples
            for one_target_item in target_items:
                cur_head_wid = one_target_item.mention.hard_span.head_wid
                is_positive[cur_head_wid] = True
                out_idxes.append(final_offset+cur_head_wid)
                out_items.append(one_target_item)
            # then neg examples
            if neg_rate>=1.:
                # include all
                for i in range(1, slen):
                    if not is_positive[i]:
                        out_idxes.append(final_offset+i)
                        out_items.append(None)
            elif neg_rate>0.:
                # sample neg
                tmp_random_sample_stream = PrepHelper.random_sample_stream
                for i in range(1, slen):
                    if not is_positive[i] and next(tmp_random_sample_stream)<neg_rate:
                        out_idxes.append(final_offset+i)
                        out_items.append(None)
            return out_idxes, out_items

    # return offsets_t, masks_t, sdists_t, items_arr, labels_t [bs, ?]
    @staticmethod
    def prep_targets(ms_items: List, sent_target_f, include_center: bool, include_outside: bool,
                     neg_rate_center: float, neg_rate_outside: float, return_items: bool):
        # collect
        ret_all_idxes, ret_all_items, ret_all_sdists = [], [], []
        for one_item in ms_items:
            one_sents = one_item.sents
            one_offsets = one_item.offsets
            one_center_idx = one_item.center_idx
            ret_idxes, ret_items, ret_sdists = [], [], []
            if include_center:
                center_sent, center_offset = one_sents[one_center_idx], one_offsets[one_center_idx]
                this_idxes, this_items = PrepHelper.add_targets(center_sent, sent_target_f, center_offset, neg_rate_center)
                ret_idxes.extend(this_idxes)
                ret_items.extend(this_items)
                ret_sdists.extend([0] * len(this_idxes))
            if include_outside:
                for this_idx, this_sent in enumerate(one_sents):
                    if this_idx == one_center_idx:
                        continue
                    this_offset = one_offsets[this_idx]
                    this_idxes, this_items = PrepHelper.add_targets(this_sent, sent_target_f, this_offset, neg_rate_outside)
                    ret_idxes.extend(this_idxes)
                    ret_items.extend(this_items)
                    ret_sdists.extend([this_idx-one_center_idx] * len(this_idxes))  # sdist: ef-evt
            ret_all_idxes.append(ret_idxes)
            ret_all_items.append(ret_items)
            ret_all_sdists.append(ret_sdists)
        # batch
        offsets_arr, masks_arr = PrepHelper.padder_idxes.pad(ret_all_idxes)  # [bs, ?]
        sdists_arr, _ = PrepHelper.padder_idxes.pad(ret_all_sdists)
        # if all empty?
        if offsets_arr.shape[-1] == 0:
            bsize = len(ms_items)
            tmp_zeros = BK.zeros((bsize, 0)).long()
            if return_items:
                return tmp_zeros, tmp_zeros.float(), tmp_zeros, np.asarray([], dtype=object).reshape((bsize, 0)), tmp_zeros
            else:
                return tmp_zeros, tmp_zeros.float(), tmp_zeros, None, None
        else:
            offsets_t = BK.input_idx(offsets_arr)
            masks_t = BK.input_real(masks_arr)
            sdists_t = BK.input_idx(sdists_arr)
            if return_items:
                items_arr, _ = PrepHelper.padder_items.pad(ret_all_items)
                # todo(+N): 0 for nil, also explicit hlidx2idx
                labels_list = [0 if z is None else z.type_idx.get_idx(-1) for z in items_arr.reshape(-1)]
                labels_t = BK.input_idx(labels_list).view(items_arr.shape)
            else:
                items_arr = labels_t = None
            return offsets_t, masks_t, sdists_t, items_arr, labels_t

    # =====
    # especially prepare for arg extraction (training time), return separate center and outside args
    # todo(note): special mode -- using predictions in training to consider more cands;
    #  if use_pred_ef, we might need to dynamically prepare these at each run!!
    @staticmethod
    def prep_train_args_one(one_item):
        one_sents = one_item.sents
        one_offsets = one_item.offsets
        one_center_idx = one_item.center_idx
        # -----
        # collect center evts
        center_sent, center_offset = one_sents[one_center_idx], one_offsets[one_center_idx]
        # -----
        # no adding if no annotations (missing one of them)
        if center_sent.events is None or center_sent.entity_fillers is None:
            return ([],) * 10
        # idxes for one instance of multi-sent
        one_evt_idxes, one_center_ef_idxes, one_outside_ef_idxes = [], [], []
        one_evt_types, one_center_ef_types, one_outside_ef_types = [], [], []
        one_center_sdists, one_outside_sdists = [], []
        one_center_roles, one_outside_roles = [], []  # ef-evt pairs, 2d
        # -----
        center_evt_iden_map = {}  # (evt-hwid, evt-type) -> idx in "one_evt_idxes" (merge same events)
        center_evt_role_maps = []  # [evt] of {id(ef) -> set[roles]} (consider multiple but not repeated roles)
        for one_evt in center_sent.events:
            cur_head_hwid, cur_type_lidx = one_evt.mention.hard_span.head_wid, one_evt.type_idx.get_idx(-1)
            cur_evt_key = (cur_head_hwid, cur_type_lidx)
            # first put evt
            if cur_evt_key in center_evt_iden_map:  # merge those with same type and position
                cur_evt_role_map = center_evt_role_maps[center_evt_iden_map[cur_evt_key]]
            else:  # make a new one
                center_evt_iden_map[cur_evt_key] = len(one_evt_idxes)
                one_evt_idxes.append(cur_head_hwid + center_offset - 1)  # exclude ROOT
                one_evt_types.append(cur_type_lidx)
                cur_evt_role_map = {}
                center_evt_role_maps.append(cur_evt_role_map)
            # then collect args
            for arg_link in one_evt.links:
                arg_link_ef_id = id(arg_link.ef)
                if arg_link_ef_id not in cur_evt_role_map:  # collect all roles between each pair
                    cur_evt_role_map[arg_link_ef_id] = set()
                cur_evt_role_map[arg_link_ef_id].add(arg_link.role_idx.get_idx(-1))
        # -----
        # midterm preparation
        efid_max_roles = {}  # id(ef) -> MaxNumRoles(Repeate times)
        center_evt_rolelist_maps = []
        for one_map in center_evt_role_maps:
            one_rolelist_map = {}
            for efid, role_set in one_map.items():
                role_list = list(role_set)
                one_rolelist_map[efid] = role_list
                efid_max_roles[efid] = max(efid_max_roles.get(efid, 0), len(role_list))
            center_evt_rolelist_maps.append(one_rolelist_map)
        # -----
        # collect all efs as args (may be repeated for multiple roles)
        for cur_sent_idx, cur_sent in enumerate(one_sents):
            # assign to-fill lists
            if cur_sent_idx == one_center_idx:
                cur_ef_idxes, cur_ef_types, cur_sdists, cur_roles = \
                    one_center_ef_idxes, one_center_ef_types, one_center_sdists, one_center_roles
            else:
                cur_ef_idxes, cur_ef_types, cur_sdists, cur_roles = \
                    one_outside_ef_idxes, one_outside_ef_types, one_outside_sdists, one_outside_roles
            # for all the efs
            cur_offset = one_offsets[cur_sent_idx]
            cur_sdist = cur_sent_idx - one_center_idx
            for one_ef in cur_sent.entity_fillers:
                cur_head_hwid, cur_type_lidx = one_ef.mention.hard_span.head_wid, one_ef.type_idx.get_idx(-1)
                cur_efidx = cur_head_hwid + cur_offset - 1
                cur_efid = id(one_ef)
                cur_mr_len = efid_max_roles.get(cur_efid, 1)  # as one neg example if no hit
                # add the info
                cur_ef_idxes.extend([cur_efidx] * cur_mr_len)
                cur_ef_types.extend([cur_type_lidx] * cur_mr_len)
                cur_sdists.extend([cur_sdist] * cur_mr_len)
                # collect the roles
                this_roles = [[None] * len(center_evt_rolelist_maps) for _ in range(cur_mr_len)]  # ef * evt
                for this_evt_idx, this_evt_rolelist_map in enumerate(center_evt_rolelist_maps):
                    this_evt_rolelist = this_evt_rolelist_map.get(cur_efid, [0])  # by default all NIL=0
                    for this_ef_idx in range(cur_mr_len):
                        this_roles[this_ef_idx][this_evt_idx] = this_evt_rolelist[this_ef_idx % len(this_evt_rolelist)]
                cur_roles.extend(this_roles)
        return one_evt_idxes, one_center_ef_idxes, one_outside_ef_idxes, \
               one_evt_types, one_center_ef_types, one_outside_ef_types, \
               one_center_sdists, one_outside_sdists, \
               one_center_roles, one_outside_roles

    @staticmethod
    def prep_train_args(ms_items: List, dynamic_prepare=False):
        # collect
        # in-sent event idxes; in-sent/outside ef idxes/types/sdists
        all_evt_idxes, all_center_ef_idxes, all_outside_ef_idxes = [], [], []
        all_evt_types, all_center_ef_types, all_outside_ef_types = [], [], []
        all_center_sdists, all_outside_sdists = [], []  # ef sdists
        all_center_roles, all_outside_roles = [], []  # ef-evt pairs, 3d
        # loop over batch size
        for one_item in ms_items:
            # if not dynamic, we can use the caching: PrepHelper.prep_train_args_one(one_item)
            one_evt_idxes, one_center_ef_idxes, one_outside_ef_idxes, \
            one_evt_types, one_center_ef_types, one_outside_ef_types, \
            one_center_sdists, one_outside_sdists, \
            one_center_roles, one_outside_roles = \
                (PrepHelper.prep_train_args_one(one_item) if dynamic_prepare else one_item.arg_pack)
            # add them
            all_evt_idxes.append(one_evt_idxes)
            all_center_ef_idxes.append(one_center_ef_idxes)
            all_outside_ef_idxes.append(one_outside_ef_idxes)
            all_evt_types.append(one_evt_types)
            all_center_ef_types.append(one_center_ef_types)
            all_outside_ef_types.append(one_outside_ef_types)
            all_center_sdists.append(one_center_sdists)
            all_outside_sdists.append(one_outside_sdists)
            all_center_roles.append(one_center_roles)
            all_outside_roles.append(one_outside_roles)
        # =====
        # pad and batch
        all_evt_idxes_arr, all_evt_masks_arr = PrepHelper.padder_idxes.pad(all_evt_idxes)
        all_center_ef_idxes_arr, all_center_ef_masks_arr = PrepHelper.padder_idxes.pad(all_center_ef_idxes)
        all_outside_ef_idxes_arr, all_outside_ef_masks_arr = PrepHelper.padder_idxes.pad(all_outside_ef_idxes)
        all_evt_types_arr, _ = PrepHelper.padder_idxes_nomask.pad(all_evt_types)
        all_center_ef_types_arr, _ = PrepHelper.padder_idxes_nomask.pad(all_center_ef_types)
        all_outside_ef_types_arr, _ = PrepHelper.padder_idxes_nomask.pad(all_outside_ef_types)
        all_center_sdists_arr, _ = PrepHelper.padder_idxes_nomask.pad(all_center_sdists)
        all_outside_sdists_arr, _ = PrepHelper.padder_idxes_nomask.pad(all_outside_sdists)
        # all_center_roles_arr, _ = PrepHelper.padder_roles.pad(all_center_roles)  # [bs, ef, evt]
        # all_outside_roles_arr, _ = PrepHelper.padder_roles.pad(all_outside_roles)  # [bs, ef, evt]
        # todo(note): here we want specific shape, using Padder will not ensure this
        all_center_roles_arr = np.zeros([len(ms_items), all_center_ef_idxes_arr.shape[-1], all_evt_idxes_arr.shape[-1]], dtype=np.long)
        all_outside_roles_arr = np.zeros([len(ms_items), all_outside_ef_idxes_arr.shape[-1], all_evt_idxes_arr.shape[-1]], dtype=np.long)
        for to_fill_arr, to_assign_lists in zip([all_center_roles_arr,all_outside_roles_arr], [all_center_roles,all_outside_roles]):
            for bs_idx, bs_assign in enumerate(to_assign_lists):
                bs_fill = to_fill_arr[bs_idx]
                for ef_idx, ef_assign in enumerate(bs_assign):
                    ef_fill = bs_fill[ef_idx]
                    ef_fill[:len(ef_assign)] = ef_assign
        # =====
        # convert to tensor
        def _arr2idx_t(arr): return BK.zeros(arr.shape).long() if arr.size==0 else BK.input_idx(arr)
        def _arr2real_t(arr): return BK.zeros(arr.shape).float() if arr.size==0 else BK.input_real(arr)
        #
        evt_pack = _arr2idx_t(all_evt_idxes_arr), _arr2real_t(all_evt_masks_arr), _arr2idx_t(all_evt_types_arr)
        center_ef_pack = _arr2idx_t(all_center_ef_idxes_arr), _arr2real_t(all_center_ef_masks_arr), _arr2idx_t(all_center_ef_types_arr),\
                         _arr2idx_t(all_center_sdists_arr), _arr2idx_t(all_center_roles_arr)
        outside_ef_pack = _arr2idx_t(all_outside_ef_idxes_arr), _arr2real_t(all_outside_ef_masks_arr), _arr2idx_t(all_outside_ef_types_arr),\
                          _arr2idx_t(all_outside_sdists_arr), _arr2idx_t(all_outside_roles_arr)
        return evt_pack, center_ef_pack, outside_ef_pack

# =====
# extract mentions on head words

class MentionExtractorConf(Conf):
    def __init__(self):
        self.hidden_dim = 512
        # training
        self.train_neg_rate = 0.5
        # testing
        self.dec_topk_label = 1  # consider how many types for decoding
        self.dec_extra_prob_thresh = 0.5  # extra types' probs should be >= best*thresh
        self.nil_penalty = 0.  # penalty for nil(idx=0)
        # special mode testing
        # read gold positions in testing, only predict labels (may also want to make nil_penalty large in this mode)
        self.pred_use_gold_posi = False
        # use the special HLabelNode?
        self.use_hlnode = False  # use hlnode for prediction
        self.hl_lab_conf = HLabelNodeConf()
        # max prediction numbers compared to sent length
        self.pred_sent_ratio = 1.

    def do_validate(self):
        self.hl_lab_conf.n_dim = self.hidden_dim

class MentionExtractor(BasicNode):
    def __init__(self, pc, conf: MentionExtractorConf, vocab: HLabelVocab, extract_type: str, input_enc_dims):
        super().__init__(pc, None, None)
        self.conf = conf
        self.vocab = vocab
        assert vocab.nil_as_zero
        VOCAB_LAYER = -1  # todo(note): simply use final largest layer
        self.lidx2hlidx = vocab.layered_hlidx[VOCAB_LAYER]  # int-idx -> HLabelIdx
        self.output_size = len(self.lidx2hlidx)
        #
        self.extract_type = extract_type
        self.gold_target_f = {"evt": lambda x: x.events, "ef": lambda x: x.entity_fillers}[extract_type]
        self.pred_target_f = lambda x: []
        self.pred_set_f = {"evt": self.set_evt, "ef": self.set_ef}[extract_type]
        self.item_creator = {"evt": self.evt_creator, "ef": self.ef_creator}[extract_type]
        #
        self.adp = self.add_sub_node('adp', TaskSpecAdp(pc, input_enc_dims, [], conf.hidden_dim))
        adp_hidden_size = self.adp.get_output_dims()[0]
        self.dec_extra_logprob_thresh = float(np.log(conf.dec_extra_prob_thresh))  # already neg number
        # predictor
        if conf.use_hlnode:
            self.predictor = self.add_sub_node('pred', HLabelNode(pc, conf.hl_lab_conf, vocab))
        else:
            self.predictor = self.add_sub_node('pred', Affine(pc, adp_hidden_size, self.output_size, init_rop=NoDropRop()))

    # =====
    def set_ef(self, sent, efs): sent.pred_entity_fillers = efs
    def set_evt(self, sent, evts): sent.pred_events = evts

    def ef_creator(self, partial_id: str, m: Mention, hlidx: HLabelIdx, logprob: float):
        # todo(warn): here does not distinguish entity or filler
        return EntityFiller("ef-"+partial_id, m, str(hlidx), None, True, type_idx=hlidx, score=logprob)

    def evt_creator(self, partial_id: str, m: Mention, hlidx: HLabelIdx, logprob: float):
        return Event("evt-"+partial_id, m, str(hlidx), type_idx=hlidx, score=logprob)
    # =====

    def loss(self, ms_items: List, bert_expr, basic_expr, margin=0.):
        conf = self.conf
        bsize = len(ms_items)
        # build targets
        offsets_t, masks_t, _, items_arr, labels_t = PrepHelper.prep_targets(
            ms_items, self.gold_target_f, True, False, conf.train_neg_rate, 0., True)
        # -----
        # return 0 if all no targets
        if BK.get_shape(offsets_t, -1) == 0:
            zzz = BK.zeros([])
            return [[zzz, zzz, zzz]]
        # -----
        arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
        sel_bert_t = bert_expr[arange_t, offsets_t]  # [bsize, ?, Fold, D]
        sel_basic_t = None if basic_expr is None else basic_expr[arange_t, offsets_t]  # [bsize, ?, D']
        hiddens = self.adp(sel_bert_t, sel_basic_t, [])  # [bsize, ?, D"]
        if conf.use_hlnode:
            # NIL_IDX = self.vocab.nil_idx
            # input_gold_idxes = np.asarray([NIL_IDX if z is None else z.type_idx for z in items_arr.reshape(-1)]).reshape(items_arr.shape)
            # todo(note): no margin employed here, simply because of laziness
            losses = self.predictor.loss(hiddens, masks_t, labels_t, margin=margin)[0]
            # loss_sum, loss_count, gold_count
            assert len(losses)==1
            losses[0].append((labels_t>0).float().sum())  # add gold count
            return losses
        else:
            # build loss
            logits = self.predictor(hiddens)  # [bsize, ?, Out]
            log_probs = BK.log_softmax(logits, -1)
            picked_log_probs = - BK.gather_one_lastdim(log_probs, labels_t).squeeze(-1)  # [bsize, ?]
            masked_losses = picked_log_probs * masks_t
            # loss_sum, loss_count, gold_count
            return [[masked_losses.sum(), masks_t.sum(), (labels_t>0).float().sum()]]

    def predict(self, ms_items: List, bert_expr, basic_expr, constrain_types=None):
        conf = self.conf
        bsize = len(ms_items)
        # build targets
        if conf.pred_use_gold_posi:
            # only get gold ones in this mode!
            offsets_t, masks_t, _, _, _ = PrepHelper.prep_targets(ms_items, self.gold_target_f, True, False, 0., 0., False)
        else:
            offsets_t, masks_t, _, _, _ = PrepHelper.prep_targets(ms_items, self.pred_target_f, True, False, 1., 0., False)
        arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
        sel_bert_t = bert_expr[arange_t, offsets_t]  # [bsize, ?, Fold, D]
        sel_basic_t = None if basic_expr is None else basic_expr[arange_t, offsets_t]  # [bsize, ?, D']
        hiddens = self.adp(sel_bert_t, sel_basic_t, [])  # [bsize, ?, D"]
        # -----
        if conf.use_hlnode:
            logits = self.predictor.predict(hiddens, None, True)
        else:
            logits = self.predictor(hiddens)  # [bsize, ?, Out]
        # -----
        log_probs = BK.log_softmax(logits, -1)
        log_probs[:,:,0] -= conf.nil_penalty  # encourage more predictions
        topk_log_probs, topk_log_labels = log_probs.topk(conf.dec_topk_label, dim=-1, largest=True, sorted=True)  # [bsize, ?, k]
        # decoding
        item_creator = self.item_creator
        head_offsets_arr = BK.get_value(offsets_t)  # [bs, ?]
        masks_arr = BK.get_value(masks_t)
        topk_log_probs_arr, topk_log_labels_arr = BK.get_value(topk_log_probs), BK.get_value(topk_log_labels)  # [bsize, ?, k]
        for one_ms_item, one_offsets_arr, one_masks_arr, one_logprobs_arr, one_labels_arr in \
                zip(ms_items, head_offsets_arr, masks_arr, topk_log_probs_arr, topk_log_labels_arr):
            center_idx = one_ms_item.center_idx
            one_sent = one_ms_item.sents[center_idx]
            minus_offset = one_ms_item.offsets[center_idx] - 1  # again consider the ROOT
            # for each sent
            sid = one_sent.sid
            partial_id0 = f"{one_sent.doc.doc_id}-s{sid}-i"
            cur_added_items = []
            num_added_item = 0
            for cur_offset, cur_valid, cur_logprobs, cur_labels in zip(one_offsets_arr, one_masks_arr, one_logprobs_arr, one_labels_arr):
                if not cur_valid:
                    continue
                this_mention = Mention(HardSpan(sid, int(cur_offset)-minus_offset, None, None))
                # decide the types
                cur_logprob_thresh = cur_logprobs[0] + self.dec_extra_logprob_thresh
                for one_logprob, one_label in zip(cur_logprobs, cur_labels):
                    if one_label == 0 or one_logprob<cur_logprob_thresh:
                        break  # NIL or under best thresh
                    # finally adding
                    this_hlidx = self.lidx2hlidx[one_label]
                    cur_added_items.append(item_creator(partial_id0 + str(num_added_item), this_mention, this_hlidx, float(one_logprob)))
                    num_added_item += 1
            # filter type constraints
            if constrain_types is not None:  # filter here
                cur_added_items = [z for z in cur_added_items if z.type in constrain_types]
            # keep the highest scored ones
            cur_max_predictions = max(1, int(conf.pred_sent_ratio * (one_sent.length-1)))
            cur_added_items.sort(key=lambda x: x.score, reverse=True)
            cur_added_items = cur_added_items[:cur_max_predictions]
            self.pred_set_f(one_sent, cur_added_items)

    def lookup(self, ms_items, constrain_types=None, *args, **kwargs):
        # simply copy the gold to pred ones
        # todo(note): may repeat because of multi-sent, but does not matter
        for ms_item in ms_items:
            for s in ms_item.sents:
                to_add_items = deepcopy(self.gold_target_f(s))
                if constrain_types is not None:
                    to_add_items = [z for z in to_add_items if z.type in constrain_types]
                self.pred_set_f(s, to_add_items)

# =====

# score mentions for saliency or affinity to certain properties
class MentionAffiScorer:
    def __init__(self):
        pass

    def loss(self):
        pass

    def predict(self):
        pass

# =====
# multi-sent argument linker

class PairScorerConf(Conf):
    def __init__(self):
        self.use_arc_score = True
        self.ff_hid_size = 0
        self.ff_hid_layer = 0
        self.use_biaffine = True
        self.use_ff = True
        self.use_ff2 = False
        self.biaffine_div = 0.
        self.biaffine_init_ortho = True
        self.nil_score0 = False
        # extra layer?
        self.arc_space = 0
        self.lab_space = 0
        self.transform_act = "elu"
        # freeze?
        self.biaffine_freeze = False

class PairScorer(BasicNode):
    def __init__(self, pc, ps_conf: PairScorerConf, input_dim: int, output_size: int):
        super().__init__(pc, None, None)
        #
        self.use_arc_score = ps_conf.use_arc_score
        self.output_size = output_size
        self.nil_mask = None
        self.nil_score0 = ps_conf.nil_score0
        #
        transform_act = ps_conf.transform_act
        arc_space = input_dim if ps_conf.arc_space<=0 else ps_conf.arc_space
        lab_space = input_dim if ps_conf.lab_space<=0 else ps_conf.lab_space
        #
        rop_getter = FreezeRop if ps_conf.biaffine_freeze else (lambda: None)
        #
        if self.use_arc_score:
            self.arc_m = self.add_sub_node("am", Affine(pc, input_dim, arc_space, act=transform_act)) if ps_conf.arc_space>0 else None
            self.arc_h = self.add_sub_node("ah", Affine(pc, input_dim, arc_space, act=transform_act)) if ps_conf.arc_space>0 else None
            self.arc_scorer = self.add_sub_node(
                "as", BiAffineScorer(pc, arc_space, arc_space, 1, ps_conf.ff_hid_size, ff_hid_layer=ps_conf.ff_hid_layer,
                                     use_biaffine=ps_conf.use_biaffine, use_ff=ps_conf.use_ff, use_ff2=ps_conf.use_ff2,
                                     biaffine_div=ps_conf.biaffine_div, biaffine_init_ortho=ps_conf.biaffine_init_ortho,
                                     init_rop=rop_getter()))
        else:
            self.arc_m = self.arc_h = self.arc_scorer = None
        self.lab_m = self.add_sub_node("lm", Affine(pc, input_dim, lab_space, act=transform_act)) if ps_conf.lab_space>0 else None
        self.lab_h = self.add_sub_node("lh", Affine(pc, input_dim, lab_space, act=transform_act)) if ps_conf.lab_space>0 else None
        self.lab_scorer = self.add_sub_node(
            "ls", BiAffineScorer(pc, lab_space, lab_space, output_size, ps_conf.ff_hid_size, ff_hid_layer=ps_conf.ff_hid_layer,
                                 use_biaffine=ps_conf.use_biaffine, use_ff=ps_conf.use_ff, use_ff2=ps_conf.use_ff2,
                                 biaffine_div=ps_conf.biaffine_div, biaffine_init_ortho=ps_conf.biaffine_init_ortho,
                                 init_rop=rop_getter()))

    def refresh(self, rop=None):
        super().refresh(rop)
        self.nil_mask = BK.input_real([0.] + [1.] * (self.output_size-1))  # used for later masking

    def __call__(self, repr_ef, repr_evt):
        lm_expr = repr_ef if self.lab_m is None else self.lab_m(repr_ef)
        lh_expr = repr_evt if self.lab_h is None else self.lab_h(repr_evt)
        lab_full_score = self.lab_scorer.paired_score(lm_expr, lh_expr)
        if self.use_arc_score:
            am_expr = repr_ef if self.arc_m is None else self.arc_m(repr_ef)
            ah_expr = repr_evt if self.arc_h is None else self.arc_h(repr_evt)
            arc_full_score = self.arc_scorer.paired_score(am_expr, ah_expr)
            full_score = lab_full_score + arc_full_score.expand([-1, -1, -1, self.output_size]) * self.nil_mask
        else:
            full_score = lab_full_score
        # if zero nil dim
        if self.nil_score0:
            full_score *= self.nil_mask
        return full_score  # [bs, ef, evt, Out]

#
class ArgLinkerConf(Conf):
    def __init__(self):
        # todo(note): this value should be enough
        self._num_ef_label = 200
        self._num_evt_label = 200
        self._num_sdist = 50
        # embedding dimensions
        self.dim_label = 50
        self.dim_sdist = 50
        self.use_evt_label = True  # whether using evt label
        self.use_ef_label = True  # whether using ef label
        self.use_sdist = False  # whether using sdist features for scoring
        # adp
        self.hidden_dim = 512
        self.share_adp = True
        # pairwise scorer
        self.ps_conf = PairScorerConf()
        self.share_scorer = True
        # -----
        # training
        self.train_drop_evt_lab = 0.33
        self.train_drop_ef_lab = 0.33
        self.center_train_neg_rate = 0.5
        self.outside_train_neg_rate = 0.
        self.lambda_center = 1.
        self.lambda_outside = 1.
        self.use_micro_loss = False
        # -----
        # decoding
        # drop evt lab at test time? -> mainly for special decoding purpose
        self.test_drop_evt_lab = 0.
        self.test_drop_ef_lab = 0.
        # constraints
        self.max_sdist = 0  # (c1) max sentence distance for arg link
        self.max_pairwise_role = 1  # (c2) max number of role between one pair of ef and evt
        self.max_pairwise_ef = 1  # (c2') max number of ef between one pair of role and evt
        self.rdec_extra_prob_thresh = 0.5  # similar to the event one, make sure the extra ones are above relative thresh
        self.nil_penalty_center = 0.  # decrease logprob for NIL deliberately
        self.nil_penalty_outside = 0.  # decrease logprob for NIL deliberately
        self.use_cons_frame = True  # whether using constraints for frames
        self.use_nearby_priority = True  # nearby cands have priority
        self.which_frame_cons = "ere"  # ere/rams/...
        # -----
        # which mode: normalize on label or arg-cand
        self.norm_mode = "label"  # label or acand
        # for event label embedding lookup, use hlnode?
        self.use_borrowed_evt_hlnode = False
        self.use_borrowed_evt_adp = False

    def do_validate(self):
        if self.norm_mode == "acand":
            assert self.ps_conf.nil_score0, "Only support NIL-score=0 in this mode!"
            assert self.center_train_neg_rate == self.outside_train_neg_rate, "Only one neg rate in this mode!"

# link pairs of ef and evt
class ArgLinker(BasicNode):
    def __init__(self, pc, conf: ArgLinkerConf, vocab: HLabelVocab, input_enc_dims,
                 borrowed_evt_hlnode: HLabelNode, borrowed_evt_adp: TaskSpecAdp):
        super().__init__(pc, None, None)
        self.conf = conf
        self.vocab = vocab
        assert vocab.nil_as_zero
        VOCAB_LAYER = -1  # todo(note): simply use final largest layer
        self.lidx2hlidx = vocab.layered_hlidx[VOCAB_LAYER]  # int-idx -> HLabelIdx
        self.output_size = len(self.lidx2hlidx)
        # -----
        # embeddings
        self.emb_ef = self.add_sub_node("eef", Embedding(pc, conf._num_ef_label, conf.dim_label, fix_row0=False))
        if conf.use_borrowed_evt_hlnode:
            self.emb_evt = borrowed_evt_hlnode  # not belonging here!
        else:
            self.emb_evt = self.add_sub_node("eevt", Embedding(pc, conf._num_evt_label, conf.dim_label, fix_row0=False))
        self.emb_sdist = self.add_sub_node("esd", Embedding(pc, conf._num_sdist, conf.dim_sdist, fix_row0=False))
        self.emb_sdist_offset = conf._num_sdist//2
        # adp
        adp_extra_inputs_evt = [conf.dim_label] if conf.use_evt_label else []
        adp_extra_inputs_ef = [conf.dim_label] if conf.use_ef_label else []
        if conf.use_sdist:
            adp_extra_inputs_ef.append(conf.dim_sdist)
        if conf.use_borrowed_evt_adp:
            self.evt_adp = borrowed_evt_adp  # not belonging here!
        else:
            self.evt_adp = self.add_sub_node("adp0", TaskSpecAdp(pc, input_enc_dims, adp_extra_inputs_evt, conf.hidden_dim))
        self.ef_center_adp = self.add_sub_node(
            "adp1", TaskSpecAdp(pc, input_enc_dims, adp_extra_inputs_ef, conf.hidden_dim))
        if conf.share_adp:
            self.ef_outside_adp = self.ef_center_adp
        else:
            self.ef_outside_adp = self.add_sub_node(
                "adp2", TaskSpecAdp(pc, input_enc_dims, adp_extra_inputs_ef, conf.hidden_dim))
        # scorer
        self.center_scorer = self.add_sub_node("s0", PairScorer(pc, conf.ps_conf, conf.hidden_dim, self.output_size))
        if conf.share_scorer:
            self.outside_scorer = self.center_scorer
        else:
            self.outside_scorer = self.add_sub_node("s1", PairScorer(pc, conf.ps_conf, conf.hidden_dim, self.output_size))
        # decoding
        self.pred_evt_f = lambda x: x.pred_events
        self.pred_ef_f = lambda x: x.pred_entity_fillers
        self.dec_extra_logprob_thresh = float(np.log(conf.rdec_extra_prob_thresh))  # already neg number
        #
        self.arg_budgets = {"ere": ERE_ARG_BUDGETS, "rams": RAMS_ARG_BUDGETS, "": None}[conf.which_frame_cons]
        # =====
        self.loss = {"label": self._loss_mode_label, "acand": self._loss_mode_acand}[conf.norm_mode]
        self.arg_cand_f = {"label": self._arg_cand_label, "acand": self._arg_cand_acand}[conf.norm_mode]

    # prepare instance
    def prepare_inst(self, inst: DocInstance):
        for sent in inst.sents:
            ms = sent.preps["ms"]
            ms.arg_pack = PrepHelper.prep_train_args_one(ms)

    # event label lookup
    def evt_label_lookup(self, evt_types_t):
        conf = self.conf
        if conf.use_borrowed_evt_hlnode:
            # lookup_idxes = np.asarray([self.lidx2hlidx[z] for z in BK.get_value(evt_types_t.view(-1))])
            # return self.emb_evt.lookup(lookup_idxes.reshape(BK.get_shape(evt_types_t)))
            return self.emb_evt.lookup(evt_types_t)
        else:
            return self.emb_evt(evt_types_t)

    # dropout idxes
    def _dropout_idxes(self, idxes, rate):
        if rate>0.:
            zero_mask = (BK.rand(BK.get_shape(idxes)) < rate).long()
            return zero_mask * idxes
        else:
            return idxes

    # get loss for the norm_label mode
    def _loss_mode_label(self, ms_items: List, bert_expr, basic_expr, dynamic_prepare):
        conf = self.conf
        bsize = len(ms_items)
        # build targets: [bs, evt, ?], [bs, ef?, ?], [bs, ef, evt]
        evt_pack, center_ef_pack, outside_ef_pack = PrepHelper.prep_train_args(ms_items, dynamic_prepare)
        evt_offsets_t, evt_masks_t, evt_types_t = evt_pack
        # -----
        # return 0 if all no targets
        if BK.get_shape(evt_offsets_t, -1) == 0:
            zzz = BK.zeros([])
            all_losses = [[zzz, zzz, zzz], [zzz, zzz, zzz]]
        # -----
        else:
            arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
            evt_bert_t = bert_expr[arange_t, evt_offsets_t]  # [bsize, evt, Fold, D]
            evt_basic_t = None if basic_expr is None else basic_expr[arange_t, evt_offsets_t]  # [bsize, evt, D']
            evt_extra_inputs = [self.evt_label_lookup(self._dropout_idxes(evt_types_t, conf.train_drop_evt_lab))] \
                if conf.use_evt_label else []  # [bsize, evt, ?]
            evt_hiddens = self.evt_adp(evt_bert_t, evt_basic_t, evt_extra_inputs)  # [bsize, evt, D"]
            # for center and outside
            all_losses = []
            for one_ef_pack, one_ef_adp, one_scorer, one_train_neg_rate in \
                    zip([center_ef_pack, outside_ef_pack], [self.ef_center_adp, self.ef_outside_adp],
                        [self.center_scorer, self.outside_scorer], [conf.center_train_neg_rate, conf.outside_train_neg_rate]):
                ef_offsets_t, ef_masks_t, ef_types_t, ef_sdists_t, roles_t = one_ef_pack
                # -----
                # make it 0 if no targets (no efs or no corresponding pairs)
                if np.prod(BK.get_shape(roles_t)) == 0:
                    all_losses.append([BK.zeros([]), BK.zeros([]), BK.zeros([])])
                    continue
                # -----
                ef_bert_t = bert_expr[arange_t, ef_offsets_t]  # [bsize, ef, Fold, D]
                ef_basic_t = None if basic_expr is None else basic_expr[arange_t, ef_offsets_t]  # [bsize, ef, D']
                # get type and sdist embeddings
                ef_extra_inputs = [self.emb_ef(self._dropout_idxes(ef_types_t, conf.train_drop_ef_lab))] \
                    if conf.use_ef_label else []  # [bsize, ef, ?]
                if conf.use_sdist:
                    ef_extra_inputs.append(self.emb_sdist(ef_sdists_t+self.emb_sdist_offset))
                ef_hiddens = one_ef_adp(ef_bert_t, ef_basic_t, ef_extra_inputs)  # [bsize, ef, D"]
                # pairwise scores
                full_score = one_scorer(ef_hiddens, evt_hiddens)  # [bs, ef, evt, Out]
                full_logprobs = BK.log_softmax(full_score, -1)
                gold_logprobs = full_logprobs.gather(-1, roles_t.unsqueeze(-1)).squeeze(-1)  # [*, len-ef, len-evt]
                # sampling and mask
                loss_mask = ef_masks_t.unsqueeze(-1) * evt_masks_t.unsqueeze(-2)
                # ====
                # first select examples (randomly)
                sel_mask = (BK.rand(BK.get_shape(loss_mask)) < one_train_neg_rate).float()  # [*, len-ef, len-evt]
                # add gold and exclude pad
                gold_mask = (roles_t > 0).float()
                sel_mask += gold_mask
                sel_mask.clamp_(max=1.)
                loss_mask *= sel_mask
                # =====
                loss_sum = - (gold_logprobs * loss_mask).sum()
                loss_count = loss_mask.sum()
                all_losses.append([loss_sum, loss_count, gold_mask.sum()])
        # =====
        # combine and get final losses
        center_loss_sum, center_loss_count, center_gold_count = all_losses[0]
        outside_loss_sum, outside_loss_count, outside_gold_count = all_losses[1]
        center_loss_sum *= conf.lambda_center
        outside_loss_sum *= conf.lambda_outside
        if conf.use_micro_loss:
            # todo(note): this mode seems strange if have lambdas!=1?
            final_ret = [[center_loss_sum+outside_loss_sum, center_loss_count+outside_loss_count,
                          center_gold_count+outside_gold_count]]
        else:
            final_ret = [[center_loss_sum, center_loss_count, center_gold_count],
                         [outside_loss_sum, outside_loss_count, outside_gold_count]]
        return final_ret

    # todo(+3): many parts are repeated with _loss_mode_label
    # get loss for the norm_carg mode
    def _loss_mode_acand(self, ms_items: List, bert_expr, basic_expr, dynamic_prepare):
        conf = self.conf
        bsize = len(ms_items)
        # build targets: [bs, evt, ?], [bs, ef?, ?], [bs, ef, evt]
        evt_pack, center_ef_pack, outside_ef_pack = PrepHelper.prep_train_args(ms_items, dynamic_prepare)
        evt_offsets_t, evt_masks_t, evt_types_t = evt_pack
        # -----
        # return 0 if all no targets
        if BK.get_shape(evt_offsets_t, -1) == 0:
            zzz = BK.zeros([])
            return [[zzz, zzz, zzz]]
        else:
            arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
            evt_bert_t = bert_expr[arange_t, evt_offsets_t]  # [bsize, evt, Fold, D]
            evt_basic_t = None if basic_expr is None else basic_expr[arange_t, evt_offsets_t]  # [bsize, evt, D']
            evt_extra_inputs = [self.evt_label_lookup(self._dropout_idxes(evt_types_t, conf.train_drop_evt_lab))] \
                if conf.use_evt_label else []  # [bsize, evt, ?]
            evt_hiddens = self.evt_adp(evt_bert_t, evt_basic_t, evt_extra_inputs)  # [bsize, evt, D"]
            # for center and outside
            all_scores, all_ef_masks, all_roles = [], [], []
            for one_ef_pack, one_ef_adp, one_scorer, one_train_neg_rate in \
                    zip([center_ef_pack, outside_ef_pack], [self.ef_center_adp, self.ef_outside_adp],
                        [self.center_scorer, self.outside_scorer], [conf.center_train_neg_rate, conf.outside_train_neg_rate]):
                ef_offsets_t, ef_masks_t, ef_types_t, ef_sdists_t, roles_t = one_ef_pack
                all_ef_masks.append(ef_masks_t)
                all_roles.append(roles_t)
                # -----
                # make it 0 if no targets (no efs or no corresponding pairs)
                if np.prod(BK.get_shape(roles_t)) == 0:
                    all_scores.append(BK.zeros(BK.get_shape(roles_t) + [self.output_size]))
                    continue
                # -----
                ef_bert_t = bert_expr[arange_t, ef_offsets_t]  # [bsize, ef, Fold, D]
                ef_basic_t = None if basic_expr is None else basic_expr[arange_t, ef_offsets_t]  # [bsize, ef, D']
                # get type and sdist embeddings
                ef_extra_inputs = [self.emb_ef(self._dropout_idxes(ef_types_t, conf.train_drop_ef_lab))] \
                    if conf.use_ef_label else []  # [bsize, ef, ?]
                if conf.use_sdist:
                    ef_extra_inputs.append(self.emb_sdist(ef_sdists_t+self.emb_sdist_offset))
                ef_hiddens = one_ef_adp(ef_bert_t, ef_basic_t, ef_extra_inputs)  # [bsize, ef, D"]
                # pairwise scores
                full_score = one_scorer(ef_hiddens, evt_hiddens)  # [bs, ef, evt, Out]
                all_scores.append(full_score)
            # =====
            # concatenate and get overall scores: [bs, ef0+ef1, evt, Lab]
            concat_scores_t, concat_ef_masks_t, concat_roles_t = BK.concat(all_scores,1), BK.concat(all_ef_masks,1), BK.concat(all_roles,1)
            # mask out invalid efs and evts
            valid_mask = concat_ef_masks_t.unsqueeze(-1) * evt_masks_t.unsqueeze(-2)  # [bs, ef0+ef1, evt]
            concat_scores_t += (1.-valid_mask).unsqueeze(-1) * Constants.REAL_PRAC_MIN  # [bs, ef0+ef1, evt, Lab]
            extra_scores_size = BK.get_shape(concat_scores_t)
            extra_scores_size[1] = 1
            concat_scores_tplus = BK.concat([concat_scores_t, BK.zeros(extra_scores_size)], 1)  # [bs, ef0+ef1+1, evt]
            # softmax at the ef dim
            full_logprobs = BK.log_softmax(concat_scores_tplus, 1)[:,:,:,1:]  # [bs, ef0+ef1+1, evt, Lab-1]
            # hit mask
            hit_mask = BK.zeros(BK.get_shape(concat_scores_t))  # [bs, ef0+ef1, evt, Lab]
            hit_mask.scatter_(-1, concat_roles_t.unsqueeze(-1), 1.)  # [bs, ef0+ef1, evt, Lab]
            extra_mask = (hit_mask.sum(1, keepdim=True)<=0.).float()  # [bs, 1, evt, Lab], extra one indicating no hit
            extra_mask *= (BK.rand(BK.get_shape(extra_mask)) < conf.center_train_neg_rate).float()  # sample on neg examples
            hit_mask_plus = BK.concat([hit_mask, extra_mask], 1)[:,:,:,1:]  # [bs, ef0+ef1+1, evt, Lab-1]
            hit_mask_plus *= evt_masks_t.unsqueeze(-2).unsqueeze(-1)
            # final loss
            gold_logprobs = (full_logprobs*hit_mask_plus).sum(1)  # [bs, evt, lab-1]
            # todo(note): no need for neg rate, and no lambdas(center/outside)!!
            loss_sum = - gold_logprobs.sum()
            loss_count = hit_mask_plus.sum()
            loss_countg = hit_mask[:,:,:,1:].sum()
            return [[loss_sum, loss_count, loss_countg]]

    # =====

    def predict(self, ms_items: List, bert_expr, basic_expr):
        conf = self.conf
        bsize = len(ms_items)
        # build targets: [bs, evt, ?], [bs, ef?, ?], [bs, ef, evt]
        evt_pack = PrepHelper.prep_targets(ms_items, self.pred_evt_f, True, False, 0., 0., True)
        evt_offsets_t, evt_masks_t, _, evt_items_arr, evt_labels_t = evt_pack  # [bs, evt, ?]
        # -----
        # simply return if no events
        if evt_items_arr.size == 0:
            return
        # -----
        arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
        evt_bert_t = bert_expr[arange_t, evt_offsets_t]  # [bsize, evt, Fold, D]
        evt_basic_t = None if basic_expr is None else basic_expr[arange_t, evt_offsets_t]  # [bsize, evt, D']
        evt_extra_inputs = [self.evt_label_lookup(self._dropout_idxes(evt_labels_t, conf.test_drop_evt_lab))] \
            if conf.use_evt_label else []  # [bsize, evt, ?]
        evt_hiddens = self.evt_adp(evt_bert_t, evt_basic_t, evt_extra_inputs)  # [bsize, evt, D"]
        # for center and outside
        center_ef_pack = PrepHelper.prep_targets(ms_items, self.pred_ef_f, True, False, 0., 0., True)
        outside_ef_pack = PrepHelper.prep_targets(ms_items, self.pred_ef_f, False, True, 0., 0., True)
        all_scores = []
        for one_ef_pack, one_ef_adp, one_scorer in \
                zip([center_ef_pack, outside_ef_pack], [self.ef_center_adp, self.ef_outside_adp],
                    [self.center_scorer, self.outside_scorer]):
            ef_offsets_t, ef_masks_t, ef_sdists_t, ef_items_arr, ef_labels_t = one_ef_pack
            # skip if no ef
            if BK.get_shape(ef_offsets_t, -1) == 0:
                all_scores.append(np.zeros([bsize, 0, BK.get_shape(evt_offsets_t, -1), self.output_size], dtype=np.float32))
                continue
            # -----
            ef_bert_t = bert_expr[arange_t, ef_offsets_t]  # [bsize, ef, Fold, D]
            ef_basic_t = None if basic_expr is None else basic_expr[arange_t, ef_offsets_t]  # [bsize, ef, D']
            # get type and sdist embeddings
            ef_extra_inputs = [self.emb_ef(self._dropout_idxes(ef_labels_t, conf.test_drop_ef_lab))] \
                if conf.use_ef_label else []  # [bsize, ef, ?]
            if conf.use_sdist:
                ef_extra_inputs.append(self.emb_sdist(ef_sdists_t+self.emb_sdist_offset))
            ef_hiddens = one_ef_adp(ef_bert_t, ef_basic_t, ef_extra_inputs)  # [bsize, ef, D"]
            # pairwise scores
            full_score = one_scorer(ef_hiddens, evt_hiddens)  # [bs, ef, evt, Out]
            # mask out invalid efs and evts
            valid_mask = ef_masks_t.unsqueeze(-1) * evt_masks_t.unsqueeze(-2)  # [bs, ef, evt]
            full_score += (1.-valid_mask).unsqueeze(-1) * Constants.REAL_PRAC_MIN  # [bs, ef, evt, Lab]
            # todo(note): not softmax here!
            # full_logprobs = BK.log_softmax(full_score, -1)
            # all_scores.append(BK.get_value(full_logprobs))  # [bs, ef, evt, Out]
            all_scores.append(BK.get_value(full_score))  # [bs, ef, evt, Out]
        # =====
        # the main decoding
        center_ef_items_arr, outside_ef_items_arr = center_ef_pack[-2], outside_ef_pack[-2]
        center_scores_arr, outside_scores_arr = all_scores
        # loop on bs
        for pack in zip(evt_items_arr, center_ef_items_arr, outside_ef_items_arr, center_scores_arr, outside_scores_arr):
            self.arg_decode(*pack)

    # decode for one ms
    def arg_decode(self, evt_arr, center_ef_arr, outside_ef_arr, center_sarr, outside_sarr):
        conf = self.conf
        cons_max_pairwise_role = conf.max_pairwise_role
        cons_max_pairwise_ef = conf.max_pairwise_ef
        use_cons_frame = conf.use_cons_frame
        cand_rank_f = (lambda x: (-abs(x[-3]), x[-1])) if conf.use_nearby_priority else (lambda x: x[-1])
        # for each frame
        for one_evt_idx, one_evt in enumerate(evt_arr):
            if one_evt is None:
                continue  # skip padding ones
            evt_sid = one_evt.mention.hard_span.sid
            one_center_sarr, one_outside_sarr = center_sarr[:, one_evt_idx], outside_sarr[:, one_evt_idx]  # [ef, Lab]
            one_evt.links.clear()  # clean old ones if there are any
            # collect all cands
            cur_cands = self.arg_cand_f(evt_sid, center_ef_arr, outside_ef_arr, one_center_sarr, one_outside_sarr)
            # decode for the links
            one_frame_budget = self.arg_budgets[one_evt.type] if use_cons_frame else None
            one_frame_role_counts = {}
            one_frame_ef_counts = {}  # id(ef) -> count
            # then sort (by sdist and score) and fill in
            cur_cands.sort(key=cand_rank_f, reverse=True)
            for this_ef, _, this_ridx, this_score in cur_cands:
                this_hlidx = self.lidx2hlidx[this_ridx]
                this_role_str = str(this_hlidx)
                # -----
                # constraint 2: pairwise
                this_evt_role_count = one_frame_role_counts.get(this_role_str, 0)
                this_evt_ef_count = one_frame_ef_counts.get(id(this_ef), 0)
                # max ef for each role and max role for each ef
                if this_evt_role_count>=cons_max_pairwise_ef or this_evt_ef_count>=cons_max_pairwise_role:
                    continue
                # -----
                # constraint 3: evt frame
                if use_cons_frame and this_evt_role_count >= one_frame_budget.get(this_role_str, 0):
                    continue  # no event frame budget
                # =====
                one_evt.add_arg(this_ef, role=this_role_str, role_idx=this_hlidx, score=this_score)
                one_frame_role_counts[this_role_str] = this_evt_role_count + 1
                one_frame_ef_counts[id(this_ef)] = this_evt_ef_count + 1

    # =====
    # get cands for one event
    # [ef], [ef], [ef, Lab], [ef, Lab]

    # normalize at LAB dim
    def _arg_cand_label(self, evt_sid, center_ef_arr, outside_ef_arr, one_center_sarr, one_outside_sarr):
        cur_cands = []
        conf = self.conf
        cons_max_sdist = conf.max_sdist
        cons_max_pairwise_role = conf.max_pairwise_role
        for cur_ef_arr, cur_sarr, cur_nil_penalty in zip([center_ef_arr, outside_ef_arr], [one_center_sarr, one_outside_sarr],
                                                         [conf.nil_penalty_center, conf.nil_penalty_outside]):
            for this_ef, this_sarr in zip(cur_ef_arr, cur_sarr):
                if this_ef is None:
                    continue  # skip padding ones
                ef_sid = this_ef.mention.hard_span.sid
                cur_sdist = ef_sid - evt_sid
                # constraint 1: sentence distance
                if abs(cur_sdist) > cons_max_sdist:
                    continue
                # constraint 2: max number of roles between one pair & >NIL[idx=0]
                # this_sarr = np.log(MathHelper.softmax(this_sarr))
                this_sarr = np.log(MathHelper.softmax(this_sarr)+1e-10)
                this_sarr[0] -= cur_nil_penalty  # encourage more links
                cur_logprob_thresh = this_sarr.max().item() + self.dec_extra_logprob_thresh
                for one_role_idx in reversed(this_sarr.argsort(-1)[-cons_max_pairwise_role:]):  # score high to low
                    if one_role_idx == 0:  # todo(note): 0 means NIL
                        break
                    this_score = this_sarr[one_role_idx].item()
                    if this_score < cur_logprob_thresh:
                        break  # fail threshold
                    # add one candidate: (ef-item, sdist, role_idx, logprob)
                    cur_cands.append((this_ef, cur_sdist, one_role_idx, this_score))
        return cur_cands

    # normalize at EF(arg-cand) dim
    def _arg_cand_acand(self, evt_sid, center_ef_arr, outside_ef_arr, one_center_sarr, one_outside_sarr):
        cur_cands = []
        conf = self.conf
        cons_max_sdist = conf.max_sdist
        cons_max_pairwise_ef = conf.max_pairwise_ef
        # todo(note): instead we add to the scores
        one_center_sarr += conf.nil_penalty_center
        one_outside_sarr += conf.nil_penalty_outside
        # concat
        concat_ef_arr = np.concatenate([center_ef_arr, outside_ef_arr, np.asarray([None])], 0)  # [ef0+ef1+1]
        concat_sarr = np.concatenate([one_center_sarr, one_outside_sarr, np.zeros([1,self.output_size])], 0)  # [ef0+ef1+1, Lab]
        # concat_logsoftmax = np.log(MathHelper.softmax(concat_sarr, 0))
        concat_logsoftmax = np.log(MathHelper.softmax(concat_sarr, 0)+1e-10)
        # constraint 1: sentence distance (note that invalids are already masked out previously and final None should be kept)
        concat_sdist_valid = np.asarray([float(abs(this_ef.mention.hard_span.sid-evt_sid)<=cons_max_sdist)
                                         if this_ef is not None else 1. for this_ef in concat_ef_arr])
        concat_logsoftmax += (1.-concat_sdist_valid[:, np.newaxis]) * Constants.REAL_PRAC_MIN
        # loop on lab: ignore 0!!
        for one_role_idx in range(1, self.output_size):
            this_sarr = concat_logsoftmax[:, one_role_idx]
            # constraint 2: max number of cands for one role
            cur_logprob_thresh = this_sarr.max().item() + self.dec_extra_logprob_thresh
            for ef_idx in reversed(this_sarr.argsort(-1)[-cons_max_pairwise_ef:]):
                this_ef = concat_ef_arr[ef_idx]
                if this_ef is None:  # hit final one or non-valid
                    break
                this_score = this_sarr[ef_idx].item()
                if this_score < cur_logprob_thresh:
                    break  # fail threshold
                cur_sdist = this_ef.mention.hard_span.sid - evt_sid
                # again constraint 1: sentence distance
                if abs(cur_sdist) > cons_max_sdist:
                    continue
                # add one candidate: (ef-item, sdist, role_idx, logprob)
                cur_cands.append((this_ef, cur_sdist, one_role_idx, this_score))
        return cur_cands

# b tasks/zie/models3/model_dec:554
