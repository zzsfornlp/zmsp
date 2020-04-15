#

# argument-augmented decoder

import numpy as np
from typing import List
from collections import Counter

from msp.utils import Conf, Constants, ZObject, JsonRW, PickleRW, zlog
from msp.nn import BK

from ..common.data import DocInstance
from ..common.data_helper import KBP17_TYPES, ERE_ARG_BUDGETS
from .model import MySimpleIEModel

#
class ArgAugConf(Conf):
    def __init__(self):
        # whether use the (if-there-are) provided ef and evt
        self.lookup_ef = False
        self.lookup_evt = False
        self.constrain_evt_types = ""  # specific for dataset?
        self.score_mini_batch = 128  # mini-batch size for all-pair scoring
        # arg decoding (constraints)
        self.max_sdist = 100  # (c1) max sentence distance for arg link
        self.max_pairwise_role = 2  # (c2) max number of role between one pair of ef and evt
        self.nil_penalty = 0.  # decrease logprob for NIL deliberately
        self.cand_sort_method = "sdist"  # cand sorting method: score, sdist, decay
        self.csm_sdist_poff = 0.5  # positive sdist (ef after evt) offset
        self.csm_decay_rate = 0.  # decay rate for the decay mode
        self.delete_repeat_propn = True  # whether delete repeated PROPN
        # arg constraints
        self.use_cons_frame = True  # whether using constraints for frames
        self.use_cons_arg = False  # whether using constraints for arg
        self.cc_load_file = ""
        self.cc_min_count = 1  # cc arg-evt-compatible constraint
        self.cc_budget_perc = 0.98  # bu role-budget constraint, >1 means no constraints
        # debug
        self.dec_debug_mode = False  # debug mode, keep the original args if lookup
        # =====
        # cross-sent arg constraints
        self.carg_role_constraints = ""  # only allow certain types to cross sent?

# the larger, the better
def get_cand_sort_keyer(conf: ArgAugConf):
    m = conf.cand_sort_method
    sdist_poff = conf.csm_sdist_poff
    sdist_f = lambda sdist: (sdist+sdist_poff if sdist>0 else -sdist)  # extra penalties for positive sdist
    if m == "score":
        return lambda x: x.score  # purely based on score
    elif m == "sdist":
        return lambda x: (-sdist_f(x.sdist), x.score)  # first key on sdist, then on score
    elif m == "decay":
        csm_decay_rate = conf.csm_decay_rate
        return lambda x: x.score - csm_decay_rate * sdist_f(x.sdist)  # actually prob*exp(-rate*|sdist|)
    else:
        raise NotImplementedError(f"UNK sort method {m}")

# todo(note): for the basic parts, actually just repeating the things in Model's "inference_on_batch"
class ArgAugDecoder:
    def __init__(self, conf: ArgAugConf, model: MySimpleIEModel):
        self.conf = conf
        self.model = model
        self.test_constrain_evt_types = {"": None, "kbp17": KBP17_TYPES}[conf.constrain_evt_types]
        # cand sorting
        self.cand_sort_keyer = get_cand_sort_keyer(conf)
        # load the constraints
        if conf.cc_load_file:
            self.argcc_checker: ArgCChecker = ArgCChecker.load(conf.cc_load_file)
            self.argcc_bu_budgets = self.argcc_checker.get_budgets(conf.cc_budget_perc)
            if conf.cc_budget_perc>1:
                zlog("No constraints for arg budgets")
                self.argcc_bu_budgets = {k: Constants.INT_PRAC_MAX for k in self.argcc_bu_budgets}
        else:
            zlog("No file to load for cc")
            self.argcc_checker = self.argcc_bu_budgets = None
        self.carg_role_set = {"": None, "P": {"Place", "Origin", "Destination"}}[conf.carg_role_constraints]

    def decode(self, inst: DocInstance):
        conf, model = self.conf, self.model
        model.refresh_batch(False)
        test_constrain_evt_types = self.test_constrain_evt_types
        with BK.no_grad_env():
            # =====
            # init the collections
            flattened_ef_ereprs, flattened_evt_ereprs = [], []
            sent_offsets = [Constants.INT_PRAC_MIN]*len(inst.sents)  # start offset of sent in the flattened erepr
            cur_offset = 0  # current offset
            all_ef_items, all_evt_items = [], []
            # =====
            # first basic run and ef and evt
            all_packs = model.bter.run([inst], training=False)
            for one_pack in all_packs:
                sent_insts, lexi_repr, enc_repr_ef, enc_repr_evt, mask_arr = one_pack
                mask_expr = BK.input_real(mask_arr)
                # =====
                # store the enc reprs and sent offsets
                sent_size, sent_len = BK.get_shape(enc_repr_ef)[:2]
                assert BK.get_shape(enc_repr_ef) == BK.get_shape(enc_repr_evt)
                flattened_ef_ereprs.append(enc_repr_ef.view(sent_size*sent_len, -1))  # [cur_flatten_size, D]
                flattened_evt_ereprs.append(enc_repr_evt.view(sent_size*sent_len, -1))
                for one_sent in sent_insts:
                    sent_offsets[one_sent.sid] = cur_offset
                    cur_offset += sent_len
                # =====
                lkrc = not conf.dec_debug_mode  # lookup.ret_copy?
                # =====
                # ef
                if conf.lookup_ef:
                    ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, ef_lab_embeds = \
                        model._lookup_mentions(sent_insts, lexi_repr, enc_repr_ef, mask_expr, model.ef_extractor, ret_copy=lkrc)
                else:
                    ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, ef_lab_embeds = \
                        model._inference_mentions(sent_insts, lexi_repr, enc_repr_ef, mask_expr, model.ef_extractor, model.ef_creator)
                # collect all valid ones
                all_ef_items.extend(ef_items[BK.get_value(ef_valid_mask).astype(np.bool)])
                # event
                if conf.lookup_evt:
                    evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, evt_lab_embeds = \
                        model._lookup_mentions(sent_insts, lexi_repr, enc_repr_evt, mask_expr, model.evt_extractor, ret_copy=lkrc)
                else:
                    evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, evt_lab_embeds = \
                        model._inference_mentions(sent_insts, lexi_repr, enc_repr_evt, mask_expr, model.evt_extractor, model.evt_creator)
                # collect all valid ones
                if test_constrain_evt_types is None:
                    all_evt_items.extend(evt_items[BK.get_value(evt_valid_mask).astype(np.bool)])
                else:
                    all_evt_items.extend([z for z in evt_items[BK.get_value(evt_valid_mask).astype(np.bool)]
                                          if z.type in test_constrain_evt_types])
            # ====
            # cross-sentence pairwise arg score
            # flattened all enc: [Offset, D]
            flattened_ef_enc_repr, flattened_evt_enc_repr = BK.concat(flattened_ef_ereprs, 0), BK.concat(flattened_evt_ereprs, 0)
            # sort by position in doc
            all_ef_items.sort(key=lambda x: x.mention.hard_span.position(True))
            all_evt_items.sort(key=lambda x: x.mention.hard_span.position(True))
            if not conf.dec_debug_mode:
                # todo(note): delete origin links!
                for z in all_ef_items:
                    if z is not None:
                        z.links.clear()
                for z in all_evt_items:
                    if z is not None:
                        z.links.clear()
            # get other info
            # todo(note): currently all using head word
            all_ef_offsets = BK.input_idx([sent_offsets[x.mention.hard_span.sid]+x.mention.hard_span.head_wid for x in all_ef_items])
            all_evt_offsets = BK.input_idx([sent_offsets[x.mention.hard_span.sid]+x.mention.hard_span.head_wid for x in all_evt_items])
            all_ef_lab_idxes = BK.input_idx([model.ef_extractor.hlidx2idx(x.type_idx) for x in all_ef_items])
            all_evt_lab_idxes = BK.input_idx([model.evt_extractor.hlidx2idx(x.type_idx) for x in all_evt_items])
            # score all the pairs (with mini-batch)
            mini_batch_size = conf.score_mini_batch
            arg_linker = model.arg_linker
            all_logprobs = BK.zeros([len(all_ef_items), len(all_evt_items), arg_linker.num_label])
            for bidx_ef in range(0, len(all_ef_items), mini_batch_size):
                cur_ef_enc_repr = flattened_ef_enc_repr[all_ef_offsets[bidx_ef:bidx_ef+mini_batch_size]].unsqueeze(0)
                cur_ef_lab_idxes = all_ef_lab_idxes[bidx_ef:bidx_ef+mini_batch_size].unsqueeze(0)
                for bidx_evt in range(0, len(all_evt_items), mini_batch_size):
                    cur_evt_enc_repr = flattened_evt_enc_repr[all_evt_offsets[bidx_evt:bidx_evt+mini_batch_size]].unsqueeze(0)
                    cur_evt_lab_idxes = all_evt_lab_idxes[bidx_evt:bidx_evt + mini_batch_size].unsqueeze(0)
                    all_logprobs[bidx_ef:bidx_ef+mini_batch_size,bidx_evt:bidx_evt+mini_batch_size] = \
                        arg_linker.predict(cur_ef_enc_repr, cur_evt_enc_repr, cur_ef_lab_idxes, cur_evt_lab_idxes,
                                           ret_full_logprobs=True).squeeze(0)
            all_logprobs_arr = BK.get_value(all_logprobs)
        # =====
        # then decode them all using the scores
        self.arg_decode(inst, all_ef_items, all_evt_items, all_logprobs_arr)
        # =====
        # assign and return
        num_pred_arg = 0
        for one_sent in inst.sents:
            one_sent.pred_entity_fillers.clear()
            one_sent.pred_events.clear()
        for z in all_ef_items:
            inst.sents[z.mention.hard_span.sid].pred_entity_fillers.append(z)
        for z in all_evt_items:
            inst.sents[z.mention.hard_span.sid].pred_events.append(z)
            num_pred_arg += len(z.links)
        info = {"doc": 1, "sent": len(inst.sents), "token": sum(s.length-1 for s in inst.sents),
                "p_ef": len(all_ef_items), "p_evt": len(all_evt_items), "p_arg": num_pred_arg}
        return info

    # inplace modification
    def arg_decode(self, inst: DocInstance, all_ef_items: List, all_evt_items: List, all_logprobs_arr):
        conf = self.conf
        cons_max_sdist = conf.max_sdist
        cons_max_pairwise_role = conf.max_pairwise_role
        # step 1: get all candidates
        all_logprobs_arr[:,:,0] -= conf.nil_penalty  # encourage more links
        all_logprobs_sortidx = all_logprobs_arr.argsort(axis=-1)
        all_cands = []
        for ef_idx, ef_item in enumerate(all_ef_items):
            ef_sid = ef_item.mention.hard_span.sid
            for evt_idx, evt_item in enumerate(all_evt_items):
                evt_sid = evt_item.mention.hard_span.sid
                cur_sdist = ef_sid - evt_sid
                # constraint 1: sentence distance
                if abs(cur_sdist) > cons_max_sdist:
                    continue
                # constraint 2: max number of roles between one pair & >NIL[idx=0]
                cur_logprobs = all_logprobs_arr[ef_idx, evt_idx]
                cur_sortidx = all_logprobs_sortidx[ef_idx, evt_idx]
                for one_role_idx in reversed(cur_sortidx[-cons_max_pairwise_role:]):  # score high to low
                    if one_role_idx == 0:  # todo(note): 0 means NIL
                        break
                    # add one candidate
                    cur_orig_score = cur_logprobs[one_role_idx].item()
                    all_cands.append(ZObject(ef=ef_item, evt=evt_item, sdist=cur_sdist,
                                             role_idx=one_role_idx, score=cur_orig_score))
        # step 2: get sorting key and sort the candidates
        all_cands.sort(key=self.cand_sort_keyer, reverse=True)
        # =====
        # debug: seeing them all
        if conf.dec_debug_mode:
            self.lookat_results(inst, all_ef_items, all_evt_items, all_cands, all_logprobs_arr)
        # =====
        # step 3: follow the sorted list and do greedy selection
        use_cons_frame = conf.use_cons_frame
        use_cons_arg = conf.use_cons_arg
        arg_linker = self.model.arg_linker
        argcc_checker = self.argcc_checker
        argcc_bu_budgets = self.argcc_bu_budgets
        carg_role_set = self.carg_role_set  # allowable cross-sent arg roles
        # frame_budgets = self.argcc_checker.frame_budget
        frame_budgets = ERE_ARG_BUDGETS
        cc_min_count = conf.cc_min_count
        added_evt_role_counts = {id(z):{} for z in all_evt_items}  # id -> {role: count}
        added_ef_role_counts = {id(z):{} for z in all_ef_items}  # id -> {role: count}
        added_ef_role_evt_counts = {id(z):{} for z in all_ef_items}  # id -> {(role, evt): count}
        for one_cand in all_cands:
            this_hlidx = arg_linker.idx2hlidx(one_cand.role_idx)
            this_role_str = str(this_hlidx)
            cur_evt, cur_ef = one_cand.evt, one_cand.ef
            this_evt_type = cur_evt.type
            cur_evt_role_counts, cur_ef_role_counts, cur_ef_role_evt_counts = \
                added_evt_role_counts[id(cur_evt)], added_ef_role_counts[id(cur_ef)], added_ef_role_evt_counts[id(cur_ef)]
            # cross-cons: cross sent arg constraints
            if one_cand.sdist!=0 and carg_role_set is not None:
                if this_role_str not in carg_role_set:
                    continue
            # c3: evt frame
            if use_cons_frame:
                this_evt_role_count = cur_evt_role_counts.get(this_role_str, 0)
                if this_evt_role_count >= frame_budgets[this_evt_type].get(this_role_str, 0):
                    continue  # no event frame budget
            # c4: arg constraints
            if use_cons_arg:
                this_ef_role_count = cur_ef_role_counts.get(this_role_str, 0)
                if this_ef_role_count >= argcc_bu_budgets.get(this_role_str, 0):
                    continue  # no role max-count budget
                this_role_evt_key = (this_role_str, this_evt_type)
                compatible = all(argcc_checker.check_compatible(this_role_evt_key, k, min_count=cc_min_count)
                                 for k in cur_ef_role_evt_counts)
                if not compatible:
                    continue  # no non-compatible roles
            # adding if survive
            if not conf.dec_debug_mode:
                cur_evt.add_arg(cur_ef, role=this_role_str, role_idx=this_hlidx, score=one_cand.score)
            # update counts
            if use_cons_frame:
                cur_evt_role_counts[this_role_str] = this_evt_role_count + 1
            if use_cons_arg:
                cur_ef_role_counts[this_role_str] = this_ef_role_count + 1
                cur_ef_role_evt_counts[this_role_evt_key] = cur_ef_role_evt_counts.get(this_role_evt_key, 0) + 1
        # step 4: delete repeated PROPN roles (keep only the closest one)
        if conf.delete_repeat_propn:
            link_sort_key = lambda x: abs(x.ef.mention.hard_span.sid-x.evt.mention.hard_span.sid)
            for one_evt_item in all_evt_items:
                sorted_links = sorted(one_evt_item.links, key=link_sort_key)  # sort by |sdist|
                surviving_links = []
                hit_ones = set()
                for one_link in sorted_links:
                    ef_sid, ef_hwid = one_link.ef.mention.hard_span.sid, one_link.ef.mention.hard_span.head_wid
                    ef_sent = inst.sents[ef_sid]
                    if ef_sent.uposes.vals[ef_hwid] == "PROPN":
                        cur_ef_key = (ef_sent.words.vals[ef_hwid].lower(), one_link.role)
                        if cur_ef_key in hit_ones:
                            continue
                        hit_ones.add(cur_ef_key)
                    surviving_links.append(one_link)
                one_evt_item.links = surviving_links
        # step 5: collect salience ones for special roles?
        # TODO(+N)

    # for debugging: ef_items already contain gold info
    def lookat_results(self, inst, all_ef_items, all_evt_items, all_cands, logprobs_arr):
        # -----
        def _env_mention(mention, sents, fcolor):
            if mention is None:
                return "NONE"
            else:
                from msp.utils import wrap_color
                sid, wid, wlen = mention.hard_span.sid, mention.hard_span.wid, mention.hard_span.length
                words = sents[sid].words.vals
                mention_str = " ".join(words[:wid]) \
                              + " " + wrap_color(" ".join(words[wid:wid + wlen]), fcolor=fcolor) + " " \
                              + " ".join(words[wid + wlen:])
                return f"SID={sid}: {mention_str}"
        def _rank_role(target_ridx, score_arr, role_list):
            rank_role_strs = []
            cidx = 0
            for ridx in reversed(np.argsort(score_arr)):
                one_str = f"R{cidx}={role_list[ridx]}({score_arr[ridx]:.3f})"
                rank_role_strs.append(one_str)
                cidx += 1
                if ridx == target_ridx:
                    break
            return "; ".join(rank_role_strs)
        # -----
        # id to idx
        evt_id_maps = {id(z):i for i,z in enumerate(all_evt_items)}
        ef_id_maps = {id(z):i for i,z in enumerate(all_ef_items)}
        #
        sents = inst.sents
        arg_linker = self.model.arg_linker
        role_list = arg_linker.vocab.layered_k[-1]  # todo(note): only use last layer for roles
        for one_evt in all_evt_items:
            evt_idx_in_arr = evt_id_maps[id(one_evt)]
            #
            sid, hwid, _ = one_evt.mention.hard_span.position(True)
            zlog(f"Event: {one_evt.type}\n{_env_mention(one_evt.mention, sents, 'blue')}")
            zlog("#====")
            # get gold args
            for one_gold_arg in one_evt.links:
                this_ridx = arg_linker.hlidx2idx(one_gold_arg.role_idx)
                this_score_arr = logprobs_arr[ef_id_maps[id(one_gold_arg.ef)], evt_idx_in_arr]
                zlog(f"GArg: {one_gold_arg.role} aug={one_gold_arg.is_aug}\n{_rank_role(this_ridx, this_score_arr, role_list)}"
                     f"\n{_env_mention(one_gold_arg.ef.mention, sents, 'red')}")
            zlog("#====")
            # get sorted candidate args
            cand_args = [z for z in all_cands if z.evt is one_evt]
            for one_cand_arg in cand_args:
                this_hlidx = arg_linker.idx2hlidx(one_cand_arg.role_idx)
                this_score_arr = logprobs_arr[ef_id_maps[id(one_cand_arg.ef)], evt_idx_in_arr]
                zlog(f"CArg: {str(this_hlidx)}\n{_rank_role(one_cand_arg.role_idx, this_score_arr, role_list)}"
                     f"\n{_env_mention(one_cand_arg.ef.mention, sents, 'green')}")
            zlog("#====")

# =====
# argument compatible checker
class ArgCChecker:
    KEY_CC0 = ("", "")  # overall counting key for (arg-role, evt-type)
    KEY_BU0 = ""  # overall counting key for budget

    def __init__(self, argcc_counts, argbu_counts):
        # three constraints: pairwise-compatible, argtype-budget, overall-budget
        self.argcc_counts = argcc_counts
        self.argbu_counts = argbu_counts  # already sorted
        #
        # self.frame_budget = ERE_ARG_BUDGETS

    #
    def check_budgets(self):
        POINTS = [0., 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
        for k in sorted(self.argbu_counts.keys()):
            sorted_counts = self.argbu_counts[k]
            sorted_counts_size = len(sorted_counts)
            piece_counts = [(p, sorted_counts[min(sorted_counts_size-1, int(sorted_counts_size*p))]) for p in POINTS]
            zlog(f"k={k}: {piece_counts}")

    def __repr__(self):
        # cc
        num_entries = len(self.argcc_counts)
        num_pairs = sum(len(z) for z in self.argcc_counts)
        return f"ArgCC: {num_entries}/{num_pairs}"

    def save(self, file):
        PickleRW.to_file(self, file)
        zlog(f"Save to file: {file} -> {self}")

    @staticmethod
    def load(file):
        ret = PickleRW.from_file(file)
        zlog(f"Read from file: {file} -> {ret}")
        return ret

    @staticmethod
    def build(docs: List[DocInstance]):
        kcc0, kbu0 = ArgCChecker.KEY_CC0, ArgCChecker.KEY_BU0
        argcc_counts = {}  # argument pairwise compatibility
        argbu_counts = {kbu0: []}  # budget: arg-role -> [counts]
        for doc in docs:
            if doc.entity_fillers is None:
                continue  # skip no args ones
            for ef in doc.entity_fillers:
                # budget
                cur_role_counts = {}
                for arg in ef.links:
                    cur_role = arg.role
                    cur_role_counts[cur_role] = cur_role_counts.get(cur_role, 0) + 1
                if len(ef.links)>0:  # todo(note): ignore zeros since currently we only care about upper constraints
                    argbu_counts[kbu0].append(len(ef.links))
                for one_r, one_c in cur_role_counts.items():
                    if one_r not in argbu_counts:
                        argbu_counts[one_r] = [one_c]
                    else:
                        argbu_counts[one_r].append(one_c)
                # compatibility
                as_arg_keys = sorted(set([(arg.role, arg.evt.type) for arg in ef.links]))
                for arg0 in as_arg_keys:
                    item = argcc_counts.get(arg0)
                    if item is None:
                        item = {kcc0: 0}
                        argcc_counts[arg0] = item
                    item[kcc0] += 1  # ("","") as the key for total counts
                    for arg1 in as_arg_keys:
                        if arg0 != arg1:
                            item[arg1] = item.get(arg1, 0) + 1
        for k,v in argbu_counts.items():
            v.sort()
        ret = ArgCChecker(argcc_counts, argbu_counts)
        zlog(f"Build from data: {len(docs)} docs -> {ret}")
        return ret

    # =====
    def get_budgets(self, percentage: float):
        ret = {}
        for k,sorted_counts in self.argbu_counts.items():
            sorted_counts_size = len(sorted_counts)
            ret[k] = sorted_counts[min(sorted_counts_size-1, int(sorted_counts_size*percentage))]
        return ret

    def check_compatible(self, a, b, min_count=1):
        return self.argcc_counts[a].get(b, 0) >= min_count

    # greedily delete the in-compatible one until the survivors are all compatible
    def filter_compatible(self, type_list: List, scores, min_count=1):
        cur_len = len(type_list)
        if scores is None:
            scores = np.ones(cur_len)
        scores = np.asarray(scores)
        # build compatible matrix
        nc_matrix = np.zeros([cur_len, cur_len])  # 1 means not compatible
        for i in range(cur_len):
            for j in range(cur_len):
                if i==j:
                    continue
                if not self.check_compatible(type_list[i], type_list[j], min_count=min_count):
                    nc_matrix[i,j] = 1.
        # build heuristics and greedy search
        neg_hscores = - (scores - (nc_matrix * scores).sum(-1))
        survived_idxes = []
        for one_idx in neg_hscores.argsort():
            if nc_matrix[survived_idxes, one_idx].sum()==0:
                survived_idxes.append(one_idx)
        survived_mask = np.zeros(cur_len)
        survived_mask[survived_idxes] = 1
        return survived_mask

# b tasks/zie/models2/decoderA:105
