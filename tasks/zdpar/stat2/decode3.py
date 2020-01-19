#

# special decoding 3: in a hierarchical way

from typing import List, Dict
import numpy as np
from collections import OrderedDict, Counter

from msp.utils import Helper, zlog
from msp.data.vocab import Vocab
from tasks.zdpar.common.data import ParseInstance
from .helper_basic import SentProcessor, main_loop, show_heatmap, SDBasicConf
from .helper_decode import get_decoder, get_fdecoder, IRConf, IterReducer

#
class SD3Conf(SDBasicConf):
    def __init__(self, args):
        super().__init__()
        #
        self.show_heatmap = False
        self.show_result = False
        self.no_diag = True
        # =====
        # reducer
        self.ir_conf = IRConf()
        # =====
        # leaf layer decoding
        self.use_fdec = False
        self.fdec_what = "reduce"  # reduce based or pos based or vocab based
        self.fdec_vocab_file = ""
        self.fdec_vr_thresh = 100  # vocab reduce thresh, <= this
        self.fdec_ir_perc = 0.5  # how many percentage put in fdec for ir mode
        self.fdec_method = "max"  # right/left/max
        self.fdec_oper = "add"  # operation between O/T
        self.fdec_Olambda = 1.  # if use score, lambda for origin one
        self.fdec_Tlambda = 1.  # if use score, lambda for transpose one
        # main layer decoding
        self.mdec_aggregate_scores = False  # aggregate score from fdec
        self.mdec_method = "m2"  # upper layer decoding method: l2r/r2l/rand/m1/m2/m3/m3s/m4(recursive-reduce)
        self.mdec_proj = True
        self.mdec_oper = "add"  # operation between O/T
        self.mdec_Olambda = 1.  # if use score, lambda for origin one
        self.mdec_Tlambda = 1.  # if use score, lambda for transpose one
        #
        self.update_from_args(args)
        self.validate()

# =====
class SentProcessor3(SentProcessor):
    def __init__(self, conf: SD3Conf):
        super().__init__()
        self.conf = conf
        self.mdecoder = get_decoder(conf.mdec_method)
        self.fdecoder = get_fdecoder(conf.fdec_method)
        #
        # self.content_pos_set = {"NOUN", "VERB", "ADJ", "PROPN"}
        self.content_pos_set = {"NOUN", "VERB", "PROPN"}
        #
        oper_dict = {"add": np.add, "max": np.maximum, "min": np.minimum}
        self.fdec_oper = oper_dict[conf.fdec_oper]
        self.mdec_oper = oper_dict[conf.mdec_oper]
        #
        self.ir_reducer = IterReducer(conf.ir_conf)
        #
        if conf.fdec_vocab_file:
            self.r_vocab = Vocab.read(conf.fdec_vocab_file)
        else:
            self.r_vocab = None

    # =====
    # todo(note): no arti-ROOT in inputs
    def test_one_sent(self, one_inst: ParseInstance):
        conf = self.conf
        which_fold = conf.which_fold
        sent = one_inst.words.vals[1:]
        uposes = one_inst.poses.vals[1:]
        dep_heads = one_inst.heads.vals[1:]
        dep_labels = [z.split(":")[0] for z in one_inst.labels.vals[1:]]  # todo(note): only first level
        #
        ret_info = {"orig": (sent, uposes, dep_heads, dep_labels)}
        slen = len(sent)
        slen_arange = np.arange(slen)
        orig_distances = one_inst.extra_features["sd2_scores"][:, :, which_fold]  # [s, s+1]
        # =====
        # step 0: pre-processing
        aug_influence_scores = np.copy(orig_distances)  # [s,s]
        if conf.no_diag:
            arange_t = slen_arange.astype(np.int32)
            aug_influence_scores[:,1:][arange_t, arange_t] = 0.
        influence_scores = aug_influence_scores[:,1:]
        # =====
        # step 0.5: separate the layers
        punct_masks = np.array([(z == "PUNCT" or z == "SYM") for z in uposes], dtype=np.bool)
        nonpunct_masks = (~punct_masks)
        content_masks = np.array([(z in self.content_pos_set) for z in uposes], dtype=np.bool)
        content_idxes = content_masks.nonzero()[0]
        fun_masks = (~content_masks)
        fun_idxes = fun_masks.nonzero()[0]
        # =====
        # step 1: leaf layer
        if conf.use_fdec:
            if conf.fdec_what == "pos":
                fdec_masks = np.copy(fun_masks)
            elif conf.fdec_what == "reduce":
                _, ir_reduce_idxes = self.ir_reducer.reduce(influence_scores, orig_distances[:, 0])
                fdec_masks = np.zeros(slen, dtype=np.bool)
                fdec_masks[ir_reduce_idxes[:int(slen*conf.fdec_ir_perc)]] = True
            elif conf.fdec_what == "vocab":
                vthr = conf.fdec_vr_thresh
                fdec_masks = np.asarray([self.r_vocab.get(z, vthr+1)<=vthr for z in sent], dtype=np.bool)
            else:
                raise NotImplementedError()
            # prepare fun scores
            original_scores = influence_scores
            fdec_scores = self.fdec_oper(conf.fdec_Olambda * original_scores, conf.fdec_Tlambda * (original_scores.T))
            fdec_scores[fdec_masks] = 0.  # no inner interactions
            # [slen], content words all have 0 as head
            fdec_preds = self.fdecoder(fdec_masks, fun_scores=fdec_scores, orig_scores=original_scores)
            group_idxes = [[i] for i in range(slen)]  # [slen] of list of children-group (include self)
            for m, h in enumerate(fdec_preds):
                if fdec_masks[m] and h>0:
                    group_idxes[h-1].append(m)
            mdec_idxes = (~fdec_masks).nonzero()[0]
        else:
            fdec_masks = np.zeros(slen, dtype=np.bool)
            fdec_preds = [0] * slen
            group_idxes = [[i] for i in range(slen)]
            mdec_idxes = list(range(slen))
        # =====
        # step 2: main layer
        root_scores = np.zeros(slen)  # TODO(!)
        if conf.mdec_aggregate_scores:
            pass  # TODO(!)
        if conf.mdec_method == "m3s":
            # special decoding starting from a middle step of m3
            pass  # TODO(!)
        else:
            # decoding the contracted sent (only mdec_idxes)
            original_scores = influence_scores
            processed_scores = self.mdec_oper(conf.mdec_Olambda * original_scores, conf.mdec_Tlambda * (original_scores.T))
            original_scores = original_scores[mdec_idxes, :][:, mdec_idxes]
            processed_scores = processed_scores[mdec_idxes, :][:, mdec_idxes]
            root_scores = root_scores[mdec_idxes]
            if len(processed_scores)>0:
                output_heads, output_root = self.mdecoder(processed_scores, original_scores, root_scores, conf.mdec_proj)
                # restore all
                assert len(output_heads) == len(mdec_idxes)
                output_root = mdec_idxes[output_root]
                real_heads = fdec_preds.copy()  # copy from fun preds
                for m,h in enumerate(output_heads):
                    if h==0:
                        real_heads[mdec_idxes[m]] = 0
                    else:
                        real_heads[mdec_idxes[m]] = mdec_idxes[h-1]+1
                output_heads = real_heads
            else:
                # only punctuations or empty sentence?
                output_root = 0
                output_heads = [0] * slen
        # ===== eval
        ret_info["output"] = (output_heads, output_root)
        # eval for UAS
        ret_info.update(self._eval(dep_heads, output_heads, nonpunct_masks, content_masks, fun_masks, fdec_masks))
        # eval for reducing
        ret_info.update(self._eval_order(dep_heads, content_masks, "POS", sent, dep_labels))
        ir_reduce_order, _ = self.ir_reducer.reduce(influence_scores, orig_distances[:, 0])
        ret_info.update(self._eval_order(dep_heads, ir_reduce_order, "IR", sent, dep_labels))
        #
        if conf.show_result:
            self.eval_order_verbose(dep_heads, ir_reduce_order, sent, dep_labels)
            self._show(one_inst, output_heads, ret_info, conf.show_heatmap, aug_influence_scores, True)
        #
        self.infos.append(ret_info)
        return ret_info

    def summary(self):
        ret = {}
        ret.update(self._sum_eval())
        ret.update(self._sum_eval_order("POS"))
        ret.update(self._sum_eval_order("IR"))
        # # tmp look
        # from .helper_basic import UD2_DEP_LABELS
        # for lab in ["<", ">"] + UD2_DEP_LABELS:
        #     zlog(f"# =====\nLooking at Label={lab}")
        #     Helper.printd(self._sum_eval_order("IR"+lab))
        return ret

#
def main(args):
    conf = SD3Conf(args)
    sp = SentProcessor3(conf)
    main_loop(conf, sp)

# PYTHONPATH=../src/ python3 -m pdb ../src/tasks/cmd.py zdpar.stat2.decode3 input_file:en_cut.ppos.conllu
# PYTHONPATH=../src/ python3 -m pdb ../src/tasks/cmd.py zdpar.stat2.decode3 input_file:_en_dev.ppos.pic already_pre_computed:1
# PYTHONPATH=../src/ python3 -m pdb ../src/tasks/cmd.py zdpar.stat2.decode3 input_file:_en_dev.ppos.pic already_pre_computed:1 show_result:1 show_heatmap:1 min_len:10 max_len:20 rand_input:1 rand_seed:100

# b tasks/zdpar/stat2/decode3:138
