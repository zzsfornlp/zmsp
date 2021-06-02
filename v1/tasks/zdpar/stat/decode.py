#

# decoding with the extra help from outside sources
# special with g1 models

import numpy as np
import pandas as pd

from msp.utils import StatRecorder, Helper, zlog, zopen
from msp.nn import BK

from ..ef.parser import G1Parser, G1ParserConf
from ..ef.parser.g1p import PruneG1Conf, ParseInstance
from ..common.confs import DepParserConf
from ..main.test import prepare_test

# todo(note): directly use the specific algo
# from ..algo.mst import mst_unproj
from ..algo.nmst import mst_unproj

from msp.zext.dpar import ParserEvaler
from ..common.data import ParseInstance, get_data_writer

from .collect import load_model, StatConf, StatVocab
from .smodel import StatModel, StatApplyConf

# speical decoding
class SDConf(DepParserConf):
    def __init__(self, parser_type, args):
        self.zprune = PruneG1Conf()
        self.smodel = ""
        self.aconf = StatApplyConf()
        # other options
        self.apply_pruning = True  # apply pruning for the sd scores with model pruning masks
        self.combine_marginals = True  # combine with raw scores or marginals?
        #
        super().__init__(parser_type, args)

def main(args):
    conf, model, vpack, test_iter = prepare_test(args, SDConf)
    # make sure the model is order 1 graph model, otherwise cannot run through
    assert isinstance(model, G1Parser) and isinstance(conf.pconf, G1ParserConf)
    # =====
    # helpers
    all_stater = StatRecorder(False)
    def _stat(k, v):
        all_stater.record_kv(k, v)
    # =====
    # explicitly doing decoding here
    if conf.smodel:
        zlog(f"Load StatModel from {conf.smodel}")
        smodel: StatModel = load_model(conf.smodel)
    else:
        zlog(f"Blank model for debug")
        dummy_vocab = StatVocab()
        dummy_vocab.sort_and_cut()
        smodel: StatModel = StatModel(StatConf([]), dummy_vocab)
    aconf = conf.aconf
    # other options
    apply_pruning = conf.apply_pruning
    combine_marginals = conf.combine_marginals
    all_insts = []
    for cur_insts in test_iter:
        # score and prune
        valid_mask, arc_score, label_score, mask_expr, marginals = model.prune_on_batch(cur_insts, conf.zprune)
        # only modifying arc score!
        valid_mask_arr = BK.get_value(valid_mask)  # [bs, slen, slen]
        arc_score_arr = BK.get_value(arc_score)  # [bs, slen, slen]
        label_score_arr = BK.get_value(label_score)  # [bs, slen, slen, L]
        marginals_arr = BK.get_value(marginals)  # [bs, slen, slen]
        # for each inst
        for one_idx, one_inst in enumerate(cur_insts):
            tokens = one_inst.words.vals
            if smodel.lc:
                tokens = [str.lower(z) for z in tokens]
            cur_len = len(tokens)
            cur_arange = np.arange(cur_len)
            # get current things: [slen, slen]
            one_valid_mask_arr = valid_mask_arr[one_idx, :cur_len, :cur_len]
            one_arc_score_arr = arc_score_arr[one_idx, :cur_len, :cur_len]
            one_label_score_arr = label_score_arr[one_idx, :cur_len, :cur_len]
            one_marginals_arr = marginals_arr[one_idx, :cur_len, :cur_len]
            # get scores from smodel
            one_sd_scores = smodel.apply_sent(tokens, aconf)  # [slen, slen]
            if apply_pruning:
                one_sd_scores *= one_valid_mask_arr  # TODO(WARN): 0 or -inf?
            orig_arc_score = (one_marginals_arr if combine_marginals else one_arc_score_arr)
            final_arc_score = one_sd_scores + orig_arc_score
            # first decoding with arc scores
            mst_heads_arr, _, _ = mst_unproj(np.expand_dims(final_arc_score, axis=0),
                                             np.array([cur_len], dtype=np.int32), labeled=False)
            mst_heads_arr = mst_heads_arr[0]  # [slen]
            # then get each one's argmax label
            argmax_label_arr = one_label_score_arr[cur_arange, mst_heads_arr].argmax(-1)  # [slen]
            # put in the results
            one_inst.pred_heads.set_vals(mst_heads_arr)  # directly int-val for heads
            one_inst.pred_labels.build_vals(argmax_label_arr, model.label_vocab)
            # extra output
            one_inst.extra_pred_misc["orig_score"] = orig_arc_score[cur_arange, mst_heads_arr].tolist()
            one_inst.extra_pred_misc["sd_score"] = one_sd_scores[cur_arange, mst_heads_arr].tolist()
            # =====
            # special analyzing with the results and the gold (only for analyzing)
            gold_heads = one_inst.heads.vals
            _stat("num_sent", 1)
            _stat("num_token", (cur_len-1))
            _stat("num_pairs", (cur_len-1)*(cur_len-1))
            _stat("num_pairs_valid", one_valid_mask_arr.sum())  # remaining ones after the pruning (pruning rate)
            _stat("num_gold_valid", one_valid_mask_arr[cur_arange, gold_heads][1:].sum())  # pruning coverage
            # about the sd scores
            _stat("num_sd_nonzero", (one_sd_scores>0.).sum())
            _stat("num_sd_correct", (one_sd_scores>0.)[cur_arange, gold_heads][1:].sum())
            # =====
        all_insts.extend(cur_insts)
    # =====
    # write and eval
    # sorting by idx of reading
    all_insts.sort(key=lambda x: x.inst_idx)
    # write
    dconf = conf.dconf
    if dconf.output_file:
        with zopen(dconf.output_file, "w") as fd:
            data_writer = get_data_writer(fd, dconf.output_format)
            data_writer.write(all_insts)
    # eval
    evaler = ParserEvaler()
    eval_arg_names = ["poses", "heads", "labels", "pred_poses", "pred_heads", "pred_labels"]
    for one_inst in all_insts:
        # todo(warn): exclude the ROOT symbol; the model should assign pred_*
        real_values = one_inst.get_real_values_select(eval_arg_names)
        evaler.eval_one(*real_values)
    report_str, res = evaler.summary()
    zlog(report_str, func="result")
    zlog("zzzzztest: testing result is " + str(res))
    # =====
    d = all_stater.summary(get_v=True, get_str=False)
    d["z_prune_rate"] = d["num_pairs_valid"] / d["num_pairs"]
    d["z_prune_coverage"] = d["num_gold_valid"] / d["num_token"]
    d["z_sd_precision"] = d["num_sd_correct"] / d["num_sd_nonzero"]
    d["z_sd_recall"] = d["num_sd_correct"] / d["num_token"]
    Helper.printd(d, "\n\n")

"""
SRC_DIR="../src/"
DATA_DIR="../data/UD_RUN/ud24/"
DATA_DIR="../data/ud24/"
CUR_LANG=en
RGPU=-1
DEVICE=-1
MODEL_DIR=./ru10k/

for combine_marginals in 0 1; do
for final_lambda in 0.1 0.25 0.5 0.75 1.; do
for feat_log in 0 1; do
for feat_alpha in 1. 0.75 0.5 0.25; do
for word_beta in 1. 0.5 0.; do
for dist_decay_v in 1. 0.9 0.5; do
for final_norm in 1 0; do
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR}:. python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat.decode ${MODEL_DIR}/_conf device:${DEVICE} dict_dir:${MODEL_DIR} model_load_name:${MODEL_DIR}/zmodel.best partype:g1 log_file: test:${DATA_DIR}/${CUR_LANG}_dev.ppos.conllu test_extra_pretrain_files:${DATA_DIR}/wiki.multi.${CUR_LANG}.filtered.vec smodel:./model.pic zprune.pruning_mthresh:0.1 combine_marginals:${combine_marginals} final_lambda:${final_lambda} feat_log:${feat_log} feat_alpha:${feat_alpha} word_beta:${word_beta} dist_decay_v:${dist_decay_v} final_norm:${final_norm}
done; done; done; done; done; done; done |& tee _log1002

# out of the loop
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR}:. python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat.decode ${MODEL_DIR}/_conf device:${DEVICE} dict_dir:${MODEL_DIR} model_load_name:${MODEL_DIR}/zmodel.best partype:g1 log_file: test:${DATA_DIR}/${CUR_LANG}_dev.ppos.conllu test_extra_pretrain_files:${DATA_DIR}/wiki.multi.${CUR_LANG}.filtered.vec smodel:./model.pic zprune.pruning_mthresh:0.1

# b tasks/zdpar/stat/decode:78, cur_len>10
"""
