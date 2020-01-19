#

# look at what structures can be found from representations

from typing import List, Dict
import numpy as np
import pickle
from collections import OrderedDict, Counter

from scipy.stats import spearmanr, pearsonr
import pandas as pd

from msp.utils import StatRecorder, Helper, zlog, zopen, Conf
from msp.nn import BK, NIConf
from msp.nn import init as nn_init
from msp.nn.modules.berter import BerterConf, Berter

from tasks.zdpar.common.confs import DepParserConf
from tasks.zdpar.common.data import ParseInstance, get_data_writer
from tasks.zdpar.main.test import prepare_test, get_data_reader, init_everything
from tasks.zdpar.ef.parser import G1Parser

from .helper_basic import SentProcessor, main_loop, show_heatmap, SDBasicConf
from .helper_feature import FeaturerConf, Featurer
from .helper_decode import get_decoder

# speical decoding2
class SD2Conf(SDBasicConf):
    def __init__(self, args):
        super().__init__()
        # =====
        # show results
        self.show_heatmap = False
        self.show_result = False
        # the processing
        self.no_diag = True  # make sure to zero out the self-influence scores
        self.no_punct = False  # ignore punctuations (PUNCT and SYM) for decoding
        # analysis 1: corr
        self.corr_min_surf_dist = 1  # analyze the ones >=this
        # analysis 2: local min
        self.local_min_thresh = 0.
        # analysis 3: directional
        self.dir_nonzero_thresh = 0.001
        # analysis 3.5: overall influence
        self.no_nearby = False  # exclude direct nearby words (since they usually have larger influence)
        # decoding
        self.dec_method = "m2"
        self.dec_rmh_fun = True  # remove the scores for function words (by POS tags)
        self.dec_proj = True
        self.dec_Olambda = 1.  # lambda for origin one
        self.dec_Tlambda = 1.  # lambda for transpose
        self.dec_Rlambda_inf = 0.  # lambda for root influence
        self.dec_Rlambda_cls = 0.  # lambda for root cls
        # =====
        #
        self.update_from_args(args)
        self.validate()

# =====
class SentProcessor2(SentProcessor):
    def __init__(self, conf: SD2Conf):
        super().__init__()
        self.conf = conf
        self.decoder = get_decoder(conf.dec_method)
        # self.content_pos_set = {"NOUN", "VERB", "ADJ", "PROPN"}
        self.content_pos_set = {"NOUN", "VERB", "PROPN"}

    # =====
    # todo(note): no arti-ROOT in inputs
    def test_one_sent(self, one_inst: ParseInstance, **kwargs):
        conf = self.conf
        which_fold = conf.which_fold
        sent = one_inst.words.vals[1:]
        uposes = one_inst.poses.vals[1:]
        dep_heads = one_inst.heads.vals[1:]
        dep_labels = one_inst.labels.vals[1:]
        #
        slen = len(sent)
        slen_arange = np.arange(slen)
        distances = one_inst.extra_features["sd2_scores"][:, :, which_fold]  # [s, s+1]
        # =====
        # step 0: pre-processing
        orig_distances = np.copy(distances)  # [s, s+1], no 0 mask for puncts or diag
        pairwise_masks = np.ones([slen, slen], dtype=np.bool)  # [s, s]
        # -----
        if conf.no_diag:
            arange_t = slen_arange.astype(np.int32)
            distances[:, 1:][arange_t, arange_t] = 0.
            pairwise_masks[arange_t, arange_t] = 0
        punct_masks = np.array([(z=="PUNCT" or z=="SYM") for z in uposes], dtype=np.bool)
        punct_idxes = punct_masks.nonzero()[0]  # starting from 0, no root
        nonpunct_masks = (~punct_masks)
        nonpunct_idxes = nonpunct_masks.nonzero()[0]
        content_masks = np.array([(z in self.content_pos_set) for z in uposes], dtype=np.bool)
        fun_masks = (~content_masks)
        if conf.no_punct:
            distances[:, 1:][:, punct_idxes] = 0.
            distances[:, 0:][punct_idxes, :] = 0.
            pairwise_masks[:, punct_idxes] = 0
            pairwise_masks[punct_idxes, :] = 0
        # =====
        # step 0.9: prepare various distances for comparing
        surface_distances = np.abs(slen_arange[:, np.newaxis] - slen_arange[np.newaxis, :])  # [s,s]
        syntax_distances, syntax_paths, syntax_depths = self._parse_heads(dep_heads)  # [s,s], [s,], [s]
        # =====
        ret_info = {"orig": (sent, uposes, dep_heads, dep_labels, syntax_distances, syntax_depths)}
        # step 1: correlation analysis
        # -----
        def _add_corrs(a, b, name):
            # ret_info[name+"_rsp"] = spearmanr(a, b)[0]
            # ret_info[name+"_rp"] = pearsonr(a, b)[0]
            # todo(note): not stable if len not enough or syn-distances(a) are all the same
            ret_info["corr_" + name] = 0.5 if (len(a)<=5 or np.std(a)==0) else spearmanr(a, b)[0]
        # -----
        corr_min_surf_dist = self.conf.corr_min_surf_dist
        all_nonzeros = pairwise_masks.nonzero()
        upper_mask = (all_nonzeros[0] < all_nonzeros[1]-corr_min_surf_dist)
        lower_mask = (all_nonzeros[0] > all_nonzeros[1]+corr_min_surf_dist)
        upper_idxes0, upper_idxes1 = [z[upper_mask] for z in all_nonzeros]
        lower_idxes0, lower_idxes1 = [z[lower_mask] for z in all_nonzeros]
        # -----
        nifs = -distances[:, 1:]  # negative influence value
        nifs_both = -(distances[:, 1:]+distances[:, 1:].T)/2.  # average for both directions
        # syntax vs. surface (upper triu)
        _add_corrs(syntax_distances[upper_idxes0, upper_idxes1], surface_distances[upper_idxes0, upper_idxes1], "syn_surf")
        #
        for target_distance, target_name in zip([syntax_distances, surface_distances], ["syn", "surf"]):
            # upper: syntax/surface vs. neg-influence
            _add_corrs(target_distance[upper_idxes0, upper_idxes1], nifs[upper_idxes0, upper_idxes1], target_name+"_nif_upper")
            # lower: syntax/surface vs. neg-influence
            _add_corrs(target_distance[lower_idxes0, lower_idxes1], nifs[lower_idxes0, lower_idxes1], target_name+"_nif_lower")
            # averaged: syntax/surface vs. neg-influence (upper triu)
            _add_corrs(target_distance[upper_idxes0, upper_idxes1], nifs_both[upper_idxes0, upper_idxes1], target_name+"_nif_avg")
        # =====
        # step 2: long range dep
        orig_scores = orig_distances[:, 1:]
        padded_orig_scores = np.pad(orig_scores, ((1,1),(1,1)), 'constant', constant_values=0.)  # min score is 0.
        # [s,s], minimum of the diff to four neighbours
        local_min_diff = np.stack([orig_scores-padded_orig_scores[1:-1, 2:], orig_scores-padded_orig_scores[1:-1, :-2],
                                   orig_scores-padded_orig_scores[2:, 1:-1], orig_scores-padded_orig_scores[:-2, 1:-1]],
                                  -1).min(-1)
        interest_points_masks = ((local_min_diff>conf.local_min_thresh) & pairwise_masks)  # excluding non-valids
        interest_points_masks = (interest_points_masks & interest_points_masks.T)  # bidirectional mask
        interest_points_idx0, interest_points_idx1 = interest_points_masks.nonzero()
        interest_points_idx_mask = (interest_points_idx0 < interest_points_idx1)  # no direction, make idx0<idx1
        interest_points_idx0, interest_points_idx1 = \
            interest_points_idx0[interest_points_idx_mask], interest_points_idx1[interest_points_idx_mask]
        interest_points_dist_syn = syntax_distances[interest_points_idx0, interest_points_idx1]
        interest_points_dist_surf = surface_distances[interest_points_idx0, interest_points_idx1]
        # if conf.show_heatmap:
        #     show_heatmap(sent, interest_points_masks.astype(np.float32), False)
        # syntax path?
        interest_points_syn_paths = []
        for a, b in zip(interest_points_idx0, interest_points_idx1):
            path_a, path_b = syntax_paths[a], syntax_paths[b]
            common_count = sum(a==b for a,b in zip(reversed(path_a), reversed(path_b)))
            path_tup = tuple([dep_labels[z] for z in path_a[:-common_count]] + [""] + [dep_labels[z] for z in path_b[:-common_count]])
            interest_points_syn_paths.append(path_tup)
        ret_info["ip_dsyn"] = interest_points_dist_syn
        ret_info["ip_dsurf"] = interest_points_dist_surf
        ret_info["ip_spath"] = interest_points_syn_paths
        ret_info["ip_maxdsyn"] = max(syntax_depths)-1  # max possible value of syntax distance
        # =====
        # step 3: direction for dep pairs
        influence_scores = distances[:, 1:]  # [s, s]
        mod_idxes = np.arange(slen)
        head_idxes = np.array(dep_heads)-1  # already minus 1 here!
        # exclude root and punct
        mod_idxes = mod_idxes[(head_idxes>=0) & (~punct_masks)]
        head_idxes = head_idxes[(head_idxes>=0) & (~punct_masks)]
        # (without-head, mod) - (without-mod, head)
        hm_score_diff1 = influence_scores[head_idxes, mod_idxes] - influence_scores[mod_idxes, head_idxes]
        # =====
        # step 3.5: direction by influencing others & depth
        overall_influence_mask = np.copy(pairwise_masks)  # no self, no punct
        # delete nearby?
        if conf.no_nearby:
            tmp_range_idxes = np.arange(slen)
            overall_influence_mask[tmp_range_idxes[1:], tmp_range_idxes[1:]-1] = 0.
            overall_influence_mask[tmp_range_idxes[:-1], tmp_range_idxes[:-1]+1] = 0.
        average_influence_score = (influence_scores * overall_influence_mask).sum(-1) / (overall_influence_mask.sum(-1) + 1e-5)
        hm_score_diff2 = average_influence_score[head_idxes] - average_influence_score[mod_idxes]
        # correlation of depth and influence
        _add_corrs(np.array(syntax_depths)[mod_idxes], average_influence_score[mod_idxes], "inf_depth")
        # -----
        for prefix, hm_score_diff in zip(["dir_", "dir2_"], [hm_score_diff1, hm_score_diff2]):
            ret_info[prefix+"num"] = len(hm_score_diff)
            ret_info[prefix+"diff_arr"] = hm_score_diff
            # ret_info[prefix+"depth"] = np.array(syntax_depths)[mod_idxes]
            # _add_corrs(ret_info["dir_depth"], ret_info[prefix+"diff_arr"], prefix+"diff_depth")
            dir_mask_posi, dir_mask_neg = (hm_score_diff>conf.dir_nonzero_thresh), (hm_score_diff<-conf.dir_nonzero_thresh)
            ret_info[prefix+"positive"] = dir_mask_posi.sum().item()
            ret_info[prefix+"negative"] = dir_mask_neg.sum().item()
            ret_info[prefix+"nearzero"] = (~(dir_mask_posi | dir_mask_neg)).sum().item()
            ret_info[prefix+"avg_pos"] = hm_score_diff[dir_mask_posi]
            ret_info[prefix+"avg_neg"] = hm_score_diff[dir_mask_neg]
        # step 3.6: best influence and root?
        ret_info["depth_best_inf"] = syntax_depths[np.argmax(average_influence_score).item()]
        ret_info["depth_best_cls"] = syntax_depths[np.argmax(distances[:,0]).item()]
        # -----
        # step 4: inference
        # score pre-processing
        original_scores = influence_scores
        processed_scores = conf.dec_Olambda * original_scores + conf.dec_Tlambda * (original_scores.T)
        root_scores = conf.dec_Rlambda_cls * distances[:,0] + conf.dec_Rlambda_inf * average_influence_score
        # delete fun word as h
        if conf.dec_rmh_fun:
            fun_mask = np.array([z not in self.content_pos_set for z in uposes]).astype(np.bool)
            processed_scores[fun_mask] = 0.
            root_scores[fun_mask] = 0.
        # exclude punct for decoding?
        if conf.no_punct:
            original_scores = original_scores[nonpunct_idxes, :][:, nonpunct_idxes]
            processed_scores = processed_scores[nonpunct_idxes, :][:, nonpunct_idxes]
            root_scores = root_scores[nonpunct_idxes]
        if len(processed_scores)>0:
            # decode
            output_heads, output_root = self.decoder(processed_scores, original_scores, root_scores, conf.dec_proj)
            # sink fun?
            # restore punct?
            if conf.no_punct:
                assert len(output_heads) == len(nonpunct_idxes)
                output_root = nonpunct_idxes[output_root]
                real_heads = [output_root+1] * slen  # put puncts to the real root
                # todo(note): be careful about the root offset (only in heads)
                for m,h in enumerate(output_heads):
                    if h==0:
                        real_heads[nonpunct_idxes[m]] = 0
                    else:
                        real_heads[nonpunct_idxes[m]] = nonpunct_idxes[h-1]+1
                output_heads = real_heads
        else:
            # only punctuations or empty sentence?
            output_root = 0
            output_heads = [0] * slen
        ret_info["output"] = (output_heads, output_root)
        # ===== eval
        ret_info.update(self._eval(dep_heads, output_heads, nonpunct_masks, content_masks, fun_masks))
        if conf.show_result:
            self._show(one_inst, output_heads, ret_info, conf.show_heatmap, distances, True)
        #
        self.infos.append(ret_info)
        return ret_info

    def summary(self):
        # get averaged(stddev) scores
        ret = OrderedDict()
        ret["#Sent"] = len(self.infos)
        ret["#Token"] = sum(len(z["orig"][0]) for z in self.infos)
        count_np = 0
        for z in self.infos:
            count_np += sum(p not in ["PUNCT", "SYM"] for p in z["orig"][1])
        ret["#TokenNP"] = count_np
        # step 1: correlation analysis
        for n in ["syn_surf"] + [a+b for a in ["syn", "surf"] for b in ["_nif_upper", "_nif_lower", "_nif_avg"]]:
            values = [z["corr_" + n] for z in self.infos]
            ret["S1_" + n] = (np.mean(values), np.std(values))
        # step 2: long range dep
        # skip this one since not much patterns here
        # step 3: direction for dep pairs
        corr_values = [z["corr_" + "inf_depth"] for z in self.infos]
        ret["S3_corr_inf_depth"] = (np.mean(corr_values), np.std(corr_values))
        for prefix in ["dir_", "dir2_"]:
            for name in ["positive", "negative", "nearzero"]:
                values = [z[prefix+name] / z[prefix+"num"] for z in self.infos if z[prefix+"num"]>0]
                ret["S3_" + prefix + name] = (np.mean(values), np.std(values))
            for name in ["avg_pos", "avg_neg"]:
                values = np.concatenate([z[prefix+name] for z in self.infos])
                ret["S3_" + prefix + name] = (np.mean(values), np.std(values))
        # best inf vs. depth
        ret["S3_depth_best_inf"] = Counter([z["depth_best_inf"] for z in self.infos])
        ret["S3_depth_best_cls"] = Counter([z["depth_best_cls"] for z in self.infos])
        # =====
        ret.update(self._sum_eval())
        return ret

#
def main(args):
    conf = SD2Conf(args)
    sp = SentProcessor2(conf)
    main_loop(conf, sp)

#
"""
SRC_DIR="../src/"
DATA_DIR="../data/UD_RUN/ud24/"
#DATA_DIR="../data/ud24/"
CUR_LANG=en
RGPU=2
DEVICE=0
MODEL_DIR=./ru10k/
# bert
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat2.decode2 device:${DEVICE} input_file:${DATA_DIR}/${CUR_LANG}_dev.ppos.conllu output_pic:_tmp.pic
## input_file:en_cut.ppos.conllu
# bert with precomputed pic
PYTHONPATH=../src/ python3 ../src/tasks/cmd.py zdpar.stat2.decode2 input_file:_en_dev.ppos.pic already_pre_computed:1 
# another example
PYTHONPATH=../src/ python3 -m pdb ../src/tasks/cmd.py zdpar.stat2.decode2 input_file:_en_dev.ppos.pic already_pre_computed:1 dec_Rlambda_inf:0 dec_Rlambda_cls:0 dec_method:m2 dec_Tlambda:0 dec_proj:1 output_file:_tmp.conllu2
# model
#CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat2.decode2 ${MODEL_DIR}/_conf device:${DEVICE} dict_dir:${MODEL_DIR} model_load_name:${MODEL_DIR}/zmodel.best partype:g1 log_file: test:${DATA_DIR}/${CUR_LANG}_dev.ppos.conllu test_extra_pretrain_files:${DATA_DIR}/wiki.multi.${CUR_LANG}.filtered.vec
#
b tasks/zdpar/stat2/decode2:348
"""
