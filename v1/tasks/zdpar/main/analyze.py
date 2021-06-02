#

# especially analyzing on graph-like models: g1p/s2p (with-model analysis)
# like decoding, but mainly for some analysis: like pruning, marginals, ...

import numpy as np
import pandas as pd

from msp.utils import StatRecorder, Helper, zlog

from ..ef.parser import G1Parser, G1ParserConf
from ..ef.parser.g1p import PruneG1Conf, ParseInstance
from ..common.confs import DepParserConf
from .test import prepare_test

class AnalyzeConf(DepParserConf):
    def __init__(self, parser_type, args):
        # extra ones
        self.zprune = PruneG1Conf()
        #
        super().__init__(parser_type, args)

class ZObject:
    pass

def main(args):
    conf, model, vpack, test_iter = prepare_test(args, AnalyzeConf)
    # make sure the model is order 1 graph model, otherwise cannot run through
    assert isinstance(model, G1Parser) and isinstance(conf.pconf, G1ParserConf)
    # =====
    # helpers
    all_stater = StatRecorder(False)
    def _stat(k, v):
        all_stater.record_kv(k, v)
    # check agreement
    def _agree2(a, b, name):
        agreement = (np.asarray(a) == np.asarray(b))
        num_agree = int(agreement.sum())
        _stat(name, num_agree)
    # do not care about efficiency here!
    step2_pack = []
    for cur_insts in test_iter:
        # score and prune
        valid_mask, arc_score, label_score, mask_expr, marginals = model.prune_on_batch(cur_insts, conf.zprune)
        # greedy on raw scores
        greedy_label_scores, greedy_label_mat_idxes = label_score.max(-1)  # [*, m, h]
        greedy_all_scores, greedy_arc_idxes = (arc_score+greedy_label_scores).max(-1)  # [*, m]
        greedy_label_idxes = greedy_label_mat_idxes.gather(-1, greedy_arc_idxes.unsqueeze(-1)).squeeze(-1)  # [*, m]
        # greedy on marginals (arc only)
        greedy_marg_arc_scores, greedy_marg_arc_idxes = marginals.max(-1)  # [*, m]
        entropy_marg = - (marginals * (marginals + 1e-10 * (marginals==0.).float()).log()).sum(-1)  # [*, m]
        # decode
        model.inference_on_batch(cur_insts)
        # =====
        z = ZObject()
        keys = list(locals().keys())
        for k in keys:
            v = locals()[k]
            try:
                setattr(z, k, v.cpu().detach().numpy())
            except:
                pass
        # =====
        for idx in range(len(cur_insts)):
            one_inst: ParseInstance = cur_insts[idx]
            one_len = len(one_inst) + 1  # [1, len)
            _stat("all_edges", one_len-1)
            arc_gold = one_inst.heads.vals[1:]
            arc_mst = one_inst.pred_heads.vals[1:]
            arc_gma = z.greedy_marg_arc_idxes[idx][1:one_len]
            # step 1: decoding agreement, how many edges agree: gold, mst-decode, greedy-marginal
            arcs = {"gold": arc_gold, "mst": arc_mst, "gma": arc_gma}
            cmp_keys = sorted(arcs.keys())
            for i in range(len(cmp_keys)):
                for j in range(i+1, len(cmp_keys)):
                    n1, n2 = cmp_keys[i], cmp_keys[j]
                    _agree2(arcs[n1], arcs[n2], f"{n1}_{n2}")
            # step 2: confidence
            arc_agree = (np.asarray(arc_gold) == np.asarray(arc_mst))
            arc_marginals_mst = z.marginals[idx][range(1, one_len), arc_mst]
            arc_marginals_gold = z.marginals[idx][range(1, one_len), arc_gold]
            arc_entropy = z.entropy_marg[idx][1:one_len]
            for tidx in range(one_len-1):
                step2_pack.append([int(arc_agree[tidx]), min(1., float(arc_marginals_mst[tidx])),
                                   min(1., float(arc_marginals_gold[tidx])), float(arc_entropy[tidx])])
    # step 2: bucket by marginals
    if True:
        NUM_BUCKET = 10
        df = pd.DataFrame(step2_pack, columns=['agree', 'm_mst', 'm_gold', 'entropy'])
        z = df.sort_values(by='m_mst', ascending=False)
        z.to_csv('res.csv')
        for cur_b in range(NUM_BUCKET):
            interval = 1. / NUM_BUCKET
            r0, r1 = cur_b*interval, (cur_b+1)*interval
            cur_v = df[(df.m_mst>=r0) & ((df.m_mst<r1))]
            zlog(f"#===== [{r0}, {r1}): {cur_v.shape}\n" + str(cur_v.describe()))
    # =====
    d = all_stater.summary(get_v=False, get_str=True)
    Helper.printd(d, "\n\n")

"""
SRC_DIR="../src/"
DATA_DIR="../data/UD_RUN/ud23/"
CUR_LANG=en
RGPU=-1
DEVICE=-1
MODEL_DIR=../zen_g1/
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 -m pdb ${SRC_DIR}/tasks/cmd.py zdpar.main.analyze ${MODEL_DIR}/_conf device:${DEVICE} dict_dir:${MODEL_DIR} test:${DATA_DIR}/${CUR_LANG}_dev.conllu model_load_name:${MODEL_DIR}/zmodel.best aux_score_test: partype:g1 zprune.pruning_mthresh:0.02
# b tasks/zdpar/main/analyze:30
"""
