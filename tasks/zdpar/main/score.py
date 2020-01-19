#

# like testing, but mainly for exporting the scores

import msp
from msp import utils
from msp.utils import Helper, zlog, zwarn
from msp.data import MultiHelper, WordVectors, VocabBuilder
from msp.nn import BK

from .test import prepare_test

#
def main(args):
    conf, model, vpack, test_iter = prepare_test(args)
    dconf = conf.dconf
    # todo(note): here is the main change
    # make sure the model is order 1 graph model, otherwise cannot run through
    all_results = []
    all_insts = []
    with utils.Timer(tag="Run-score", info="", print_date=True):
        for cur_insts in test_iter:
            all_insts.extend(cur_insts)
            batched_arc_scores, batched_label_scores = model.score_on_batch(cur_insts)
            batched_arc_scores, batched_label_scores = BK.get_value(batched_arc_scores), BK.get_value(batched_label_scores)
            for cur_idx in range(len(cur_insts)):
                cur_len = len(cur_insts[cur_idx])+1
                # discarding paddings
                cur_res = (batched_arc_scores[cur_idx, :cur_len, :cur_len], batched_label_scores[cur_idx, :cur_len, :cur_len])
                all_results.append(cur_res)
    # reorder to the original order
    orig_indexes = [z.inst_idx for z in all_insts]
    orig_results = [None] * len(orig_indexes)
    for new_idx, orig_idx in enumerate(orig_indexes):
        assert orig_results[orig_idx] is None
        orig_results[orig_idx] = all_results[new_idx]
    # saving
    with utils.Timer(tag="Run-write", info=f"Writing to {dconf.output_file}", print_date=True):
        import pickle
        with utils.zopen(dconf.output_file, "wb") as fd:
            for one in orig_results:
                pickle.dump(one, fd)
    utils.printing("The end.")
