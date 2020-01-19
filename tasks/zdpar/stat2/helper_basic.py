#

from typing import List, Dict
import numpy as np
import pickle
from collections import OrderedDict, Counter, defaultdict

from scipy.stats import spearmanr, pearsonr
import pandas as pd

from msp.utils import StatRecorder, Helper, zlog, zopen, Conf
from msp.nn import BK, NIConf
from msp.nn import init as nn_init
from msp.nn.modules.berter import BerterConf, Berter
from msp.data.vocab import Vocab

from tasks.zdpar.common.confs import DepParserConf
from tasks.zdpar.common.data import ParseInstance, get_data_writer
from tasks.zdpar.main.test import prepare_test, get_data_reader, init_everything
from tasks.zdpar.ef.parser import G1Parser

from .helper_feature import FeaturerConf, Featurer
from .helper_decode import get_decoder

# speical decoding
class SDBasicConf(Conf):
    def __init__(self):
        # =====
        # io
        self.input_file = ""
        self.output_file = ""
        self.output_pic = ""
        self.rand_input = False
        self.rand_seed = 12345
        # filter
        self.min_len = 0
        self.max_len = 100000
        # feature extractor
        self.already_pre_computed = False
        self.fake_scores = False  # create all 0. distance scores for debugging
        self.fconf = FeaturerConf()
        self.which_fold = -1
        # msp
        self.niconf = NIConf()  # nn-init-conf
        # todo(+N): how to deal with strange influences of low-freq words? (topic influence?)
        # replace by bert
        self.sent_repl_times = 0
        # replace unk with vocab
        self.vocab_file = ""
        self.unk_repl_token = "[UNK]"
        self.unk_repl_upos = []
        self.unk_repl_thresh = -1  # replace unk if <=this
        self.unk_repl_split_thresh = 100  # replace unk if a word is split larger than this
        # =====
        self.debug_no_gold = False
        self.processing = True  # no processing maybe means precomputing things
        # =====

# =====
# helpers

# showing with heatmap
def show_heatmap(sent, distances, col_add_cls):
    import matplotlib.pyplot as plt
    from .helper_draw import heatmap, annotate_heatmap
    fig, ax = plt.subplots()
    im, cbar = heatmap(distances, sent, (["[CLS]"]+sent) if col_add_cls else sent, ax=ax,
                       cmap="YlGn", cbarlabel="distance")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    plt.show()

#
UD2_DEP_LABELS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']

class SentProcessor:
    def __init__(self):
        self.infos = []

    def test_one_sent(self, one_inst: ParseInstance):
        raise NotImplementedError("Need to specify in sub-classes!!")

    def summary(self):
        raise NotImplementedError("Need to specify in sub-classes!!")

    # uas eval
    def _eval(self, gold_heads, pred_heads, nonpunct_masks, content_masks, fun_masks, fdec_masks):
        slen = len(gold_heads)
        ret = defaultdict(int)
        all_masks = np.ones(slen)
        for midx in range(slen):
            h_gold = gold_heads[midx] - 1
            h_pred = pred_heads[midx] - 1
            h_corr_d = int(h_gold == h_pred)
            h_corr_u = int((h_gold == h_pred) or (h_pred >= 0 and gold_heads[h_pred] == midx + 1))  # for UUAS
            #
            for one_name, one_masks in zip(["all", "np", "ct", "fun", "fdec"],
                                           [all_masks, nonpunct_masks, content_masks, fun_masks, fdec_masks]):
                if one_masks[midx]:
                    ret[one_name+"_dhit"] += h_corr_d
                    ret[one_name+"_uhit"] += h_corr_u
                    ret[one_name+"_count"] += 1
                else:
                    ret[one_name + "_dhit"] += 0
                    ret[one_name + "_uhit"] += 0
                    ret[one_name + "_count"] += 0
        ret = dict(ret)
        return ret

    # check the reduce orders of gold h/m
    def _eval_order(self, gold_heads, reduce_order, prefix_name: str, sent, dep_labels):
        ret = defaultdict(int)
        for m, h in enumerate(gold_heads):
            h -= 1
            if h<0:
                continue
            r_m, r_h = reduce_order[m], reduce_order[h]
            if prefix_name=="IR" and r_m==r_h:
                raise RuntimeError()
            corr, unk, wrong = int(r_m<r_h), int(r_m==r_h), int(r_m>r_h)
            for infix, hit in zip(["", "<", ">", dep_labels[m]], [True, r_m<len(gold_heads)//2, r_m>=len(gold_heads)//2, 1]):
                if hit:
                    cur_prefix = prefix_name + infix
                    ret[cur_prefix + "_all"] += 1
                    ret[cur_prefix + "_corr"] += corr
                    ret[cur_prefix + "_unk"] += unk
                    ret[cur_prefix + "_wrong"] += wrong
        ret = dict(ret)
        return ret

    def eval_order_verbose(self, gold_heads, reduce_order, sent, dep_labels):
        all_corr = 0
        for m in np.argsort(reduce_order):  # sort by reduce order
            h = gold_heads[m]-1
            if h<0:
                continue
            r_m, r_h = reduce_order[m], reduce_order[h]
            corr = r_m < r_h
            print(f"m={m}({sent[m]}), h={h}({sent[h]}), LAB={dep_labels[m]}, order={r_m}-{r_h}-{corr}")
            all_corr += int(corr)
        print(f"Final: {all_corr}/{len(sent)}")

    def _sum_eval(self):
        r = {}
        for cc in "du":
            for one_name in ["all", "np", "ct", "fun", "fdec"]:
                hit, count = sum(z.get(f"{one_name}_{cc}hit", 0) for z in self.infos), \
                             sum(z.get(f"{one_name}_count", 0) for z in self.infos)
                r[f"SEVAL_{cc}_{one_name}_UAS"] = hit / max(1, count)
                r[f"SEVAL_{cc}_{one_name}_Detail"] = f"{hit}/{count}={hit/max(1, count)}"
        return r

    def _sum_eval_order(self, prefix_name):
        r = {}
        for one_name in ["corr", "unk", "wrong"]:
            hit, count = sum(z.get(f"{prefix_name}_{one_name}",0) for z in self.infos), \
                         sum(z.get(f"{prefix_name}_all", 0) for z in self.infos)
            r[f"SEVAL_{prefix_name}_{one_name}_%"] = hit / max(1, count)
            r[f"SEVAL_{prefix_name}_{one_name}_Detail"] = f"{hit}/{count}={hit/max(1, count)}"
        return r

    def _show(self, one_inst, pred_heads, info, show_hmap: bool, distances, set_trace):
        sent, uposes, dep_heads = one_inst.words.vals[1:], one_inst.poses.vals[1:], one_inst.heads.vals[1:]
        slen = len(sent)
        aug_sent = ["R"] + sent
        all_tokens = [("R", "--", -1, "--", -1, "--")] + \
                     [(sent[i], uposes[i], dep_heads[i], aug_sent[dep_heads[i]],
                       pred_heads[i], aug_sent[pred_heads[i]])
                      for i in range(slen)]
        print(pd.DataFrame(all_tokens).to_string())
        print("; ".join([f"{one_name}:{info[one_name+'_dhit']}/{info[one_name+'_uhit']}/{info[one_name+'_count']}"
                         for one_name in ["all", "np", "ct", "fun"]]))
        # =====
        if show_hmap:
            print(sent)
            print(uposes)
            print(one_inst.extra_features.get("feat_seq", None))
            for one_ri, one_seq in enumerate(one_inst.extra_features.get("sd3_repls", [])):
                print(f"#R{one_ri}: {one_seq}")
            show_heatmap(sent, distances, True)
        if set_trace:
            import pdb
            pdb.set_trace()

    def _parse_heads(self, dep_heads: List[int]):
        # todo(note): a naive root-path matching method
        slen = len(dep_heads)
        up_paths = [None for _ in range(slen)]
        for m, h in enumerate(dep_heads):
            # clear the up path
            cur_m, cur_h = m, h
            cur_stack = []
            while cur_m >= 0 and up_paths[cur_m] is None:
                cur_stack.append(cur_m)
                cur_m = cur_h - 1
                cur_h = dep_heads[cur_h - 1]
            # assign them
            upper_path = up_paths[cur_m] if cur_m >= 0 else []  # the upper path that is known
            for one_i2, one_m in enumerate(cur_stack):
                assert up_paths[one_m] is None
                up_paths[one_m] = cur_stack[one_i2:] + upper_path
        # once we know the paths, we know the depth
        depths = [len(z) for z in up_paths]  # starting from 1
        # then assign pairwise distance by finding common prefix
        # todo(note): not efficient since there are much repeated computations
        syntax_distances = np.zeros([slen, slen])
        for i1 in range(slen):
            for i2 in range(i1 + 1, slen):
                common_count = sum(a == b for a, b in zip(reversed(up_paths[i1]), reversed(up_paths[i2])))
                one_dist = depths[i1] + depths[i2] - 2 * common_count
                syntax_distances[i1, i2] = one_dist
                syntax_distances[i2, i1] = one_dist
        return syntax_distances, up_paths, depths

# =====
# data reader
def yield_data(filename):
    if filename is None or filename == "":
        zlog("Start to read raw sentence from stdin")
        while True:
            line = input(">> ")
            if len(line) == 0:
                break
            sent = line.split()
            cur_len = len(sent)
            one = ParseInstance(sent, ["_"] * cur_len, [0] * cur_len, ["_"] * cur_len)
            yield one
    # todo(note): judged by special ending!!
    elif filename.endswith(".pic"):
        zlog(f"Start to read pickle from file {filename}")
        with zopen(filename, 'rb') as fd:
            while True:
                try:
                    one = pickle.load(fd)
                    yield one
                except EOFError:
                    break
    else:
        # otherwise read collnu
        zlog(f"Start to read conllu from file {filename}")
        for one in get_data_reader(filename, "conllu", "", True):
            yield one

# =====
def main_loop(conf: SDBasicConf, sp: SentProcessor):
    np.seterr(all='raise')
    nn_init(conf.niconf)
    np.random.seed(conf.rand_seed)
    #
    # will trigger error otherwise, save time of loading model
    featurer = None if conf.already_pre_computed else Featurer(conf.fconf)
    output_pic_fd = zopen(conf.output_pic, 'wb') if conf.output_pic else None
    all_insts = []
    vocab = Vocab.read(conf.vocab_file) if conf.vocab_file else None
    unk_repl_upos_set = set(conf.unk_repl_upos)
    with BK.no_grad_env():
        input_stream = yield_data(conf.input_file)
        if conf.rand_input:
            inputs = list(input_stream)
            np.random.shuffle(inputs)
            input_stream = inputs
        for one_inst in input_stream:
            # -----
            # make sure the results are the same; to check whether we mistakenly use gold in that jumble of analysis
            if conf.debug_no_gold:
                one_inst.heads.vals = [0] * len(one_inst.heads.vals)
                if len(one_inst.heads.vals) > 2:
                    one_inst.heads.vals[2] = 1  # avoid err in certain analysis
                one_inst.labels.vals = ["_"] * len(one_inst.labels.vals)
            # -----
            if len(one_inst)>=conf.min_len and len(one_inst)<=conf.max_len:
                folded_distances = one_inst.extra_features.get("sd2_scores")
                if folded_distances is None:
                    if conf.fake_scores:
                        one_inst.extra_features["sd2_scores"] = np.zeros(featurer.output_shape(len(one_inst.words.vals[1:])))
                    else:
                        # ===== replace certain words?
                        word_seq = one_inst.words.vals[1:]
                        upos_seq = one_inst.poses.vals[1:]
                        if conf.unk_repl_thresh > 0:
                            word_seq = [(conf.unk_repl_token if (u in unk_repl_upos_set and
                                                                 vocab.getval(w, 0)<=conf.unk_repl_thresh) else w)
                                        for w,u in zip(word_seq, upos_seq)]
                        if conf.unk_repl_split_thresh<10:
                            berter_toker = featurer.berter.tokenizer
                            word_seq = [conf.unk_repl_token if (u in unk_repl_upos_set and
                                                                len(berter_toker.tokenize(w))>conf.unk_repl_split_thresh) else w
                                         for w,u in zip(word_seq, upos_seq)]
                        # ===== auto repl by bert?
                        sent_repls = [word_seq]
                        sent_fixed = [np.zeros(len(word_seq)).astype(np.bool)]
                        for _ in range(conf.sent_repl_times):
                            new_sent, new_fixed = featurer.repl_sent(sent_repls[-1], sent_fixed[-1])
                            sent_repls.append(new_sent)
                            sent_fixed.append(new_fixed)  # once fixed, always fixed
                        one_inst.extra_features["sd3_repls"] = sent_repls
                        one_inst.extra_features["sd3_fixed"] = sent_fixed
                        # ===== score
                        folded_distances = featurer.get_scores(sent_repls[-1])
                        one_inst.extra_features["sd2_scores"] = folded_distances
                        one_inst.extra_features["feat_seq"] = word_seq
                if output_pic_fd is not None:
                    pickle.dump(one_inst, output_pic_fd)
                if conf.processing:
                    one_info = sp.test_one_sent(one_inst)
                    # put prediction
                    one_inst.pred_heads.set_vals([0] + list(one_info["output"][0]))
                    one_inst.pred_labels.set_vals(["_"] * len(one_inst.labels.vals))
                all_insts.append(one_inst)
    if output_pic_fd is not None:
        output_pic_fd.close()
    if conf.output_file:
        with zopen(conf.output_file, 'w') as wfd:
            data_writer = get_data_writer(wfd, "conllu")
            data_writer.write(all_insts)
    # -----
    Helper.printd(sp.summary())
