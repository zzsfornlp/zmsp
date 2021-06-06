#

# check (& fix) conll data

from typing import List
from collections import Counter
from msp2.utils import zopen, zlog, Conf, OtherHelper
from msp2.data.rw import ConllHelper

# --
class MainConf(Conf):
    def __init__(self):
        self.input = "/dev/stdin"
        self.output = "/dev/null"
        # --
        # format
        self.col_word = 3
        self.col_lemma = 6
        self.col_pred_id = 7
        self.col_arg_start = 11
        self.col_final_extras = 1  # final more extra cols?
        # special fixings
        self.do_fix_word_ar = False  # remove tags for ar
        self.do_fix_sense_zh = False  # give a dummy sense for zh
        self.do_fix_lemma_zh = False  # simply copy words to lemma for zh
        self.do_discard_nonflat = True  # discard preds if there are nonflat args

# --
AR_DEL_CHARS = {chr(z) for z in [0x64e, 0x64f, 0x650, 0x651, 0x652]}
# --

# --
def process_one(lines: List[str], conf: MainConf, stat):
    assert not (conf.do_fix_lemma_zh and conf.do_fix_sense_zh)
    # --
    fields = [line.split() for line in lines]  # [slen, ncol]
    n_col = len(fields[0])
    assert all(len(z)==n_col for z in fields)
    n_pred = n_col - conf.col_final_extras - conf.col_arg_start
    # --
    # special fix
    if conf.do_fix_sense_zh:
        for widx, lemma in enumerate([z[conf.col_lemma] for z in fields]):
            if lemma != "-" and fields[widx][conf.col_pred_id] == '-':
                fields[widx][conf.col_pred_id] = "XX"  # dummy!!
                stat["pred_fix_sense_zh"] += 1
    # --
    # check num of predicates
    col_pred_ids = [z[conf.col_pred_id] for z in fields]
    n_pred_id = 0
    pred_widxes = []
    for widx, pid in enumerate(col_pred_ids):
        if pid != "-":  # hit one!
            n_pred_id += 1
            pred_widxes.append(widx)
    assert n_pred_id == n_pred
    # --
    # special fix
    if conf.do_fix_word_ar:
        for widx in range(len(fields)):
            w0 = fields[widx][conf.col_word]
            # note: first get rid of special marks!!
            w0 = w0.split("#", 1)[0]
            if len(w0)>1 and not (w0[0]=="-" and w0[-1] == "-"):  # not special tokens like "-LRB-"
                if w0[0] == "-":
                    w0 = w0[1:]
                if w0[-1] == "-":
                    w0 = w0[:-1]
            # note: further delete vowels!!
            tmp_w0 = ''.join([c for c in w0 if c not in AR_DEL_CHARS])
            if len(tmp_w0) > 0:
                w0 = tmp_w0  # avoid deleting all!!
            # --
            fields[widx][conf.col_word] = w0
            stat["tok_fix_word_ar"] += 1
    if conf.do_fix_lemma_zh:
        for widx in pred_widxes:
            if fields[widx][conf.col_lemma] == "-":
                fields[widx][conf.col_lemma] = fields[widx][conf.col_word]  # directly copy word!
                stat["pred_fix_lemma_zh"] += 1
    if conf.do_discard_nonflat:
        discarded_cols = set()
        for pidx, widx in enumerate(pred_widxes):
            cur_args = [z[pidx+conf.col_arg_start] for z in fields]
            try:
            # if 1:
                ConllHelper.get_f_args(widx, cur_args)
            except AssertionError:
                # simply remove this pred
                discarded_cols.add(pidx+conf.col_arg_start)
                fields[widx][conf.col_pred_id] = '-'
                stat["pred_discard"] += 1
        if len(discarded_cols) > 0:  # discard cols
            fields = [[z2 for i,z2 in enumerate(z) if i not in discarded_cols] for z in fields]
    # --
    stat["sent"] += 1
    stat["tok"] += len(fields)
    stat["pred"] += n_pred
    return ["\t".join(z) for z in fields]

# --
def main(args):
    conf = MainConf()
    conf.update_from_args(args)
    stat = Counter()
    # --
    with zopen(conf.input) as fin, zopen(conf.output, 'w') as fout:
        cur_lines = []
        for line in fin:
            line = line.rstrip()
            if line.lstrip().startswith("#"): continue  # ignore comments!
            if len(line) == 0:
                if len(cur_lines) > 0:
                    lines2 = process_one(cur_lines, conf, stat)
                    fout.write("".join([z+'\n' for z in lines2]) + "\n")
                cur_lines.clear()
            else:
                cur_lines.append(line)
        if len(cur_lines) > 0:
            lines2 = process_one(cur_lines, conf, stat)
            fout.write("".join([z + '\n' for z in lines2]) + "\n")
    # --
    zlog(f"Read from {fin}, write to {fout}, stat=\n{OtherHelper.printd_str(stat)}")

# PYTHONPATH=../../../zsp2021/src/ python3 fix_conll_data.py ...
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
