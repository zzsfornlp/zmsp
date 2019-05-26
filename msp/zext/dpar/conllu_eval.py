#

# d-parser evaluation
from msp.utils import Helper, NumHelper

class ParserEvaler:
    # only for PTB, CTB
    DEFAULT_PUNCT_POS_SET = {'.', '``', "''", ':', ',', 'PU'}

    def __init__(self, ignore_punct=True, punct_set=DEFAULT_PUNCT_POS_SET):
        self.ignore_punct = ignore_punct
        self.punct_set = punct_set
        #
        self.stat = {}
        self.confusion = {}     # (only for correct heads) gold-tag => pred-tag => num

    def add_confusion(self, tg, tp):
        if tg not in self.confusion:
            self.confusion[tg] = {}
        if tp not in self.confusion[tg]:
            self.confusion[tg][tp] = 0
        self.confusion[tg][tp] += 1

    # evaluate for one sentence
    def eval_one(self, gold_pos, gold_heads, gold_labels, pred_heads, pred_labels):
        curr_stat = {}
        # add 1 for current stat
        def _a1cs(_name):
            Helper.stat_addone(curr_stat, _name)
        #
        _a1cs("sent")
        for idx in range(len(gold_pos)):
            cur_pos = gold_pos[idx]
            cur_gold_head, cur_gold_label = gold_heads[idx], gold_labels[idx]
            cur_pred_head, cur_pred_label = pred_heads[idx], pred_labels[idx]
            #
            _a1cs("tok")
            if cur_pos not in self.punct_set:
                the_suffixes = ("", "_np")
                _a1cs("tok_np")
            else:
                the_suffixes = ("", )
            #
            for one_suffix in the_suffixes:
                if cur_gold_head == cur_pred_head:
                    _a1cs("tok_corrU" + one_suffix)
                    if cur_gold_label == cur_pred_label:
                        _a1cs("tok_corrL" + one_suffix)
                    if one_suffix == "":
                        self.add_confusion(cur_gold_label, cur_pred_label)
            # ROOT
            if cur_gold_head == 0:
                _a1cs("tok_root_all")
                if cur_pred_head == 0:
                    _a1cs("tok_root_corr")
            if cur_pred_head == 0:
                _a1cs("tok_root_pall")
        # accumulate
        if curr_stat.get("tok_np", 0) > 0:
            _a1cs("sent_np")
        for one_suffix in ("", "_np"):
            this_tok_num = curr_stat.get("tok" + one_suffix, 0)
            for which_metric in ("U", "L"):
                if curr_stat.get("tok_corr" + which_metric + one_suffix, 0) == this_tok_num:
                    _a1cs("sent_corr" + which_metric + one_suffix)
        Helper.stat_addv(self.stat, curr_stat)
        # return self.summary_one(curr_stat)

    # return (report_str, result)
    def summary_one(self, stat, verbose):
        _P = lambda x: NumHelper.truncate_float(x, 6)
        #
        final_stat = {}
        s = ""
        if verbose:
            s += "Matrix:\n"
            for tg in sorted(self.confusion.keys()):
                s += str(tg) + " => "
                thems = [(num, tp) for tp,num in self.confusion[tg].items()]
                for num, tp in sorted(thems, reverse=True):
                    s += str(tp)+"~" + str(num) + " "
                s += "\n"
        # main eval
        for one_suffix in ("", "_np"):
            sent_all = stat.get("sent"+one_suffix, 0)
            sent_corrU = stat.get("sent_corrU"+one_suffix, 0)
            sent_corrL = stat.get("sent_corrL"+one_suffix, 0)
            sent_uas = _P(sent_corrU/sent_all)
            sent_las = _P(sent_corrL/sent_all)
            tok_all = stat.get("tok"+one_suffix, 0)
            tok_corrU = stat.get("tok_corrU"+one_suffix, 0)
            tok_corrL = stat.get("tok_corrL"+one_suffix, 0)
            tok_uas = _P(tok_corrU/tok_all)
            tok_las = _P(tok_corrL/tok_all)
            tok_root_all = stat.get("tok_root_all", 0)
            tok_root_pall = stat.get("tok_root_pall", 0)
            tok_corrR = stat.get("tok_root_corr", 0)
            # todo(+1): this is only ROOT recall
            tok_root_recall = _P(tok_corrR/tok_root_all)
            tok_root_precision = _P(tok_corrR/tok_root_pall if tok_root_pall>0. else 0.)
            #
            final_stat[one_suffix] = {"sent_uas": sent_uas, "sent_las": sent_las, "tok_uas": tok_uas, "tok_las": tok_las,
                                      "tok_root_rec": tok_root_recall, "tok_root_pre": tok_root_precision}
            s += "== Eval of " + one_suffix + ": "
            s += "tok: %d/%d(%.5f)/%d(%.5f) || " % (tok_all, tok_corrU, tok_uas, tok_corrL, tok_las)
            s += "sent: %d/%d(%.5f)/%d(%.5f) || " % (sent_all, sent_corrU, sent_uas, sent_corrL, sent_las)
            s += "root: %d/%d(%.5f);%d/%d(%.5f)\n" % (tok_root_all, tok_corrR, tok_root_recall,
                                                      tok_root_pall, tok_corrR, tok_root_precision)
            # todo(warn): should look at LAS
            final_stat[one_suffix]["res"] = final_stat[one_suffix]["tok_las"]
        s += "== End Eval"
        # return
        if self.ignore_punct:
            return s, final_stat["_np"]
        else:
            return s, final_stat[""]

    def summary(self, verbose=False):
        return self.summary_one(self.stat, verbose)

