#

# d-parser evaluation
from msp.utils import Helper, NumHelper

class ParserEvaler:
    # for UD, PTB, CTB
    # DEFAULT_PUNCT_POS_SET = {"PUNCT", "SYM", '.', '``', "''", ':', ',', 'PU'}
    # todo(warn): following CoNLL, no excluding punct for UD
    DEFAULT_PUNCT_POS_SET = {'.', '``', "''", ':', ',', 'PU'}
    # FUNCTION relations + punct relation
    CLAS_EXCLUDE_LABELS = {'aux', 'case', 'cc', 'clf', 'cop', 'det', 'mark', 'punct'}

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
    def eval_one(self, gold_pos, gold_heads, gold_labels, pred_pos, pred_heads, pred_labels):
        curr_stat = {}
        # add 1 for current stat
        def _a1cs(_name):
            Helper.stat_addone(curr_stat, _name)
        # =====
        # collect children info for UD***
        gold_children, pred_children = [[] for _ in range(len(gold_pos)+1)], [[] for _ in range(len(gold_pos)+1)]
        gold_children_labels, pred_children_labels = [[] for _ in range(len(gold_pos)+1)], [[] for _ in range(len(gold_pos)+1)]
        for i, m in enumerate(gold_heads):
            gold_children[m].append(i)
            gold_children_labels[m].append(gold_labels[i])
        for i, m in enumerate(pred_heads):
            pred_children[m].append(i)
            pred_children_labels[m].append(pred_labels[i])
        gold_children, pred_children = gold_children[1:], pred_children[1:]
        gold_children_labels, pred_children_labels = gold_children_labels[1:], pred_children_labels[1:]
        # collect labels info for CLAS
        gold_label_inc = [(z.split(":")[0] not in ParserEvaler.CLAS_EXCLUDE_LABELS) for z in gold_labels]
        pred_label_inc = [(z.split(":")[0] not in ParserEvaler.CLAS_EXCLUDE_LABELS) for z in pred_labels]
        # =====
        _a1cs("sent")
        cur_len = len(gold_pos)
        if pred_pos is None:
            pred_pos = [""] * cur_len
        for idx in range(cur_len):
            cur_gold_pos, cur_pred_pos = gold_pos[idx], pred_pos[idx]
            cur_gold_head, cur_gold_label, cur_gold_children, cur_gold_children_labels = \
                gold_heads[idx], gold_labels[idx], gold_children[idx], gold_children_labels[idx]
            cur_pred_head, cur_pred_label, cur_pred_children, cur_pred_children_labels = \
                pred_heads[idx], pred_labels[idx], pred_children[idx], pred_children_labels[idx]
            # token count
            _a1cs("tok")
            if cur_gold_pos not in self.punct_set:
                the_suffixes = ("", "_np")
                _a1cs("tok_np")
            else:
                the_suffixes = ("", )
            # detailed evals
            for one_suffix in the_suffixes:
                labeled_correct = False
                # pos
                if cur_gold_pos == cur_pred_pos:
                    _a1cs("pos_corr" + one_suffix)
                # plain UAS/LAS
                if cur_gold_head == cur_pred_head:
                    _a1cs("tok_corrU" + one_suffix)
                    if cur_gold_label == cur_pred_label:
                        labeled_correct = True
                        _a1cs("tok_corrL" + one_suffix)
                    if one_suffix == "":
                        self.add_confusion(cur_gold_label, cur_pred_label)
                # HDUAS/HDLAS
                if cur_gold_head == cur_pred_head and cur_gold_children == cur_pred_children:
                    _a1cs("tok_corrHDU" + one_suffix)
                    if cur_gold_label == cur_pred_label and cur_gold_children_labels == cur_pred_children_labels:
                        _a1cs("tok_corrHDL" + one_suffix)
                # CLAS
                if gold_label_inc[idx]:
                    _a1cs("tok_clas_all"+one_suffix)
                    if labeled_correct:
                        _a1cs("tok_clas_corr"+one_suffix)  # must be also included by pred if correct
                if pred_label_inc[idx]:
                    _a1cs("tok_clas_pall"+one_suffix)
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
        # =====
        Helper.stat_addv(self.stat, curr_stat)
        return curr_stat

    # return (report_str, result)
    def summary_one(self, stat, verbose):
        _P = lambda x: NumHelper.truncate_float(x, 6)
        _DIV = lambda a, b: 0. if b==0 else a/b
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
            # token: plain
            tok_all = stat.get("tok"+one_suffix, 0)
            tok_corrU = stat.get("tok_corrU"+one_suffix, 0)
            tok_corrL = stat.get("tok_corrL"+one_suffix, 0)
            tok_uas = _P(tok_corrU/tok_all)
            tok_las = _P(tok_corrL/tok_all)
            # special: root
            tok_root_all = stat.get("tok_root_all", 0)
            tok_root_pall = stat.get("tok_root_pall", 0)
            tok_root_corr = stat.get("tok_root_corr", 0)
            tok_root_recall = _P(_DIV(tok_root_corr, tok_root_all))
            tok_root_precision = _P(_DIV(tok_root_corr, tok_root_pall))
            tok_root_f1 = _P(_DIV(2*tok_root_recall*tok_root_precision, tok_root_recall+tok_root_precision))
            # special: hd
            tok_corrHDU = stat.get("tok_corrHDU"+one_suffix, 0)
            tok_corrHDL = stat.get("tok_corrHDL"+one_suffix, 0)
            tok_hduas = _P(tok_corrHDU/tok_all)
            tok_hdlas = _P(tok_corrHDL/tok_all)
            # special: clas
            tok_clas_all = stat.get("tok_clas_all"+one_suffix, 0)
            tok_clas_pall = stat.get("tok_clas_pall"+one_suffix, 0)
            tok_clas_corr = stat.get("tok_clas_corr"+one_suffix, 0)
            tok_clas_recall = _P(_DIV(tok_clas_corr, tok_clas_all))
            tok_clas_precision = _P(_DIV(tok_clas_corr, tok_clas_pall))
            tok_clas_f1 = _P(_DIV(2*tok_clas_recall*tok_clas_precision, tok_clas_recall+tok_clas_precision))
            # special: pos for all (including punct)
            tok_pos_corr = stat.get("pos_corr", 0)
            tok_pos_all = stat.get("tok", 0)
            tok_pos_acc = _P(_DIV(tok_pos_corr, tok_pos_all))
            #
            final_stat[one_suffix] = {
                "sent_uas": sent_uas, "sent_las": sent_las, "tok_uas": tok_uas, "tok_las": tok_las,
                "tok_root_rec": tok_root_recall, "tok_root_pre": tok_root_precision, "tok_root_f1": tok_root_f1,
                "tok_hduas": tok_hduas, "tok_hdlas": tok_hdlas,
                "tok_clas_recall": tok_clas_recall, "tok_clas_precision": tok_clas_precision, "tok_clas_f1": tok_clas_f1,
                "tok_pos_acc": tok_pos_acc,
            }
            s += "== Eval of " + one_suffix + ": "
            s += f"tok: {tok_all:d}/{tok_corrU:d}({tok_uas:.5f})/{tok_corrL:d}({tok_las:.5f}) || "
            s += f"pos: {tok_pos_acc:5f} || "
            s += f"sent: {sent_all:d}/{sent_corrU:d}({sent_uas:.5f})/{sent_corrL:d}({sent_las:.5f}) || "
            s += f"root: {tok_root_all:d}/{tok_root_corr:d}({tok_root_recall:.5f});{tok_root_pall:d}/{tok_root_corr:d}({tok_root_precision:.5f}) [{tok_root_f1:.5f}] || "
            s += f"hd: {tok_all:d}/{tok_corrHDU:d}({tok_hduas:.5f})/{tok_corrHDL:d}({tok_hdlas:.5f}) || "
            s += f"clas: {tok_clas_all:d}/{tok_clas_corr:d}({tok_clas_recall:.5f});{tok_clas_pall:d}/{tok_clas_corr:d}({tok_clas_precision:.5f}) [{tok_clas_f1:.5f}]\n"
            # todo(warn): here we average uas and las?
            # final_stat[one_suffix]["res"] = final_stat[one_suffix]["tok_las"]
            final_stat[one_suffix]["res"] = (final_stat[one_suffix]["tok_las"]+final_stat[one_suffix]["tok_uas"])/2
        s += "== End Eval"
        # return
        if self.ignore_punct:
            return s, final_stat["_np"]
        else:
            return s, final_stat[""]

    def summary(self, verbose=False):
        return self.summary_one(self.stat, verbose)
