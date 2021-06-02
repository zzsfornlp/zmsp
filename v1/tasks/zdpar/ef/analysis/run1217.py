#

# case study for the ef one
# error breakdown on labels,steps; which ones are "easy"(first-decoded) ones

#

import sys
from collections import Counter
from msp.zext.ana import AnalyzerConf, Analyzer, ZRecNode, AnnotationTask

try:
    from .ann import *
except:
    from ann import *

#
class NewAnalysisConf(AnalysisConf):
    def __init__(self, args):
        super().__init__(args)
        #
        self.step_div = 5
        self.use_label0 = True

def main(args):
    conf = NewAnalysisConf(args)
    # =====
    if conf.load_name == "":
        # recalculate them
        # read them
        zlog("Read them all ...")
        gold_parses = list(yield_ones(conf.gold))
        sys_parses = [list(yield_ones(z)) for z in conf.fs]
        if conf.use_label0:
            # todo(note): force using label0 (language-independent)
            zlog("Force label0 ...")
            for one_parses in sys_parses + [gold_parses]:
                for one_parse in one_parses:
                    for one_token in one_parse.get_tokens():
                        one_token.label = one_token.label0
        # use vocab?
        voc = Vocab.read(conf.vocab) if len(conf.vocab)>0 else None
        # =====
        # stat them
        zlog("Stat them all ...")
        all_sents, all_tokens = get_objs(gold_parses, sys_parses, conf.getter)
        analyzer = ParsingAnalyzer(conf.ana, all_sents, all_tokens, conf.labeled, vocab=voc)
        analyzer.set_var("nsys", len(conf.fs), explanation="init", history_idx=-1)
        if conf.save_name != "":
            analyzer.do_save(conf.save_name)
    else:
        analyzer = ParsingAnalyzer(conf.ana, None, None, conf.labeled)
        analyzer.do_load(conf.load_name)
    # =====
    # special analysis
    # ----
    def _num_same_sibs(_node):
        # todo(note): here split label again
        _lab = _node.label
        if conf.use_label0:
            _count = sum(z.split(":")[0]==_lab for z in _node.get_head().childs_labels)
        else:
            _count = sum(z==_lab for z in _node.get_head().childs_labels)
        assert _count>=1
        return _count-1
    # ----
    all_sents = analyzer.get_var("sents")
    nsys = analyzer.get_var("nsys")
    step_div = conf.step_div  # how many bins for ef-steps?
    breakdown_labels = {}  # label -> {gold: {count, numsib, dist}, preds: [{count, numsib, dist, lcorr, stepp}]}
    for _lab in ulabel2type["Nivre17"].keys():
        breakdown_labels[_lab] = {"gold": {"count": 0, "numsib": 0, "dist": 0},
                                  "preds": [{"count": 0, "numsib": 0, "dist": 0, "lcorr": 0, "stepp": 0} for _ in range(nsys)]}
    breakdown_steps = {}  # stepbin -> {count, dist, counter(label), acc, acc-all}
    for _stepbin in range(step_div):
        breakdown_steps[_stepbin] = {"count": 0, "dist": 0, "labels": Counter(), "lcorrs": [0]*nsys}
    # -----
    # collect
    for one_sobj in all_sents:
        cur_length = one_sobj.len
        for one_tobj in one_sobj.rtoks:  # all real toks
            # -----
            # get stat
            gold_label = one_tobj.g.label
            gold_numsib = _num_same_sibs(one_tobj.g)
            gold_dist = abs(one_tobj.g.ddist)
            # breakdown-label
            breakdown_labels[gold_label]["gold"]["count"] += 1
            breakdown_labels[gold_label]["gold"]["numsib"] += gold_numsib
            breakdown_labels[gold_label]["gold"]["dist"] += gold_dist
            for i, p in enumerate(one_tobj.ss):
                pred_label = p.label
                if pred_label in ["<z_non_z>", "<z_r_z>"]:
                    pred_label = "dep"  # todo(note): fix padding prediction
                pred_numsib = _num_same_sibs(p)
                pred_dist = abs(p.ddist)
                pred_lcorr = p.lcorr
                pred_stepi = getattr(p, "efi", None)
                if pred_stepi is None:
                    pred_stepi = getattr(p, "gmi", None)
                assert pred_stepi is not None
                pred_stepbin = int(pred_stepi*step_div/cur_length)
                pred_stepp = pred_stepi / cur_length
                # breakdown-label
                breakdown_labels[pred_label]["preds"][i]["count"] += 1
                breakdown_labels[pred_label]["preds"][i]["numsib"] += pred_numsib
                breakdown_labels[pred_label]["preds"][i]["dist"] += pred_dist
                breakdown_labels[pred_label]["preds"][i]["lcorr"] += pred_lcorr
                breakdown_labels[pred_label]["preds"][i]["stepp"] += pred_stepp
                # breakdown-steps
                if i==0:  # todo(note): only record the first one!!
                    breakdown_steps[pred_stepbin]["count"] += 1
                    breakdown_steps[pred_stepbin]["dist"] += pred_dist
                    breakdown_steps[pred_stepbin]["labels"][pred_label] += 1
                    for i2, p2 in enumerate(one_tobj.ss):  # all nodes' correctness for this certain node!
                        breakdown_steps[pred_stepbin]["lcorrs"][i2] += p2.lcorr
    # -----
    # summary
    data_labels = []
    for k, dd in breakdown_labels.items():
        gold_count = max(dd["gold"]["count"], 1e-5)
        res = {"K": k, "gold_count": gold_count, "numsib": dd["gold"]["numsib"]/gold_count,
               "dist": dd["gold"]["dist"]/gold_count}
        for pidx, preds in enumerate(dd["preds"]):
            pred_count = max(preds["count"], 1e-5)
            res[f"pred{pidx}_count"] = pred_count
            res[f"pred{pidx}_numsib"] = preds["numsib"]/pred_count
            res[f"pred{pidx}_dist"] = preds["dist"]/pred_count
            res[f"pred{pidx}_stepp"] = preds["stepp"]/pred_count
            P, R = preds["lcorr"]/pred_count, preds["lcorr"]/gold_count
            F = 2*P*R/(P+R) if (P+R)>0 else 0.
            res.update({f"pred{pidx}_P": P, f"pred{pidx}_R": R, f"pred{pidx}_F": F})
        data_labels.append(res)
    data_steps = []
    TOP_LABEL_K = 5
    for k, dd in breakdown_steps.items():
        dd_count = max(dd["count"], 1e-5)
        res = {"K": k, "count": dd_count, "dist": dd["dist"]/dd_count}
        for common_idx, common_p in enumerate(dd["labels"].most_common(TOP_LABEL_K)):
            common_label, common_count = common_p
            res[f"common{common_idx}"] = f"{common_label}({common_count/dd['count']:.3f})"
        for pidx, pcorr in enumerate(dd["lcorrs"]):
            res[f"pred{pidx}_acc"] = pcorr/dd_count
        data_steps.append(res)
    # =====
    pd_labels = pd.DataFrame({k: [d[k] for d in data_labels] for k in data_labels[0].keys()})
    pd_labels = pd_labels.sort_values(by="gold_count", ascending=False)
    selections = ["K", "gold_count", "numsib", "dist", "pred0_numsib", "pred0_stepp", "pred0_F",
                  "pred1_numsib", "pred1_stepp", "pred1_F"]
    pd_labels2 = pd_labels[selections]
    pd_steps = pd.DataFrame({k: [d[k] for d in data_steps] for k in data_steps[0].keys()})
    zlog(f"#-----\nLABELS: \n{pd_labels2.to_string()}\n\n")
    zlog(f"#-----\nSTEPS: \n{pd_steps.to_string()}\n\n")
    # specific table
    TABLE_LABEL_K = 10
    num_all_tokens = sum(z["gold_count"] for z in data_labels)
    lines = []
    for i in range(TABLE_LABEL_K):
        ss = pd_labels.iloc[i]
        fields = [ss["K"], f"{ss['gold_count']/num_all_tokens:.2f}", f"{ss['numsib']:.2f}", f"{ss['dist']:.2f}",
                  f"{ss['pred0_F']*100:.2f}", f"{ss['pred0_numsib']:.2f}",
                  f"{ss['pred1_F']*100:.2f}", f"{ss['pred1_numsib']:.2f}"]
        lines.append(" & ".join(fields))
    table_ss = "\\\\\n".join(lines)
    zlog(f"#=====\n{table_ss}")
    # -----
    # import pdb
    # pdb.set_trace()
    return

if __name__ == '__main__':
    main(sys.argv[1:])

# runnings
"""
PYTHONPATH=../../src/ python3 -m pdb run.py gold:en_dev.gold fs:en_dev.zef_ru.pred,en_dev.zg1_ru.pred
"""
