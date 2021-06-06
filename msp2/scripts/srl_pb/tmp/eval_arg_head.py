#

# eval dep and arg's head
# (strictly paired comparison!!)

import sys
from typing import List
from collections import Counter
from msp2.utils import Conf, zlog, init_everything, zopen
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        self.gold = ReaderGetterConf()
        self.pred = ReaderGetterConf()

def main(args):
    conf = MainConf()
    conf.update_from_args(args)
    zlog(f"Ready to evaluate with: {conf.gold} {conf.pred}")
    # --
    stat = Counter()
    gold_sents = list(yield_sents(conf.gold.get_reader()))
    pred_sents = list(yield_sents(conf.pred.get_reader()))
    assert len(gold_sents) == len(pred_sents)
    # --
    set_ee_heads(gold_sents)
    set_ee_heads(pred_sents)
    # --
    for g_sent, p_sent in zip(gold_sents, pred_sents):
        stat["sent"] += 1
        slen = len(g_sent)
        assert slen == len(p_sent)
        stat["tok"] += slen
        # check tokens
        for widx in range(slen):
            corr_pos = int(g_sent.seq_upos.vals[widx]==p_sent.seq_upos.vals[widx])
            corr_dhead = int(g_sent.tree_dep.seq_head.vals[widx]==p_sent.tree_dep.seq_head.vals[widx])
            corr_dlab = corr_dhead * int(g_sent.tree_dep.seq_label.vals[widx]==p_sent.tree_dep.seq_label.vals[widx])
            stat["tok_pos"] += corr_pos
            stat["tok_dhead"] += corr_dhead
            stat["tok_dlab"] += corr_dlab
        # check args
        assert len(g_sent.events) == len(p_sent.events)
        for g_evt, p_evt in zip(g_sent.events, p_sent.events):
            assert g_evt.mention.is_equal(p_evt.mention) and g_evt.label==p_evt.label
            stat["frame"] += 1
            assert len(g_evt.args) == len(p_evt.args)
            frame_head_corr = 1
            for g_arg, p_arg in zip(g_evt.args, p_evt.args):
                stat["arg"] += 1
                assert g_arg.mention.is_equal(p_arg.mention) and g_arg.label == p_arg.label
                arg_head_corr = int(g_arg.mention.shead_widx == p_arg.mention.shead_widx)
                stat["arg_head"] += arg_head_corr
                frame_head_corr *= arg_head_corr
            stat["frame_head"] += frame_head_corr
        # --
    # --
    # report
    for k,v in stat.items():
        v0 = 0
        if len(k.split("_", 1)) > 1:
            k0 = k.split("_", 1)[0]
            if k0 in stat:
                v0 = stat[k0]
        # --
        zlog(f"{k}: {v}" + (f" -> {v}/{v0}={v/v0}" if v0>0 else ""))
    # --

# PYTHONPATH=../../../zsp2021/src/ python3 eval_arg_head.py gold.input_path:? pred.input_path:?
# for ff in ../conll05/*.conll.ud.json ../conll12b/*.conll.ud.json ../conll12c/*.conll.ud.json; do python3 eval_arg_head.py gold.input_path:$ff pred.input_path:${ff%.ud.json}.ud2.json; done |& tee _log_eval_head
if __name__ == '__main__':
    main(sys.argv[1:])
