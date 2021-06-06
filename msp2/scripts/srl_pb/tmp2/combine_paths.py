#

# compute the dep-paths to predicate (both gold and pred)
# -> used as input features for srl
# note: use the first one as the backbone!

import sys
from typing import List
from collections import Counter
from msp2.utils import Conf, zlog, OtherHelper
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        self.gold = ReaderGetterConf()
        self.pred = ReaderGetterConf()
        # --
        self.output = WriterGetterConf()

def main(args):
    conf = MainConf()
    conf.update_from_args(args)
    zlog(f"Ready to evaluate with: {conf.gold} {conf.pred} => {conf.output}")
    # --
    final_insts = list(conf.gold.get_reader())  # to modify inplace!
    stat = Counter()
    gold_sents = list(yield_sents(final_insts))
    pred_sents = list(yield_sents(conf.pred.get_reader()))
    assert len(gold_sents) == len(pred_sents)
    for g_sent, p_sent in zip(gold_sents, pred_sents):
        stat["sent"] += 1
        slen = len(g_sent)
        assert slen == len(p_sent)
        stat["tok"] += slen
        # put features
        assert len(g_sent.events) == len(p_sent.events)
        for g_evt, p_evt in zip(g_sent.events, p_sent.events):
            assert g_evt.mention.is_equal(p_evt.mention) and g_evt.label==p_evt.label
            stat["frame"] += 1
            stat["ftok"] += slen
            assert len(g_evt.args) == len(p_evt.args)
            # --
            evt_widx = g_evt.mention.shead_widx
            g_paths = [[len(z) for z in g_evt.sent.tree_dep.get_path(ii, evt_widx)] for ii in range(slen)]
            p_paths = [[len(z) for z in p_evt.sent.tree_dep.get_path(ii, evt_widx)] for ii in range(slen)]
            stat["ftok_corr"] += sum(a==b for a,b in zip(g_paths, p_paths))
            # assign
            g_evt.info["dpaths"] = [g_paths, p_paths]  # [2(g/p), SLEN, 2(word, predicate)]
        # --
    # --
    # report
    OtherHelper.printd(stat)
    zlog(f"FtokPathAcc: {stat['ftok_corr']} / {stat['ftok']} = {stat['ftok_corr']/stat['ftok']}")
    # --
    # write
    if conf.output.output_path:
        with conf.output.get_writer() as writer:
            writer.write_insts(final_insts)
    # --

# PYTHONPATH=../../../zsp2021/src/ python3 combine_paths.py gold.input_path:? pred.input_path:? output.output_path:
"""
# cp2
for ff in ../conll05/*.conll.ud.json ../conll12b/*.conll.ud.json; do PYTHONPATH=../../../zsp2021/src/ python3 combine_paths.py gold.input_path:$ff pred.input_path:${ff%.ud.json}.ud2.json output.output_path:${ff%.ud.json}.cp2.json; done |& tee _log_cp2
# cp3
for wset in dev test.wsj test.brown; do PYTHONPATH=../../src/ python3 combine_paths.py gold.input_path:../../pb/conll05/${wset}.conll.ud.json pred.input_path:./_zout.${wset}.conll.ud.json output.output_path:_c05.${wset}.cp3.json; done |& tee _log_cp3
"""
if __name__ == '__main__':
    main(sys.argv[1:])
