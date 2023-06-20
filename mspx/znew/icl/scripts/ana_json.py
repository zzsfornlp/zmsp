#

# analyze with json files

import pandas as pd
import numpy as np
from mspx.utils import default_json_serializer, init_everything, zlog, zglob1
from mspx.data.vocab import SeqVocab
from mspx.data.rw import WriterGetterConf
from mspx.proc.eval import FrameEvalConf
from mspx.tools.analyze import AnalyzerConf, Analyzer, MatchedList

class MyAnalyzerConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        self.gold = ""
        self.preds = []

@MyAnalyzerConf.conf_rd()
class MyAnalyzer(Analyzer):
    def __init__(self, conf: MyAnalyzerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MyAnalyzerConf = self.conf
        # --
        all_insts = self.read_data()
        self.set_var("il", all_insts, explanation="init")  # sent pair

    def read_data(self):
        conf: MyAnalyzerConf = self.conf
        # --
        gold_insts = default_json_serializer.load_list(conf.gold)
        all_pred_insts = [default_json_serializer.load_list(z) for z in conf.preds]
        assert all(len(z)==len(gold_insts) for z in all_pred_insts)
        # --
        inst_lists = [MatchedList([gg] + [z[ii] for z in all_pred_insts]) for ii,gg in enumerate(gold_insts)]
        # breakpoint()
        return inst_lists

def ana_main(*args):
    conf = MyAnalyzerConf()
    conf: MyAnalyzerConf = init_everything(conf, args)
    ana: MyAnalyzer = conf.make_node()
    ana.main()
    # --

# python3 -mpdb -m mspx.znew.icl.scripts.ana_json
if __name__ == '__main__':
    import sys
    ana_main(*sys.argv[1:])
