#

# evaluator

import sys
from typing import List
from msp2.utils import Conf, zlog, init_everything, zopen
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.proc import Evaluator, EvalConf

class MainConf(Conf):
    def __init__(self):
        self.gold = ReaderGetterConf()
        self.pred = ReaderGetterConf()
        self.result_file = ""  # file to output details
        self.econf: EvalConf = None
        self.print_details = True  # whether print get_detailed_str()

def main(evaluator: str, *args):
    # find evaluator
    conf = MainConf()
    e_res = Evaluator.try_load_and_lookup(evaluator)
    one_conf, one_type = e_res.conf, e_res.T
    conf.econf = one_conf()
    # --
    conf = init_everything(conf, args)
    zlog(f"Ready to evaluate with {evaluator}: {conf.gold} {conf.pred}")
    # --
    gold_insts = list(conf.gold.get_reader())
    pred_insts = list(conf.pred.get_reader())
    evaler: Evaluator = one_type(conf.econf)
    res = evaler.eval(gold_insts, pred_insts)
    if conf.result_file:
        with zopen(conf.result_file, 'a') as fd:  # note: here we use append mode
            fd.write(f"# Eval with {args}:\n{res.get_brief_str()}\n{res.get_detailed_str()}\n")
    zlog(f"Eval on {conf.gold} vs. {conf.pred}; RESULT = {res}")
    if conf.print_details:
        zlog(f"#-- details:\n{res.get_detailed_str()}")

# PYTHONPATH=../src/ python3 -m msp2.cli.evaluate <??> gold.input_path:? pred.input_path:?
if __name__ == '__main__':
    main(*sys.argv[1:])
