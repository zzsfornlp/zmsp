#

# evaluator

import sys
from typing import List
from mspx.utils import Conf, zlog, init_everything, zopen
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.proc.eval import Evaluator, EvalConf

class MainConf(Conf):
    def __init__(self):
        self.pred = ReaderGetterConf()
        self.gold = ReaderGetterConf()
        self.result_file = ""  # file to output details
        self.econf: EvalConf = None
        self.print_details = True  # whether print get_detailed_str()

def main(e_name: str, *args):
    # find evaluator
    conf = MainConf()
    conf.econf = EvalConf.key2cls(e_name)()
    # --
    conf = init_everything(conf, args)
    evaler: Evaluator = conf.econf.make_node()
    zlog(f"Ready to evaluate with {evaler}: {conf.pred} {conf.gold}")
    # --
    res = evaler.eval(conf.pred.get_reader(), conf.gold.get_reader())
    if conf.result_file:
        with zopen(conf.result_file, 'a') as fd:  # note: here we use append mode
            fd.write(f"# Eval with {args}:\n{res.get_str(brief=True)}\n{res.get_str(brief=False)}\n")
    zlog(f"Eval on {conf.pred} vs {conf.gold}; RESULT = {res}")
    if conf.print_details:
        zlog(f"#-- details:\n{res.get_str(brief=False)}")
    # --

# PYTHONPATH=../src/ python3 -m mspx.cli.evaluate <??> gold.input_path:? pred.input_path:?
if __name__ == '__main__':
    main(*sys.argv[1:])
