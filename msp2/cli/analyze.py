#

# analyzer

import sys
from msp2.utils import Conf, zlog, init_everything
from msp2.tools.analyze import Analyzer

class MainConf(Conf):
    def __init__(self):
        self.do_loop = True

def main(analyzer: str, *args):
    conf = MainConf()
    a_res = Analyzer.try_load_and_lookup(analyzer)
    one_conf, one_type = a_res.conf, a_res.T
    conf.aconf = one_conf()
    # --
    conf = init_everything(conf, args)
    zlog(f"Ready to analyze with {analyzer}.")
    # --
    ana: Analyzer = one_type(conf.aconf)
    if conf.do_loop:
        ana.main()
    # --

# PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.analyze frame ...
if __name__ == '__main__':
    main(*sys.argv[1:])
