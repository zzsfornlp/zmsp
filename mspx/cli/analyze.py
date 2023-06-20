# --

from mspx.utils import init_everything, zlog
from mspx.tools.analyze import *

def ana_main(analyzer: str, *args):
    conf = AnalyzerConf.key2cls(analyzer)()
    conf = init_everything(conf, args)
    zlog(f"Ready to analyze with {analyzer}.")
    ana: Analyzer = conf.make_node()
    ana.main()
    # --

# PYTHONPATH=../src/ python3 -mpdb -m mspx.cli.analyze WHAT ...
if __name__ == '__main__':
    import sys
    ana_main(*sys.argv[1:])
