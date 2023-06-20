# --

from mspx.utils import init_everything, zlog
from mspx.tools.go import *

def ana_main(*args):
    conf = init_everything(GoerConf(), args)
    goer = Goer(conf)
    goer.main()
    # --

# PYTHONPATH=../src/ python3 -mpdb -m mspx.cli.go ...
if __name__ == '__main__':
    import sys
    ana_main(*sys.argv[1:])

# examples
"""
python3 -m mspx.scripts.tools.run_para -i *.json -c "echo [[IN]] [[OUT]]" -o "x+'.what'"
python3 -m mspx.cli.go --inputs: *.json -- "cmd:echo [[IN]] [[OUT]]" "i2o_patterns:OUT:x+'.what'"
python3 -m mspx.cli.go cpus:3 req_rs:cpu:1 --inputs: *.json -- "cmd:[py]import time; time.sleep(1); print('[[IN]] [[OUT]]')" "i2o_patterns:OUT:x+'.what'"
# another example:
PYTHONPATH=`readlink -f ../../src/` python3 -m mspx.cli.go cpus:3 req_rs:cpu:1 --inputs: zh_ext/AA/wiki_?? -- "cmd:python3 -m mspx.scripts.data.mlm.proc_wiki cl:zh input_path:[[IN]] output_path:[[IN]].tok.bz2 stanza_processors:tokenize 2>&1 | tee [[IN]].tok.log"
# --
python3 -mpdb -m mspx.cli.go gpus:0,1,2 input_table_file:t.py name:debug220705
python3 -mpdb -m mspx.cli.go gpus:0,1,2 input_table_file:t.py name:debug220705 para:0 var:V:Y::NP:Y::PYOPTS:-mpdb
python3 -mpdb -m mspx.cli.go gpus:0,1,2 input_table_file:t.py name:debug220705 sels:A1B
"""
