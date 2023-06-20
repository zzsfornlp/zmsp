#

# change format

import sys
from typing import List
from mspx.utils import Conf, zlog, init_everything
from mspx.data.rw import ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    reader = conf.R.get_reader()
    with conf.W.get_writer() as writer:
        for one in reader:
            writer.write_inst(one)
    # --

# PYTHONPATH=../src/ python3 -m mspx.cli.change_format R.input_path:? W.output_path:?
if __name__ == '__main__':
    main(*sys.argv[1:])
