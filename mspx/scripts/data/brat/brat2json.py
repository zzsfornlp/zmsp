#

# read from brat (and tokenize and align mentions)

import os
import sys
from collections import OrderedDict, Counter
from mspx.utils import zopen, zlog, zwarn, Conf, init_everything, zglobs
from mspx.data.rw import ReaderGetterConf, WriterGetterConf, BratFormator, BratFormatorConf

# --
class MainConf(Conf):
    def __init__(self):
        self.input_path = []  # input dir!
        self.W = WriterGetterConf()
        self.convert = BratFormatorConf(default_toker='stanza')

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    converter = BratFormator(conf.convert)
    # --
    input_files = zglobs(conf.input_path)
    docs = list(converter.read_brat(input_files, cc=cc))
    zlog(f"Read from {input_files} to {conf.W}: {cc}")
    # --
    if conf.W.has_path():
        with conf.W.get_writer() as writer:
            writer.write_insts(docs)
    # --

# python3 -m mspx.scripts.data.brat.brat2json ...
if __name__ == '__main__':
    main(*sys.argv[1:])
