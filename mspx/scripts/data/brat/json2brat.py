#

# convert back to brat for visualization

import os
import sys
from collections import OrderedDict, Counter
from mspx.utils import zopen, zlog, mkdir_p
from mspx.data.rw import ReaderGetterConf, WriterGetterConf, BratFormator, BratFormatorConf

def main(input_file: str, output_prefix: str):
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    conf_convert = BratFormatorConf.direct_conf(char_map=":_", _finish=True)  # not configurable!
    converter = BratFormator(conf_convert)
    cc = converter.write_brat(reader, output_prefix)
    zlog(f"Read from {input_file} to {output_prefix}: {cc}")
    # --

# python3 -m mspx.scripts.data.brat.json2brat ...
if __name__ == '__main__':
    main(*sys.argv[1:])
