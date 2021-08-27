#

# simple stat

from msp2.utils import zlog, zwarn, zopen, default_json_serializer, OtherHelper
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.data.vocab import SimpleVocab
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from collections import Counter

# --
def main(input_format, *input_files: str):
    reader_conf = ReaderGetterConf().direct_update(input_format=input_format)
    reader_conf.validate()
    # --
    all_insts = []
    for ff in input_files:
        one_insts = list(reader_conf.get_reader(input_path=ff))
        cc = Counter()
        for sent in yield_sents(one_insts):
            cc['sent'] += 1
            for evt in sent.events:
                cc['evt'] += 1
                cc['arg'] += len(evt.args)
        zlog(f"Read from {ff}: {cc['sent']/1000:.1f}k&{cc['evt']/1000:.1f}k&{cc['arg']/1000:.1f}k")
    # --

# --
# PYTHONPATH=../src/ python3 stat_simple.py [format] *files
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
