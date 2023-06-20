#

# preprocess for faster reading
# (aggregate sents + sub-tokenize)

import sys
from collections import Counter
from mspx.data.rw import ReaderGetterConf
from mspx.data.vocab import TokerPretrained
from mspx.utils import Conf, init_everything, zopen, zlog, zglob1, StatRecorder

class MainConf(Conf):
    def __init__(self):
        super().__init__()
        self.R = ReaderGetterConf()
        self.output_path = ""
        self.report_interval = 1000
        self.toker = ""
        self.aspec = "60:128"  # [len0,len1]
        # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    from mspx.proc.run import yield_aggr_sents
    toker = TokerPretrained(conf.toker)
    _len_f = lambda x: len(x.sent.seq_word.get_sf(sub_toker=toker))
    _aspec = conf.aspec.split(":")
    reader = conf.R.get_reader()
    zlog(f"Preprocess texts for {reader} with {toker}")
    rr = StatRecorder(report_key='doc', report_interval=conf.report_interval)
    with zopen(conf.output_path, 'w') as fd:
        for inst in reader:
            cc = Counter()
            cc['doc'] += 1
            cc['sent_orig'] += len(inst.sents)
            cc['tok_orig'] += sum(len(z) for z in inst.sents)
            for a_sent in yield_aggr_sents([inst], *_aspec, len_f=_len_f):
                subtoks = a_sent.seq_word.get_sf(sub_toker=toker).vals
                cc['seq_out'] += 1
                cc[f'seq_out_L=20*{len(subtoks)//20}'] += 1
                cc['subtok_out'] += len(subtoks)
                cc['sent_out'] += len(a_sent.cache['combine_sents'])
                cc['tok_out'] += len(a_sent)
                fd.write(" ".join(subtoks) + "\n")
            cc['sent_miss'] += cc['sent_orig'] - cc['sent_out']
            # --
            rr.record(cc)
    rr.summary("Finished")
    # --

# python3 -m mspx.scripts.data.mlm.preprocess ...
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# example
python3 -m mspx.scripts.data.mlm.preprocess input_path:dev.tok.json output_path:dev.tok.txt toker:bert-base-cased
python3 -m mspx.scripts.data.mlm.preprocess input_path:train0.tok.bz2 output_path:train0.tok.txt.bz2 toker:bert-base-cased
# wiki_0*:
# -- (old version: "60:100:0.5") 
#Finished: Counter({'subtok_out': 43949, 'tok_orig': 38213, 'tok_out': 37438, 'sent_orig': 1792, 'sent_out': 1745, 'seq_out': 427, 'doc': 100, 'sent_miss': 47, '_time_doc': 4.77}) [4.77s][20.98d/s]
#Finished: Counter({'subtok_out': 455554774, 'tok_orig': 393877069, 'tok_out': 391974683, 'sent_orig': 16877383, 'sent_out': 16726170, 'seq_out': 4251145, 'doc': 224649, 'sent_miss': 151213, '_time_doc': 44325.83}) [44325.83s][5.07d/s]
# -- (new_version: "60:139")
#Finished: Counter({'subtok_out': 42818, 'tok_orig': 38213, 'tok_out': 36499, 'sent_orig': 1792, 'sent_out': 1697, 'seq_out': 390, 'seq_out_L=20*5': 179, 'seq_out_L=20*6': 127, 'doc': 100, 'sent_miss': 95, 'seq_out_L=20*4': 62, 'seq_out_L=20*3': 22, '_time_doc': 4.75}) [4.75s][21.05d/s]
#
"""
