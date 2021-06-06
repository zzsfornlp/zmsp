#

# align sentences from two files

from msp2.utils import zlog
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
import numpy as np
from collections import OrderedDict

# --
def main(output_prefix, *input_files):
    # input
    all_sents = []
    for f in input_files:
        one_reader = ReaderGetterConf().get_reader(input_path=f)
        one_insts = list(one_reader)
        all_sents.append([z for z in yield_sents(one_insts)])
        zlog(f"Read from {f}: {len(all_sents[-1])} sents")
    # align
    sent_map = OrderedDict()
    for fidx, sents in enumerate(all_sents):
        for sent in sents:
            doc_id = sent.info.get("doc_id", "UNK")
            if doc_id.split("/", 1)[0] == "ontonotes":
                doc_id = doc_id.split("/", 1)[1]
            key = doc_id + "|".join(sent.seq_word.vals)  # map by doc_id + key
            if key not in sent_map:
                sent_map[key] = [sent]
            else:
                sent_map[key].append(sent)
    # --
    num_files = len(input_files)
    matched_sents = [vs for vs in sent_map.values() if len(vs)==num_files]
    unmatched_sents = [vs for vs in sent_map.values() if len(vs)!=num_files]
    zlog(f"Aligned sent of {len(matched_sents)}")
    breakpoint()
    # output
    for outi in range(num_files):
        out_sents = [z[outi] for z in matched_sents]
        writer = WriterGetterConf().get_writer(output_path=f"{output_prefix}{outi}")
        writer.write_insts(out_sents)
        writer.close()
    # --

# PYTHONPATH=../src/ python3 align_sents.py _tmp.json ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
