#

# filter out exemplar sentences that appear in fulltext or UNANN ones

import sys
import json
from collections import Counter
from msp2.utils import zlog, OtherHelper
from msp2.data.rw import DataReader, LineStreamer, DataWriter, get_text_dumper

def main(input_fulltext: str, input_exemplars: str, output_exemplars: str):
    hit_keys = set()
    cc = Counter()
    # first read fulltext
    ft_docs = list(DataReader(LineStreamer(input_fulltext), "zjson_doc"))
    for doc in ft_docs:
        cc["doc"] += 1
        for sent in doc.sents:
            cc["sent"] += 1
            tok_key = ''.join(sent.seq_word.vals).lower()
            tok_key = ''.join(tok_key.split())  # split and join again
            hit_keys.add(tok_key)
    cc["sent_key"] = len(hit_keys)
    # then filter exemplars
    with DataWriter(get_text_dumper(output_exemplars)) as writer:
        for one_exemplar_sent in DataReader(LineStreamer(input_exemplars), "zjson_sent"):
            cur_key = ''.join(one_exemplar_sent.seq_word.vals).lower()
            cc["sent_e_all"] += 1
            delete_reason = None
            if cur_key in hit_keys:
                delete_reason = "ft"
            norole = False
            for evt in one_exemplar_sent.events:
                # todo(note): still keep the ones that have no args (around 4k in fn15, 5k in fn17)
                if evt.info["status"] == "UNANN" and len(evt.args)==0:
                    norole = True
            if norole:
                delete_reason = "unann"
            cc[f"sent_e_{delete_reason}"] += 1
            if delete_reason is None:
                writer.write_inst(one_exemplar_sent)
    # =====
    zlog(f"Finished filter_exemplars: {input_fulltext} + {input_exemplars} -> {output_exemplars}")
    OtherHelper.printd(cc)

if __name__ == '__main__':
    main(*sys.argv[1:])

# ====
"""
# fn15
PYTHONPATH=../../src/ python3 fn_filter_exemplars.py fn15/fulltext.json fn15/exemplars.json fn15/exemplars.filtered.json
# --
doc: 78
sent: 5946
sent_e_None: 154452
sent_e_all: 154485
sent_e_ft: 28
sent_e_unann: 5
sent_key: 5513
# fn17
PYTHONPATH=../../src/ python3 fn_filter_exemplars.py fn17/fulltext.json fn17/exemplars.json fn17/exemplars.filtered.json
# --
doc: 107
sent: 10147
sent_e_None: 173029
sent_e_all: 200744
sent_e_ft: 27661
sent_e_unann: 54
sent_key: 9648
"""
