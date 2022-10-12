#

# open empty ones and split into different files
# (fix previous None ef/arg)

from collections import Counter
from msp2.utils import zlog, zopen, OtherHelper
from msp2.data.inst import Doc, Sent, Mention, set_ee_heads
from msp2.data.vocab import SimpleVocab
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
def main(input_file: str, output_prefix: str):
    docs = list(ReaderGetterConf().get_reader(input_path=input_file))
    zlog(f"Read docs from {input_file}: {len(docs)} docs")
    # --
    all_cc = Counter()
    good_ones, fixed_ones = [], []
    for doc in docs:
        all_cc['doc'] += 1
        doc_flag = 0
        for sent in doc.sents:
            all_cc['sent'] += 1
            sent_flag = 0
            if sent.entity_fillers is None:
                sent.entity_fillers = []
                doc_flag = sent_flag = 1
            if sent.events is None:
                sent.events = []
                doc_flag = sent_flag = 1
            else:
                for evt in sent.events:
                    if evt.args is None:
                        evt.args = []
                        doc_flag = sent_flag = 1
                        all_cc['evt1'] += 1
                    else:
                        all_cc['evt0'] += 1
            all_cc[f'sent{sent_flag}'] += 1
        all_cc[f'doc{doc_flag}'] += 1
        if doc_flag:
            fixed_ones.append(doc)
        else:
            good_ones.append(doc)
    # --
    OtherHelper.printd(all_cc)
    if len(good_ones) > 0:
        with WriterGetterConf().get_writer(output_path=f"{output_prefix}.f0.json") as writer:
            writer.write_insts(good_ones)
    if len(fixed_ones) > 0:
        with WriterGetterConf().get_writer(output_path=f"{output_prefix}.f1.json") as writer:
            writer.write_insts(fixed_ones)
    return docs

# --
# PYTHONPATH=?? python3 fix_empty.py <INPUT> <OUTPREFIX>
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
