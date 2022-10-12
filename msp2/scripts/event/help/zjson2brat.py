#

# convert zjson docs to brat

from collections import Counter
from msp2.utils import zlog, zopen
from msp2.data.inst import Doc, Sent, Mention, set_ee_heads
from msp2.data.vocab import SimpleVocab
from msp2.data.rw import ReaderGetterConf

# --
def convert_ann(doc):
    cc = Counter()
    lines = []
    # --
    doc_text = doc.get_text()
    mention_map = {}  # id->E??
    # first mentions
    cc["doc"] += 1
    for sent in doc.sents:
        cc["sent"] += 1
        positions = sent.word_positions
        for iname, items in zip(["ef", "evt"], [sent.entity_fillers, sent.events]):
            if items is not None:
                for one_item in items:
                    cc[iname] += 1
                    assert id(one_item) not in mention_map
                    name = f"T{len(mention_map)}"
                    mention_map[id(one_item)] = name
                    _widx, _wlen = one_item.mention.get_span()
                    _cstart, _cend = positions[_widx][0], sum(positions[_widx+_wlen-1])
                    lines.append(f"{name}\t{one_item.label} {_cstart} {_cend}\t{doc_text[_cstart:_cend]}\n")
    # then events
    evt_map = {}
    for sent in doc.sents:
        if sent.events is not None:
            for evt in sent.events:
                assert id(evt) not in evt_map
                name = f"E{len(evt_map)}"
                evt_map[id(evt)] = name
                arg_str = f"{evt.label}:{mention_map[id(evt)]}"
                if evt.args is not None:
                    for aa in evt.args:
                        cc["arg"] += 1
                        arg_str += f" {aa.label}:{mention_map[id(aa.arg)]}"
                lines.append(f"{name}\t{arg_str}\n")
    # --
    return "".join(lines), cc

# --
def main(input_file: str, output_prefix=""):
    docs = list(ReaderGetterConf().get_reader(input_path=input_file))
    zlog(f"Read docs from {input_file}: {len(docs)} docs")
    # --
    all_cc = Counter()
    for doc in docs:
        cur_doc_id = doc.id
        assert cur_doc_id is not None
        _pref = f"{output_prefix}{cur_doc_id}"
        # first write txt
        with zopen(_pref+".txt", 'w') as fd:
            fd.write(doc.get_text())
        # then write ann
        one_ss, one_cc = convert_ann(doc)
        with zopen(_pref+".ann", 'w') as fd:
            fd.write(one_ss)
        all_cc += one_cc
    # --
    zlog(f"Finished: {all_cc}")
    return docs

# --
# PYTHONPATH=?? python3 zjson2brat.py <INPUT> <OUTPREFIX>
# -- see "zop20/brat_jun_191004/run.sh"
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
