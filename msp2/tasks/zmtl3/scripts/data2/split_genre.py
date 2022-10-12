#

# split genre for onto & ace

import sys
from collections import Counter
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zlog, zwarn

# --
def judge_ace_genre(doc_id: str):
    cols = [
       ('bc', ["CNN_CF", "CNN_IP", "CNN_LE"]),
       ('bn', ["CNN_ENG", "CNNHL_ENG"]),
       ('cts', ["fsh"]),
       ('nw', ["AFP", "APW", "NYT", "XIN"]),
       ('un', ["alt", "aus", "Austin", "Integritas", "marcellapr", "misc", "rec", "seattle", "soc", "talk", "uk"]),
       ('wl', ["AGGRESSIVEVOICEDAILY", "BACONSREBELLION", "FLOPPINGACES", "GETTINGPOLITICAL", "HEALINGIRAQ", "MARKBACKER", "MARKETVIEW", "OIADVANTAGE", "TTRACY"]),
    ]
    for gg, pp in cols:
        if any(doc_id.startswith(p) for p in pp):
            return gg
    raise RuntimeError()
# --

def stat_doc(doc, cc):
    cc['doc'] += 1
    cc['sent'] += len(doc.sents)
    for sent in doc.sents:
        for evt in sent.events:
            cc['evt'] += 1
            cc['arg'] += len(evt.args)
    # --

# --
def main(input_file: str, output_prefix: str):
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    genre_docs = {}
    genre_ccs = {}
    for inst in reader:
        _id = inst.id
        if 'onto' in input_file:
            assert _id.startswith("ontonotes/")
            _genre = _id.split("/")[1]
        elif 'ace' in input_file:  # ace?
            _genre = judge_ace_genre(_id)
        else:
            raise NotImplementedError()
        # --
        if _genre not in genre_ccs:
            genre_ccs[_genre] = Counter()
            genre_docs[_genre] = []
        genre_docs[_genre].append(inst)
        stat_doc(inst, genre_ccs[_genre])
    # --
    for gg in sorted(genre_ccs.keys()):
        zlog(f"Output genre={gg}(L={len(genre_docs[gg])}): {genre_ccs[gg]}")
        with WriterGetterConf().get_writer(output_path=f"{output_prefix}.{gg}.json") as writer:
            writer.write_insts(genre_docs[gg])
    # --
# --

# python3 -m msp2.tasks.zmtl3.scripts.data2.split_genre IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])

# --
"""
# simply prepare things at qadistill
cat ../../events/data/data21f/en.ace2.{train,dev,test}.json >en.ace2.all.json
python3 -m msp2.tasks.zmtl3.scripts.data2.split_genre en.ace2.all.json en.ace2
# Output genre=bc(L=60): Counter({'sent': 3642, 'arg': 1266, 'evt': 914, 'doc': 60})
# Output genre=bn(L=226): Counter({'sent': 4125, 'arg': 1924, 'evt': 1184, 'doc': 226})
# Output genre=cts(L=39): Counter({'sent': 5877, 'arg': 576, 'evt': 468, 'doc': 39})
# Output genre=nw(L=105): Counter({'arg': 2603, 'sent': 2114, 'evt': 1542, 'doc': 105})
# Output genre=un(L=48): Counter({'sent': 2665, 'arg': 877, 'evt': 696, 'doc': 48})
# Output genre=wl(L=119): Counter({'sent': 2370, 'arg': 809, 'evt': 507, 'doc': 119})
cp ../../events/data/data21f/en.ontoC.dev.ud.json .
python3 -m msp2.tasks.zmtl3.scripts.data2.split_genre en.ontoC.train.ud.q1.json en.ontoC.train.ud.q1
# Output genre=bc(L=19): Counter({'arg': 85215, 'evt': 27809, 'sent': 10429, 'doc': 19})
# Output genre=bn(L=763): Counter({'arg': 94997, 'evt': 32059, 'sent': 9723, 'doc': 763})
# Output genre=mz(L=64): Counter({'arg': 80034, 'evt': 25841, 'sent': 6911, 'doc': 64})
# Output genre=nw(L=745): Counter({'arg': 175632, 'evt': 57630, 'sent': 15288, 'doc': 745})
# Output genre=pt(L=230): Counter({'arg': 134865, 'evt': 41314, 'sent': 15263, 'doc': 230})
# Output genre=tc(L=36): Counter({'arg': 52809, 'evt': 16581, 'sent': 11162, 'doc': 36})
# Output genre=wb(L=83): Counter({'arg': 61650, 'evt': 19777, 'sent': 6411, 'doc': 83})
python3 -m msp2.tasks.zmtl3.scripts.data2.split_genre en.ontoC.dev.ud.json en.ontoC.dev.ud
# Output genre=bc(L=4): Counter({'arg': 15305, 'evt': 4975, 'sent': 1946, 'doc': 4})
# Output genre=bn(L=91): Counter({'arg': 11726, 'evt': 3940, 'sent': 1172, 'doc': 91})
# Output genre=mz(L=7): Counter({'arg': 7196, 'evt': 2317, 'sent': 642, 'doc': 7})
# Output genre=nw(L=88): Counter({'arg': 24503, 'evt': 8183, 'sent': 2054, 'doc': 88})
# Output genre=pt(L=15): Counter({'arg': 10039, 'evt': 3065, 'sent': 1075, 'doc': 15})
# Output genre=tc(L=5): Counter({'arg': 7769, 'evt': 2447, 'sent': 1634, 'doc': 5})
# Output genre=wb(L=12): Counter({'arg': 9210, 'evt': 2905, 'sent': 1080, 'doc': 12})
"""
