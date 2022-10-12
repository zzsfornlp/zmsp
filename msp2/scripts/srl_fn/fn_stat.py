#

# collect some statistics and produce related vocabs

import sys
from typing import Dict, List
from itertools import chain
from collections import Counter
from msp2.utils import zlog, default_json_serializer, OtherHelper
from msp2.data.inst import Doc, Sent, Mention
from msp2.data.vocab import SimpleVocab
from msp2.data.rw import DataReader, LineStreamer, DataWriter

# -----
def overlap_status(m1: Mention, m2: Mention):
    if m1.sent is m2.sent:
        start1, end1 = m1.widx, m1.wridx
        start2, end2 = m2.widx, m2.wridx
        if (start1, end1) > (start2, end2):  # small as 1
            start1, end1, start2, end2 = start2, end2, start1, end1
        if start2 >= end1:
            return "NoOverlap"
        assert start2 >= start1
        if start2 == start1:  # left-aligned
            assert end2 >= end1
            if end2 == end1:
                return "Same"
            else:
                return "Nesting"
        else:
            if end2 <= end1:
                return "Nesting"
            else:
                return "Crossing"
    else:
        return "DiffSent"

# helper
_cache_frames = None
def get_core_type(evt, role):
    global _cache_frames
    if _cache_frames is None:
        import os
        fname = os.environ.get("_CACHE_FRAMES", None)
        if fname is None: return "UNK"
        frames = default_json_serializer.from_file(fname)
        _cache_frames = {}
        for f, v in frames.items():
            evt_type = v["name"]
            one_entry = {}
            for fe in v["FE"]:
                arg_role, core_type = fe["name"], fe["coreType"]
                # assert arg_role not in one_entry
                if arg_role in one_entry:
                    zlog(f"warning: repeated arg entry in lexicon evt={evt_type} -> {fe} vs. {one_entry[arg_role]}")
                one_entry[arg_role] = core_type
            assert evt_type not in _cache_frames
            _cache_frames[evt_type] = one_entry
    # -----
    if evt not in _cache_frames:
        zlog(f"warning: unfound evt {evt}")
        return "UNK"
    if role not in _cache_frames[evt]:
        zlog(f"warning: unfound evt-role {evt}-{role}")
        return "UNK"
    return _cache_frames[evt][role]

def _stat_sent(sent: Sent):
    stat = Counter()
    stat["sent"] += 1
    stat[f"sent_NumEvt={min(5, len(sent.events))}"] += 1
    # =====
    stat["tok"] += len(sent)
    if sent.wsl is None:
        stat[f"tok_UNK"] += len(sent)
    else:
        for s in sent.wsl:
            stat[f"tok_{s}"] += 1
    # =====
    assert len(sent.entity_fillers) == sum(len(z.args) for z in sent.events)  # this is because of the corpus
    stat["evt"] += len(sent.events)
    for e in sent.events:
        evt_status = e.info.get('status')
        stat[f"evt_Status={evt_status}"] += 1
        if evt_status == "UNANN":
            assert len(e.args) == 0, "UNANN gets args?"
        if sent.wsl is not None:
            if any(sent.wsl[ii]=="NT" for ii in range(e.mention.widx, e.mention.widx+e.mention.wlen)):
                zlog(f"warning: Annotated NT (maybe caused by discontinous span): {e.mention}")
        # evt trigger length
        stat[f"evt-L={min(3, e.mention.wlen)}"] += 1
        # evt trigger continous?
        stat[f"evt-DTrig={e.info.get('DTrgs') is not None}"] += 1
        # check evt span overlap
        all_overlap_status = set()
        for e2 in sent.events:
            if e2 is not e:
                one_status = overlap_status(e.mention, e2.mention)
                if one_status != "NoOverlap":
                    all_overlap_status.add(one_status)
        stat[f"evt-Overlap={'|'.join(sorted(all_overlap_status))}"] += 1
        # =====
        iarg_counts = Counter()  # role -> count
        for iarg in e.iargs:
            stat[f"argI"] += 1
            stat[f"argI={iarg['itype']}"] += 1
            iarg_counts[iarg['role']] += 1
        for sarg in e.sargs:
            stat[f"argS"] += 1
            stat[f"argS={sarg['role']}"] += 1
        # if len(e.args) == 0:
        #     breakpoint()
        stat[f"evt_A={min(len(e.args), 5)}"] += 1
        # --
        for aprefix, core_only in zip(["ARG", "CARG"], [False, True]):
            arg_counts = Counter()  # role -> count
            for a in e.args:
                role = a.role
                if core_only:  # skip non-core ones!
                    if not get_core_type(e.type, role).startswith("Core"):
                        continue
                stat[f"{aprefix}"] += 1
                stat[f"{aprefix}_R={a.info['rank']}"] += 1
                stat[f"{aprefix}_L={min(a.arg.mention.wlen, 99):02d}"] += 1
                arg_counts[role] += 1
                # ----
                all_overlap_status = set()
                for a2 in e.args:
                    if a is not a2:
                        one_status = overlap_status(a.arg.mention, a2.arg.mention)
                        if one_status != "NoOverlap":
                            all_overlap_status.add(one_status)
                stat[f"{aprefix}_Overlap={'|'.join(sorted(all_overlap_status))}"] += 1
            # repeat args
            for a_role, a_count in arg_counts.items():
                # assert a_role not in iarg_counts, "Cannot be both implicit and explicit!"
                stat[f"{aprefix}_EIRepeat={a_role in iarg_counts}"] += 1
                if a_count>=2:
                    stat[f"{aprefix}_RepeatRole"] += a_count
                else:
                    stat[f"{aprefix}_RepeatRole=NO"] += a_count
            if any(z>=2 for z in arg_counts.values()):
                stat[f"evt_{aprefix}_RepeatRole={max(arg_counts.values())}"] += 1
            else:
                stat[f"evt_{aprefix}_RepeatRole=NO"] += 1
    # ====
    return stat

def deal_sents(sents: List[Sent], print_vocab: bool):
    frame_vocab = SimpleVocab.build_empty()
    fe_vocab = SimpleVocab.build_empty()
    word_vocab = SimpleVocab.build_empty()
    # -----
    for sent in sents:
        word_vocab.feed_iter(sent.seq_word.vals)
        for evt in sent.events:
            frame_vocab.feed_one(evt.type)
            fe_vocab.feed_iter([a.role for a in evt.args])
    frame_vocab.build_sort()
    fe_vocab.build_sort()
    word_vocab.build_sort()
    # =====
    if print_vocab:
        zlog(f"Vocab info of Frame:\n" + frame_vocab.get_info_table().to_string())
        zlog(f"Vocab info of FE:\n" + fe_vocab.get_info_table().to_string())
        zlog(f"Vocab info of Word:\n" + word_vocab.get_info_table().to_string())
        zlog("#=====")

def main_docs(input_file: str, print_vocab=0):
    print_vocab = int(print_vocab)
    reader = DataReader(LineStreamer(input_file), "zjson_doc")
    all_stat = Counter()
    sents = []
    for one_doc in reader:
        all_stat["doc"] += 1
        sents.extend(one_doc.sents)
        one_doc_stat = Counter()
        for one_sent in one_doc.sents:
            one_sent_stat = _stat_sent(one_sent)
            one_doc_stat += one_sent_stat
        all_stat += one_doc_stat
        rate_ann = one_doc_stat["evt_Status=MANUAL"] / one_doc_stat["tok_O"]
        zlog(f"Read one doc={one_doc.id}:\n{one_doc_stat}")
        if rate_ann <= 0.2:
            zlog(f"warning: might be not fully annotated, evt={one_doc_stat['evt']}/{one_doc_stat['evt_Status=MANUAL']}/{one_doc_stat['evt_Status=UNANN']}, rate_ann={rate_ann}")
    deal_sents(sents, print_vocab)
    OtherHelper.printd(all_stat)
    zlog(f"all_stat={all_stat}")

def main_sents(input_file: str, print_vocab=0):
    print_vocab = int(print_vocab)
    reader = DataReader(LineStreamer(input_file), "zjson_sent")
    sents = list(reader)
    deal_sents(sents, print_vocab)
    all_stat = Counter()
    for sent in sents:
        all_stat += _stat_sent(sent)
    OtherHelper.printd(all_stat)
    zlog(f"all_stat={all_stat}")

def main_frames(input_file: str, print_vocab=0):
    print_vocab = int(print_vocab)
    frames = default_json_serializer.from_file(input_file)
    # build vocabs and collect stat
    all_stat = Counter()
    fe_vocab = SimpleVocab.build_empty()
    lex_vocab = SimpleVocab.build_empty()
    for f, v in frames.items():
        all_stat["Frame"] += 1
        for fe in v["FE"]:
            fe_vocab.feed_one(fe["name"])
            all_stat["FE"] += 1
            all_stat[f"FE_semTypes={len(fe['semTypes'])}"] += 1
            all_stat[f"FE_Require={len(fe['requiresFE'])}"] += 1
            all_stat[f"FE_Exclude={len(fe['excludesFE'])}"] += 1
            all_stat[f"FE_Core={fe['coreType']}"] += 1
        for lu in v["lexUnit"]:
            lex_vocab.feed_one(lu["name"].split(".")[0])
            all_stat["LexUnit"] += 1
    fe_vocab.build_sort()
    lex_vocab.build_sort()
    # =====
    if print_vocab:
        zlog(f"#Frame={len(frames)}, #AllFE={sum(len(v['FE']) for v in frames.values())}, "
             f"#ALLLexUnit={sum(len(v['lexUnit']) for v in frames.values())}")
        zlog(f"Vocab info of FE:\n" + fe_vocab.get_info_table().to_string())
        zlog(f"Vocab info of lexUnit:\n" + lex_vocab.get_info_table().to_string())
    # --
    OtherHelper.printd(all_stat)
    zlog(f"all_stat={all_stat}")

# =====
if __name__ == '__main__':
    globals()[f"main_{sys.argv[1]}"](*sys.argv[2:])

# run
# todo(note): 1) UNANN gets no FEs, can be ignored in training; 2) NT mostly is fine unless discontinous span
"""
# examples
PYTHONPATH=../../src/ python3 -m pdb fn_stat.py docs fn15/fulltext.dev.json
# run
for ff in fn15 fn17; do
export _CACHE_FRAMES="$ff/frames.json"
PYTHONPATH=../../src/ python3 fn_stat.py frames $ff/frames.json 1 |& tee logs/_${ff}_frames.log
PYTHONPATH=../../src/ python3 fn_stat.py sents $ff/exemplars.json 1 |& tee logs/_${ff}_exemplars.log
PYTHONPATH=../../src/ python3 fn_stat.py sents $ff/exemplars.filtered.json 1 |& tee logs/_${ff}_exemplars_filtered.log
for ff2 in train dev test test1; do
PYTHONPATH=../../src/ python3 fn_stat.py docs $ff/fulltext.$ff2.json 1 |& tee logs/_${ff}_${ff2}.log
done
unset _CACHE_FRAMES
done
"""
