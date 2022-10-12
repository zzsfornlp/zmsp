#

# read evt's lemma and filter frame data

from typing import Dict
from collections import Counter, defaultdict
from msp2.utils import Conf, zlog, zglob1, init_everything, default_pickle_serializer
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.data.vocab import ZFrameCollection, ZFrameCollectionHelper
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        self.input_evt = ""  # input evt
        self.frame_file = ""  # from lemma to frame, if not, then directly use lemma!
        self.input_data = ""  # data file to filter
        self.output_data = ""
        # --
        self.lemma_filter_count = 2  # filter by: lemma[evt][_lemma] >= this
        self.ignore_lemmas = ['be', 'have', 'do']  # ignore lemmas
        self.ignore_frames = []  # ignore frames
        # --

# --
def stat_counts(counts: Dict[str, Counter], name: str):
    zlog(f"#-- Types & {name}")
    for k in sorted(counts.keys(), key=(lambda x: sum(counts[x].values())), reverse=True):
        zlog(f"'{k}': {counts[k]},")
    lemma_set = set(kk for v in counts.values() for kk in v.keys())
    zlog(f"{name}-stat: evt={len(counts)},itemsA={sum(len(v) for v in counts.values())},itemsS={len(lemma_set)}")
    return lemma_set

# --
def main(*args):
    conf = MainConf()
    conf: MainConf = init_everything(conf, args)
    _ignore_lemmas = set(conf.ignore_lemmas)
    _ignore_frames = set(conf.ignore_frames)
    # --
    # step 1: read evt data and get lemma
    cc = Counter()
    lemma_counts = defaultdict(Counter)
    for inst in ReaderGetterConf().get_reader(input_path=conf.input_evt):
        cc['num_inst'] += 1
        set_ee_heads([inst])
        for sent in yield_sents(inst):
            cc['num_sent'] += 1
            for evt in sent.events:
                cc['num_evt'] += 1
                evt_type = evt.type
                lemma = sent.seq_lemma.vals[evt.mention.shead_widx]
                if lemma is None:
                    continue
                lemma = lemma.lower()
                if lemma in _ignore_lemmas:
                    continue
                lemma_counts[evt_type][lemma] += 1
    # --
    zlog(f"Read from {conf.input_evt}: {cc}")
    lemma_set = stat_counts(lemma_counts, "Lemma")
    if conf.lemma_filter_count > 1:
        for vv in lemma_counts.values():
            for kk in list(vv.keys()):
                if vv[kk] < conf.lemma_filter_count:
                    del vv[kk]
        zlog(f"Filter by {conf.lemma_filter_count}")
        lemma_set = stat_counts(lemma_counts, "Lemma")
    # --
    # step 2: read frames and get hit frames
    frame_counts = defaultdict(Counter)
    if conf.frame_file:
        fc_file = zglob1(conf.frame_file, check_iter=10)
        _fc: ZFrameCollection = default_pickle_serializer.from_file(fc_file)
        zlog(f"All frames = {len(_fc.frames)}")
        _lu_map = ZFrameCollectionHelper.build_lu_map(_fc)
        for kk, vv in lemma_counts.items():
            for _lu, _count in vv.items():
                hit_frames = _lu_map.get(_lu, [])
                for _frame in hit_frames:
                    if _frame not in _ignore_frames:
                        frame_counts[kk][_frame] += 1
    frame_set = stat_counts(frame_counts, "Frame")
    if not conf.frame_file:
        frame_set = None
    # --
    # step 3: read data and filter
    if conf.input_data:
        data_insts = list(ReaderGetterConf().get_reader(input_path=conf.input_data))
        cc2 = Counter()
        final_insts = []
        for inst in data_insts:
            has_valid = False
            cc2['num_inst'] += 1
            set_ee_heads([inst])
            for sent in yield_sents(inst):
                cc2['num_sent'] += 1
                for evt in list(sent.events):  # remember copy!
                    _hit = False
                    cc2['num_evt'] += 1
                    evt_type = evt.type
                    if frame_set is not None:  # by frame
                        if evt_type in frame_set:
                            _hit = True
                    else:  # by lemma
                        lemma = sent.seq_lemma.vals[evt.mention.shead_widx]
                        if lemma is not None and lemma.lower() in lemma_set:
                            _hit = True
                    if _hit:
                        cc2['num_evtV'] += 1
                        has_valid = True
                    else:  # note: delete inplace
                        sent.delete_frame(evt, 'evt')
            if has_valid:
                cc2['num_instV'] += 1
                final_insts.append(inst)
        # --
        zlog(f"Read from {conf.input_data}: {cc2}")
        if conf.output_data:
            with WriterGetterConf().get_writer(output_path=conf.output_data) as writer:
                writer.write_insts(final_insts)
    # --

# --
# PYTHONPATH=../src/ python3 filter_data.py lemma_filter_count:? input_evt:?? frame_file:?? input_data:?? output_data:??
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
