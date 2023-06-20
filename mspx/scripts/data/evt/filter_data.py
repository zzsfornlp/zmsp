#

# filter data

import os
import sys
from collections import OrderedDict, Counter
from mspx.utils import zopen, zlog, zwarn, Conf, init_everything, zglobs
from mspx.data.inst import Doc
from mspx.data.rw import ReaderGetterConf, WriterGetterConf

# --
class MainConf(Conf):
    def __init__(self):
        self.input_path = ""
        self.output_path = ""
        self.min_len = 0  # minimum word-len
        self.min_alpha_ratio = 0.  # minimum alpha-rate
        self.output_ind_sents = False  # output individual sentences?
        # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    # --
    reader = ReaderGetterConf().get_reader(input_path=conf.input_path)
    valid_docs = []
    for doc in reader:
        cc['doc'] += 1
        valid_sents = []
        for sent in doc.sents:
            cc['sent'] += 1
            # --
            _len = len(sent)
            _arate = sum((any(str.isalpha(c) for c in t)) for t in sent.seq_word.vals) / _len
            cc['frame'] += len(sent.get_frames())
            if _len >= conf.min_len and _arate >= conf.min_alpha_ratio:
                valid_sents.append(sent)
                cc['sentV'] += 1
            else:
                cc['sentD'] += 1
                # --
                # delete extra frames
                for frame in sent.get_frames():
                    frame.del_self()
                    cc['frameD'] += 1
                # --
        if len(valid_sents) > 0:
            cc['docV'] += 1
            # --
            # note: keep only valid sents
            if conf.output_ind_sents:
                for one_sent in valid_sents:
                    old_frames = one_sent.get_frames()
                    new_doc = Doc([one_sent])
                    for frame in old_frames:
                        cc['frameK'] += 1
                        new_doc.add_frame(frame)
                    valid_docs.append(new_doc)
            else:
                old_frames = doc.get_frames()
                new_doc = Doc(valid_sents, id=doc.id)
                new_doc.info.update(doc.info)
                for frame in old_frames:
                    cc['frameK'] += 1
                    new_doc.add_frame(frame)
                valid_docs.append(new_doc)
        else:
            cc['docD'] += 1
    # --
    zlog(f"Process {conf.input_path} -> {conf.output_path}: {cc}")
    if conf.output_path:
        with WriterGetterConf().get_writer(output_path=conf.output_path) as writer:
            writer.write_insts(valid_docs)
    # --

# python3 -m mspx.scripts.data.evt.filter_data ...
if __name__ == '__main__':
    main(*sys.argv[1:])
