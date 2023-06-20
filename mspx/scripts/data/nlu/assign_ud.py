#

# assign UD syntax information

from collections import Counter
from mspx.data.inst import Sent, Doc, NLTKTokenizer, yield_sents
from mspx.data.rw import ReaderGetterConf, WriterGetterConf, FileStreamer
from mspx.data.vocab import SeqSchemeHelperStr
from mspx.utils import zlog, zwarn, Conf, Random, zglobs, init_everything, default_json_serializer

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        self.ud_paths = []

def get_key(sent):
    return tuple(["".join([c for c in t if str.isalnum(c)]).lower() for t in sent.seq_word.vals])

def do_assign(sent, sent_ref):
    assert len(sent) == len(sent_ref)
    if [z.lower() for z in sent.seq_word.vals] != [z.lower() for z in sent_ref.seq_word.vals]:
        zwarn(f"Some mismatch: {sent.seq_word.vals} vs {sent_ref.seq_word.vals}")
    sent.build_uposes(sent_ref.seq_lemma.vals)
    sent.build_dep_tree(sent_ref.tree_dep.seq_head.vals, sent_ref.tree_dep.get_labels(level=1))  # L1!

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    # --
    # read UD paths
    all_maps = []
    for one_ud_path in conf.ud_paths:
        one_map = {}
        for inst in conf.R.get_reader(input_path=one_ud_path):
            for sent in yield_sents(inst):
                _key = get_key(sent)
                if _key in one_map:
                    # zwarn(f"Repeat key: {_key}")
                    pass
                else:
                    one_map[_key] = sent
        all_maps.append(one_map)
    # --
    # assign them!
    cc = Counter()
    all_insts = []
    for inst in conf.R.get_reader():
        for sent in yield_sents(inst):
            cc['sent'] += 1
            _key = get_key(sent)
            for one_ii, one_map in enumerate(all_maps):
                if _key in one_map:
                    do_assign(sent, one_map[_key])
                    cc[f'sent_A{one_ii}'] += 1
                    break
                zlog(f"Notfound[{one_ii}]: {sent.seq_word}")
        all_insts.append(inst)
    # --
    if conf.W.has_path():
        with conf.W.get_writer() as writer:
            writer.write_insts(all_insts)
    zlog(f"Read from {conf.R} to {conf.W}: {cc}", timed=True)
    # --

# python3 -m mspx.scripts.data.nlu.assign_ud
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
