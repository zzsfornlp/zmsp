#

# specifically read CoNLL-U
# -- including reading meta info!

from collections import Counter
from mspx.data.inst import Sent, Doc
from mspx.data.rw import ReaderGetterConf, WriterGetterConf, FileStreamer
from mspx.data.vocab import SeqSchemeHelperStr
from mspx.utils import zlog, zopen, Conf, Random, zglobs, init_everything, zwarn

class MainConf(Conf):
    def __init__(self):
        self.input_path = []
        self.W = WriterGetterConf()
        self.store_meta_keys = ['newdoc id', 'sent_id']
        self.store_in_doc = True  # store in full docs?
        self.deplab_level = 100  # should be enough!
        self.no_tree = False  # no adding dep-tree!
        # --

FIX_TYPOS = {"oprhan": "orphan"}

def yield_docs(file: str, conf: MainConf):
    store_meta_keys = set(conf.store_meta_keys)
    deplab_level = conf.deplab_level
    store_in_doc = conf.store_in_doc
    # --
    _newdoc_key = 'newdoc id'
    _sentid_key = 'sent_id'
    streamer = FileStreamer(file, mode='mline')
    curr_doc_id, curr_sents = None, []
    for mline in streamer:
        lines = [z.strip() for z in mline.strip().split("\n")]
        meta_lines = [z for z in lines if z.startswith("#")]
        content_lines = [z for z in lines if not z.startswith('#')]
        # read meta
        meta_info = {}
        for _line in meta_lines:
            _line = _line.strip('#').strip()
            if ' = ' in _line:
                k, v = [z.strip() for z in _line.split(" = ", 1)]
                if k in store_meta_keys:
                    meta_info[k] = v
        # read content
        fields0 = [z.split('\t') for z in content_lines]
        fields = [z for z in fields0 if str.isdigit(z[0])]
        assert [int(z[0]) for z in fields] == list(range(1, len(fields)+1))
        seq_word = [z[1] for z in fields]
        seq_lemma = [z[3] for z in fields]
        seq_upos = [z[3] for z in fields]
        seq_head = [int(z[6]) for z in fields]
        if deplab_level is not None:
            seq_label = [":".join(z[7].split(":")[:deplab_level]) for z in fields]
        else:
            seq_label = [z[7] for z in fields]
        # --
        # fix typo
        typo_counts = Counter()
        fixed_seq_label = []
        for z in seq_label:
            for x, y in FIX_TYPOS.items():
                if x in z:
                    typo_counts[x] += 1
                    z = z.replace(x, y)
            fixed_seq_label.append(z)
        if len(typo_counts) > 0:
            seq_label = fixed_seq_label
            zwarn(f"Fix typos: {typo_counts}")
        # --
        sent = Sent(seq_word)
        if meta_info:
            sent.info.update(meta_info)
        sent.build_lemmas(seq_lemma)
        sent.build_uposes(seq_upos)
        if not conf.no_tree:
            sent.build_dep_tree(seq_head, seq_label)
        # --
        if store_in_doc:
            if _newdoc_key in meta_info:  # new doc!
                if len(curr_sents) > 0:
                    yield Doc(curr_sents, id=curr_doc_id)
                    curr_doc_id, curr_sents = None, []
                curr_doc_id = meta_info[_newdoc_key]
            assert meta_info[_sentid_key].startswith(curr_doc_id)
            curr_sents.append(sent)
        else:
            ret = sent.make_singleton_doc()
            if _sentid_key in meta_info:
                ret.set_id(meta_info[_sentid_key])
            yield ret
    # --
    if len(curr_sents) > 0:
        yield Doc(curr_sents, id=curr_doc_id)
    # --

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    # --
    input_paths = zglobs(conf.input_path)
    all_insts = []
    for f in input_paths:
        cc['file'] += 1
        for doc in yield_docs(f, conf):
            all_insts.append(doc)
            cc['doc'] += 1
            cc['sent'] += len(doc.sents)
            cc['tok'] += sum(len(z) for z in doc.sents)
    # --
    if conf.W.has_path():
        with conf.W.get_writer() as writer:
            writer.write_insts(all_insts)
    zlog(f"Read from {input_paths} to {conf.W.output_path}: {cc}", timed=True)
    # --

# python3 -m mspx.scripts.data.ud.read_conllu input_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
