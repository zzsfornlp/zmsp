#

# prep conll NER data
# note: from UTF8 txt inputs!

from collections import Counter
from mspx.data.inst import Sent, Doc
from mspx.data.rw import ReaderGetterConf, WriterGetterConf, FileStreamer
from mspx.data.vocab import SeqSchemeHelperStr
from mspx.utils import zlog, zopen, Conf, Random, zglobs, init_everything

class MainConf(Conf):
    def __init__(self):
        self.input_path = []
        self.W = WriterGetterConf()

# --
def yield_sent(file: str):
    streamer = FileStreamer(file, mode='mline')
    docstart = True  # at the very start
    bio_helper, io_helper = SeqSchemeHelperStr("BIO"), SeqSchemeHelperStr("IO")
    for mline in streamer:
        lines = [z.strip() for z in mline.strip().split("\n")]
        fields = [z.split() for z in lines]
        assert len(fields) > 0 and len(fields[0][0]) > 0
        if fields[0][0] == "-DOCSTART-":
            assert fields[0][-1] == 'O'
            docstart = True
            fields = fields[1:]
            if len(fields) == 0:
                continue  # stand-alone marker!
        # process the sentence
        tokens = [z[0] for z in fields]
        sent = Sent(tokens, make_singleton_doc=True)
        if docstart:
            sent.info['docstart'] = True
            docstart = False
        tags = [z[-1] for z in fields]
        helper = bio_helper if any(z.startswith("B-") for z in tags) else io_helper
        spans = helper.tags2spans(tags)
        for _widx, _wlen, _label in spans:
            sent.make_frame(_widx, _wlen, _label, 'ef')  # make a new entity
        yield sent
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
        for sent in yield_sent(f):
            all_insts.append(sent.doc)
            cc['sent'] += 1
            cc['tok'] += len(sent)
            for ef in sent.yield_frames(cates='ef'):
                cc['ef'] += 1
                cc[f'ef_{ef.label}'] += 1
    # --
    if conf.W.has_path():
        with conf.W.get_writer() as writer:
            writer.write_insts(all_insts)
    zlog(f"Read from {input_paths} to {conf.W.output_path}: {cc}", timed=True)
    # --

# python3 -m mspx.scripts.data.ner.prep_conll input_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
