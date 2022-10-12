#

# read from brat (and tokenize and align mentions)

import os
import sys
from collections import OrderedDict, Counter
from msp2.utils import zopen, zlog, zwarn
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, CharIndexer
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
class StanzaTokenizer:
    def __init__(self, lang='en'):
        self.lang = lang
        import stanza
        self.nlp = stanza.Pipeline(processors='tokenize', lang=lang, use_gpu=False)

    def tokenize(self, input_str: str, offset=0):
        res = self.nlp(input_str)
        rets = []
        for sent in res.sentences:
            tokens, positions = [], []
            for tok in sent.tokens:
                _text, _start_char, _end_char = tok.text, tok._start_char, tok._end_char
                assert _text == input_str[_start_char:_end_char]
                tokens.append(_text)
                positions.append((_start_char+offset, _end_char-_start_char))
            rets.append({"tokens": tokens, "positions": positions})
            # breakpoint()
        return rets

# --
def read_brat(afile: str, tfile: str, doc_id_from_file: str, toker):
    assert os.path.basename(afile).startswith(doc_id_from_file)
    assert os.path.basename(tfile).startswith(doc_id_from_file)
    # read text and anns
    with zopen(tfile) as fd:
        source_str = fd.read()
    if os.path.isfile(afile):
        with zopen(afile) as fd:
            ann_lines = [line.rstrip() for line in fd]
    else:
        zwarn(f"Cannot find {afile}!")
        ann_lines = []
    # mentions
    mentions = OrderedDict()
    for line in ann_lines:
        fields = line.split("\t")
        tag = fields[0]
        if tag.startswith("T"):  # mention
            mtype, mposi = fields[1].split(" ", 1)
            mstart, mend = [int(z) for z in mposi.split(" ")]
            assert tag not in mentions
            mentions[tag] = (tag, mtype, mstart, mend)
    # events
    orig_events = OrderedDict()
    for line in ann_lines:
        fields = line.split("\t")
        tag = fields[0]
        if tag.startswith("E"):  # event
            args = [z.split(":",1) for z in fields[1].split()]
            _mention = mentions[args[0][1]]
            assert _mention[1] == args[0][0]
            assert tag not in orig_events
            orig_events[tag] = (tag, _mention[0], args[1:])
    # relations
    relations = OrderedDict()
    for line in ann_lines:
        fields = line.split("\t")
        tag = fields[0]
        if tag.startswith("R"):  # relation
            rtype, a1, a2 = fields[1].split()
            assert a1.startswith("Arg1:") and a2.startswith("Arg2:")
            a1, a2 = a1[5:], a2[5:]
            assert (a1 in mentions or a1 in orig_events) and (a2 in mentions or a2 in orig_events)
            assert tag not in relations
            relations[tag] = (tag, rtype, a1, a2)
    # --
    # brat to doc
    cc = Counter()
    # tokenize and build doc
    tok_res = toker.tokenize(source_str)
    sents = []
    for one_res in tok_res:
        one_sent = Sent.create(one_res['tokens'])
        one_sent.build_word_positions(one_res['positions'])
        sents.append(one_sent)
    doc = Doc.create(sents, text=source_str, id=doc_id_from_file)
    cc['doc'] += 1
    cc['sent'] += len(sents)
    # put mentions and relations
    char_indexer = CharIndexer.build_from_doc(doc, source_str)
    events = {}  # id -> evt, note: put them as events for easier processings!
    for tag, mtype, mstart, mend in mentions.values():
        cc['evt'] += 1
        cur_posi, cur_err = char_indexer.get_posi(mstart, mend-mstart)
        cc[f'evt_C={cur_err}'] += 1
        if cur_posi is not None:
            cc[f'evtV'] += 1
            evt = doc.sents[cur_posi[0]].make_event(cur_posi[1], cur_posi[2], type=mtype, id=tag)
            events[tag] = evt
    for tag, mtag, args in orig_events.values():
        cc['erel0'] += 1
        if mtag in events:
            events[tag] = events[mtag]  # for later evt-evt-relation!
        for role, rtag in args:
            cc['erel'] += 1
            if mtag in events and rtag in events:
                cc['erelV'] += 1
                e1, e2 = events[mtag], events[rtag]
                e1.add_arg(e2, role=role)  # note: simply from A1 -> A2
    for tag, rtype, a1, a2 in relations.values():
        cc['rel'] += 1
        if a1 in events and a2 in events:
            cc['relV'] += 1
            e1, e2 = events[a1], events[a2]
            e1.add_arg(e2, role=rtype)   # note: simply from A1 -> A2
    # --
    return doc, cc
    # --

# --
def main(input_dir: str, output_file: str):
    cc = Counter()
    docs = []
    toker = StanzaTokenizer()
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.txt'):
            cur_doc_id = file[:-4]
            afile = os.path.join(input_dir, cur_doc_id+".ann")
            tfile = os.path.join(input_dir, cur_doc_id+".txt")
            one_doc, cc2 = read_brat(afile, tfile, cur_doc_id, toker)
            docs.append(one_doc)
            cc += cc2
    zlog(f"Read from {input_dir} to {output_file}: {cc}")
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(docs)
    # --

# python3 -m msp2.tasks.zmtl3.mat.prep.brat2json ...
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# --
# v0307
python3 -m msp2.tasks.zmtl3.mat.prep.brat2json Collection_v0307 mat.v0307.json
# -> Read from Collection_v0307 to mat.v0307.json: Counter({'evt': 1034, 'evtV': 1034, 'evt_C=': 1013, 'rel': 987, 'relV': 987, 'sent': 175, 'doc': 23, 'evt_C=WarnRight': 13, 'evt_C=WarnLeft': 6, 'evt_C=WarnRDot': 2})
# python3 -m msp2.tasks.zmtl3.mat.prep.json2brat mat.v0307.json _tmp/
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:mat.v0307.json output_path:mat.v0307.ud2.json
# split
python3 -m msp2.tasks.zmtl3.mat.prep.split mat.v0307.ud2.json 6 _split_v0307/mat.v0307.
# --
# v0320
python3 -m msp2.tasks.zmtl3.mat.prep.brat2json Collection_v0320 mat.v0320.json
Read from Collection_v0320 to mat.v0320.json: Counter({'evt': 1357, 'evtV': 1357, 'evt_C=': 1330, 'rel': 1299, 'relV': 1299, 'sent': 219, 'doc': 31, 'evt_C=WarnRight': 16, 'evt_C=WarnLeft': 8, 'evt_C=WarnRDot': 3})
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:mat.v0320.json output_path:mat.v0320.ud2.json
python3 -m msp2.tasks.zmtl3.mat.prep.split mat.v0320.ud2.json 5 _split_v0320/mat.v0320.
# --
# v0: join v0417 + v0425
python3 -m msp2.tasks.zmtl3.mat.prep.brat2json Collection_v0425 mat.v0425.json
Read from Collection_v0425 to mat.v0425.json: Counter({'rel': 584, 'relV': 576, 'evt': 566, 'evtV': 565, 'evt_C=': 561, 'sent': 93, 'doc': 13, 'evt_C=WarnRight': 3, 'evt_C=ErrDiffSent': 1, 'evt_C=WarnLeft': 1})
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:mat.v0425.json output_path:mat.v0425.ud2.json
# note: simply add to train
cat _split_v0320/mat.v0320.0.train.json mat.v0425.ud2.json >_split_v0425/mat.v0425.0.train.json
# -> training comparison
v0320: 'evt': 769, 'evt1': 769, 'arg': 740, 'arg1': 740, 'sent': 118, 'sent1': 102
v0425: 'evt': 1334, 'evt1': 1334, 'arg': 1316, 'arg1': 1316, 'sent': 211, 'sent1': 177
# --
# v1: sep
for vv in v0417 v0425; do
python3 -m msp2.tasks.zmtl3.mat.prep.brat2json Collection_${vv} mat.${vv}.json
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:mat.${vv}.json output_path:mat.${vv}.ud2.json
cat _split_v0320/mat.v0320.0.train.json mat.${vv}.ud2.json >_split_v0425/mat.v1_${vv}.0.train.json
done
# Read from Collection_v0417 to mat.v0417.json: Counter({'rel': 338, 'relV': 338, 'evt': 322, 'evtV': 322, 'evt_C=': 319, 'sent': 54, 'doc': 7, 'evt_C=WarnRight': 2, 'evt_C=WarnLeft': 1})
# Read from Collection_v0425 to mat.v0425.json: Counter({'rel': 246, 'evt': 244, 'evtV': 243, 'evt_C=': 242, 'relV': 238, 'sent': 39, 'doc': 6, 'evt_C=ErrDiffSent': 1, 'evt_C=WarnRight': 1})
# --
# mixed0508
for nn in v0307 v0320 v0417 v0425; do unzip Collection_${nn}.zip -d Collection_$nn; done
mv Collection_v0307/Updated_Collection/ Collection_v0307/New_Abstracts
for vv in v0307 v0320 v0417 v0425; do
python3 -m msp2.tasks.zmtl3.mat.prep.brat2json Collection_${vv}/New_Abstracts mat.${vv}.json
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:mat.${vv}.json output_path:mat.${vv}.ud2.json
done
# 23+8+7+6
mkdir -p _split_m0508/
cat mat.v*.ud2.json >_split_m0508/mat.all.ud2.json
python3 -m msp2.tasks.zmtl3.mat.prep.split _split_m0508/mat.all.ud2.json 5 _split_m0508/mat.
# --
# further add v0524 (+13, 57 totally)
unzip Collection_v0524.zip -d Collection_v0524
# ...
mkdir -p _split_m0524/
cat mat.v*.ud2.json >_split_m0524/mat.all.ud2.json
python3 -m msp2.tasks.zmtl3.mat.prep.split _split_m0524/mat.all.ud2.json 10 _split_m0524/mat.
# =====
# v0: join _split_v0320 + v0417+v0425+v0524
mkdir -p _split_v0524
cat _split_v0320/mat.v0320.0.train.json mat.v0417.ud2.json mat.v0425.ud2.json mat.v0524.ud2.json >_split_v0524/mat.v0524.0.train.json
"""
