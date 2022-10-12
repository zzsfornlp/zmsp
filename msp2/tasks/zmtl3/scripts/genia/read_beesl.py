#

# simply read genia data from the beesl format

import os
import re
import math
import sys
from itertools import chain
from collections import Counter, defaultdict, OrderedDict
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, Frame, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob

# --
def _yield_lines(fd, cc):
    cur_lines = []
    for line in chain(fd, ['']):
        if line.strip() == "":
            if len(cur_lines) > 0:
                yield cur_lines
            cur_lines = []
        else:
            cc['line'] += 1
            cur_lines.append(line)
# --
def _yield_sents(fd, cc, repl_prot):
    for cur_lines in _yield_lines(fd, cc):
        assert cur_lines[0].startswith("# doc_id = ")
        _docid = cur_lines[0].split()[-1]
        cc['sent'] += 1
        # --
        all_fields = [line.strip().split('\t') for line in cur_lines[1:]]
        words = [z[0] for z in all_fields]
        ef_ids = [z[2] for z in all_fields]
        efs = [z[3] for z in all_fields]
        if any(len(z)<8 for z in all_fields):  # split them
            labs0, labs1 = [], []
            for z in all_fields:
                if z[6] == 'O':
                    _f0 = _f1 = 'O'
                else:
                    _subs = z[6].split("$")
                    _f0s, _f1s = [z2.split("|", 1)[0] for z2 in _subs], [z2.split("|", 1)[1] for z2 in _subs]
                    assert all(z2==_f0s[0] for z2 in _f0s)
                    _f0 = _f0s[0]
                    _f1 = '$'.join(['B-'+z2 for z2 in _f1s if z2!='O'])
                labs0.append(_f0)
                labs1.append(_f1 if _f1 else 'O')
        else:
            labs0, labs1 = [z[6] for z in all_fields], [z[7] for z in all_fields]
        # --
        sent = Sent.create([('P' if w=='$PROTEIN$' else w) for w in words])
        sent.info['orig_words'] = [z[0] for z in all_fields]
        sent.info['char_posi_info'] = [z[1] for z in all_fields]
        # build evts
        slen = len(words)
        widx2ef = [None] * slen
        widx2evt = [[] for _ in range(slen)]
        for widx in range(slen):
            _lab = labs0[widx]
            assert '|' not in _lab
            if labs0[widx] != 'O':
                assert _lab.startswith("B-")
                _lab = _lab[2:]
                if _lab != 'Protein':
                    cc[f'evt_trig_{len(_lab.split("////"))}'] += 1
                    for _lab2 in _lab.split("////"):
                        evt = sent.make_event(widx, 1, type=_lab2)
                        widx2evt[widx].append(evt)
                        cc['evt'] += 1
        # build efs of proteins
        for widx in range(slen):
            if efs[widx]=='[ENT]Protein':
                assert ef_ids[widx]!='O'
                if labs0[widx] not in ('B-Protein', 'O'):
                    zwarn(f"Strange protein: {labs0[widx]}")
                    cc['ef_strange'] += 1
                ef = sent.make_entity_filler(widx, 1, type='Protein', id=ef_ids[widx])
                widx2ef[widx] = ef
                cc['ef'] += 1
            else:
                assert ef_ids[widx]=='O' and efs[widx]=='[ENT]-'
                if labs0[widx] == 'B-Protein':
                    zwarn(f"Strange protein: {labs0[widx]}")
                    cc['ef_strange'] += 1
        # build args
        for widx in range(slen):
            _lab = labs1[widx]
            if _lab != 'O':
                if widx2ef[widx] is None:
                    cc[f'arg_N={len(widx2evt[widx])}'] += 1
                    if len(widx2evt[widx]) == 0:
                        zwarn(f"Skip no item edge: {_lab}")
                        continue
                    item = widx2evt[widx][0]  # simply put the first evt!
                else:
                    item = widx2ef[widx]
                assert item is not None
                # find the pred!
                for _lab2 in _lab.split("$"):
                    _role, _etype, _dist = _lab2.split("|")
                    assert _role.startswith("B-")
                    _role, _dist = _role[2:], int(_dist)
                    cands = [z for z in sum(widx2evt[widx+1:],[]) if z is not None and z.label==_etype] \
                        if _dist>0 else [z for z in sum(widx2evt[:widx],[]) if z is not None and z.label==_etype]
                    try:
                        pred = cands[(_dist-1) if _dist>0 else _dist]
                        pred.add_arg(item, role=_role)
                        cc['arg'] += 1
                    except:
                        zwarn(f"Cannot find arg: {_lab2}")
                        cc['arg_strange'] += 1
        # --
        if repl_prot:  # replace Protein with placeholder
            sent.build_words([(repl_prot if ef is not None else z) for z,ef in zip(sent.seq_word.vals, widx2ef)])
        # --
        yield sent, _docid
# --
def _yield_docs(fd, cc, repl_prot):
    cur_sents = []
    for sent, _docid in chain(_yield_sents(fd, cc, repl_prot), [(None, None)]):
        if len(cur_sents) > 0 and _docid != cur_sents[-1][1]:
            # make a new doc
            doc = Doc.create([z[0] for z in cur_sents], id=cur_sents[-1][1])
            cc['doc'] += 1
            yield doc
            # --
            cur_sents = []
        if sent is not None:
            cur_sents.append((sent, _docid))
# --

def main(input_file: str, output_file: str, repl_prot: str = ''):
    cc = Counter()
    # --
    with zopen(input_file) as fd:
        docs = list(_yield_docs(fd, cc, repl_prot))
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(docs)
    zlog(f"Read {len(docs)} from {input_file} to {output_file}: {cc}")
# --

# --
# python3 -m msp2.tasks.zmtl3.scripts.genia.read_beesl IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
