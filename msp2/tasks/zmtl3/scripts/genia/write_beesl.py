#

# the reverse of "read_beesl"

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
def main(input_file: str, output_file: str):
    cc = Counter()
    RMAP = {'Theme1': 'Theme', 'Theme2': 'Theme'}
    # --
    with zopen(output_file, 'w') as fd:
        reader = ReaderGetterConf().get_reader(input_path=input_file)
        for inst in reader:
            cc['doc'] += 1
            doc_id = inst.id
            for sent in inst.sents:
                cc['sent'] += 1
                slen = len(sent)
                # --
                cols0 = list(sent.seq_word.vals)
                cols1 = sent.info['char_posi_info']
                # --
                widx2ef = [None] * slen
                widx2evt = [[] for _ in range(slen)]
                for ef in sent.entity_fillers:
                    if ef.label != 'Protein':
                        zwarn(f"Get a strange ef: {ef}")
                        ef.set_label('Protein')
                    widx2ef[ef.mention.shead_widx] = ef
                    cc['ef'] += 1
                for evt in sent.events:
                    widx2evt[evt.mention.shead_widx].append(evt)
                    cc['evt'] += 1
                # --
                fd.write(f"# doc_id = {doc_id}\n")
                for widx in range(slen):
                    if len(widx2evt[widx]) == 0:
                        items = [] if widx2ef[widx] is None else [widx2ef[widx]]
                    else:
                        items = widx2evt[widx]
                    if len(items) == 0:
                        c6 = c7 = 'O'
                    else:
                        c6 = "B-" + "////".join([z.label for z in items])
                        c7s = []
                        for _item in items:
                            for arglink in _item.as_args:
                                _pred = arglink.main
                                _pwidx = _pred.mention.shead_widx
                                if _pwidx < widx:
                                    _extra_count = sum(z2.label==_pred.label for z1 in widx2evt[_pwidx+1:widx] for z2 in z1)
                                    _dist = str(-1 - _extra_count)
                                else:
                                    _extra_count = sum(z2.label==_pred.label for z1 in widx2evt[widx+1:_pwidx] for z2 in z1)
                                    _dist = f"+{1+_extra_count}"
                                _role = arglink.label
                                _role = RMAP.get(_role, _role)
                                _c7 = f"B-{_role}|{_pred.label}|{_dist}"
                                c7s.append(_c7)
                        c7 = "O" if len(c7s)==0 else '$'.join(c7s)
                    # --
                    line = '\t'.join([
                        "$PROTEIN$" if widx2ef[widx] is not None else cols0[widx],
                        cols1[widx],
                        widx2ef[widx].id if widx2ef[widx] is not None else 'O',
                        "[ENT]Protein" if widx2ef[widx] is not None else '[ENT]-',
                        f"[POS]{sent.seq_upos.vals[widx]}",
                        f"[DEP]{sent.tree_dep.seq_label.vals[widx]}",
                        c6,
                        c7,
                    ])
                    fd.write(line + "\n")
                fd.write("\n")
    # --
    zlog(f"Read from {input_file} to {output_file}: {cc}")
# --

# --
# python3 -m msp2.tasks.zmtl3.scripts.genia.write_beesl IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
