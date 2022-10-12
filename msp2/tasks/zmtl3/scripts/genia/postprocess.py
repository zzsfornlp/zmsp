#

# postprocess outputs from srl model (adapting to the evts & proteins)

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
# DOWN_LABS = {'conj', 'flat', 'compound', 'list', 'dep', 'reparandum', 'goeswith', 'amod', 'nmod', 'appos', 'nummod'}
# DOWN_LABS = {'conj', 'flat', 'compound', 'amod', 'appos'}
DOWN_LABS = {'conj', 'compound'}
UP_LABS = {}
def search_for_args(tree_dep, widx: int):
    dep_chs = tree_dep.chs_lists
    dep_heads = tree_dep.seq_head.vals
    dep_labs = [z.split(":")[0] for z in tree_dep.seq_label.vals]
    hit_widxes = set()
    # --
    def _search(_cidx):
        if _cidx in hit_widxes:
            return
        hit_widxes.add(_cidx)
        yield _cidx
        for _ch in dep_chs[_cidx+1]:
            if dep_labs[_ch] in DOWN_LABS:
                yield from _search(_ch)
    # --
    cur_widx = widx
    while cur_widx >= 0:
        yield from _search(cur_widx)
        if dep_labs[cur_widx] in UP_LABS:
            cur_widx = dep_heads[cur_widx] - 1
        else:
            break
    # --

def get_all_pars(evt_pars, widx: int):
    ret = set()
    def _run(_i):
        ret.add(_i)
        for _i2 in evt_pars[_i]:
            _run(_i2)
    _run(widx)
    return ret

def process(doc, cc):
    # RMAP = {'Theme1': 'Theme', 'Theme2': 'Theme'}  # note: not here!
    RMAP = {}
    cc['doc'] += 1
    for sent in yield_sents(doc):
        cc['sent'] += 1
        tree_dep = sent.tree_dep
        slen = len(sent)
        # --
        widx2evt = [[] for _ in range(slen)]
        widx2ef = [None] * slen
        for ef in sent.entity_fillers:
            if ef.label == 'Protein':
                cc['ef'] += 1
                _widx = ef.mention.shead_widx
                assert widx2ef[_widx] is None
                widx2ef[_widx] = ef  # put protein!
        for evt in sent.events:
            cc['evt'] += 1
            _widx = evt.mention.shead_widx
            widx2evt[_widx].append(evt)
        # change the args
        evt_pars = [[] for _ in range(slen)]  # record evt-evt parents
        for evt in sent.events:
            evt_widx = evt.mention.shead_widx
            widx2arg = [None] * slen  # (item, role)
            # find args
            for arg in sorted(evt.args, key=(lambda x: x.score), reverse=True):
                arg.delete_self()  # no matter what, just delete it!
                _widx = arg.mention.shead_widx
                _role = arg.label
                if _role != 'Agent':  # ignore the dummy one!
                    allow_evt = ('regulation' in evt.label.lower())
                    _cands = [z for z in search_for_args(tree_dep, _widx) if widx2arg[z] is None and z!=evt_widx]
                    for _twidx in _cands:
                        _par_set = get_all_pars(evt_pars, evt_widx)
                        if allow_evt and _twidx not in _par_set and len(widx2evt[_twidx])>0:  # allow to put an evt
                            widx2arg[_twidx] = (widx2evt[_twidx][0], _role)
                            evt_pars[_twidx].append(evt_widx)
                            break  # end if find an evt!
                        elif widx2ef[_twidx] is not None:  # find the protein
                            widx2arg[_twidx] = (widx2ef[_twidx], _role)
            # add args
            for one in widx2arg:
                if one is not None:
                    cc['arg'] += 1
                    evt.add_arg(one[0], role=RMAP.get(one[1], one[1]))
        # remove unused efs
        for ef in list(sent.entity_fillers):
            if ef.label != 'Protein':
                sent.delete_frame(ef, 'ef')
        # --

# --
def main(input_file: str, output_file: str):
    cc = Counter()
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    with WriterGetterConf().get_writer(output_path=output_file) as writer:
        for inst in reader:
            process(inst, cc)
            writer.write_inst(inst)
    # --
    zlog(f"Read from {input_file} to {output_file}: {cc}")

# --
# python3 -m msp2.tasks.zmtl3.scripts.genia.postprocess IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
