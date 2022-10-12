#

# --
# note: adapted from https://github.com/CogComp/SRL-English/blob/main/preprocess_nombank/preprocess_nombank.py
# --

import nltk.corpus.reader.nombank as nombank_reader
import nltk.corpus
from nltk.data import PathPointer, FileSystemPathPointer
from nltk.tree import Tree

import os
import sys
import copy
import glob
from collections import Counter, OrderedDict

from msp2.data.inst import Doc, Sent
from msp2.utils import zlog, zwarn, OtherHelper
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# =====

def read_wsj(wsj_dir: str, cc):
    docs = OrderedDict()
    for wset in range(25):
    # for wset in [2]:
        wset_dir = os.path.join(wsj_dir, 'PARSED', 'MRG', 'WSJ', f"{wset:02d}")
        files = sorted(os.listdir(wset_dir))
        for filename in files:
            assert filename.startswith(f"WSJ_{wset:02d}") and filename.endswith(".MRG")
            doc_id = filename.split(".", 1)[0].lower()
            # --
            doc_sents = []
            cc['doc'] += 1
            reader = nltk.corpus.BracketParseCorpusReader(wset_dir, filename)
            trees = reader.parsed_sents()
            for _tree in trees:
                cc['sent'] += 1
                new_sent = Sent.create([])  # currently simply store the tree
                new_sent._tree = _tree
                new_sent._frames = []
                doc_sents.append(new_sent)
            # --
            new_doc = Doc.create(doc_sents, id=doc_id)
            assert doc_id not in docs
            docs[doc_id] = new_doc
    return docs

# get a frame
def get_span_from_tp(tree, tp):
    assert isinstance(tp, nombank_reader.NombankTreePointer)
    all_leaves = tree.leaves()
    # --
    arg_tree = tp.select(tree)
    start_idx = tp.wordnum
    arg_leaves = arg_tree.leaves()
    # locate this argument
    diff = 0
    matched = False
    while start_idx - diff >= 0:
        if all_leaves[(start_idx-diff):(start_idx-diff+len(arg_leaves))] == arg_leaves:
            matched = True
            break
        diff += 1
    assert matched
    # more span?
    if diff > 0:
        zwarn(f"Extra span: {arg_leaves[:diff]} ||| {arg_leaves[diff:]}")
    return start_idx, len(arg_leaves)-diff

def get_frame(tree, inst, cc):
    all_args = []
    for arg in inst.arguments:
        cc['g_arg'] += 1
        h_idx = [i for i in range(len(arg[1])) if arg[1].startswith('-H', i)]  # hyps
        arg_label = arg[1] if len(h_idx)==0 else arg[1][:h_idx[0]]
        if '-' in arg_label and (str.isdigit(arg_label[3]) or arg_label.startswith("Support")):
            arg_label, suffix = arg_label.split("-", 1)
            cc[f'g_arg?_{suffix}'] += 1
        cc['g_arg_H'] += (len(h_idx) > 0)
        # --
        if isinstance(arg[0], nombank_reader.NombankTreePointer):  # plain span
            cc['g_arg_S1'] += 1
            _posi = get_span_from_tp(tree, arg[0])
            all_args.append({'posi': _posi, 'label': arg_label, 'type': 'p'})
        else:
            _posi = [get_span_from_tp(tree, p) for p in arg[0].pieces]
            if isinstance(arg[0], nombank_reader.NombankSplitTreePointer):
                _type = 's'
            elif isinstance(arg[0], nombank_reader.NombankChainTreePointer):
                _type = 'c'
            else:
                raise RuntimeError()
            cc[f'g_arg_S2{_type}'] += 1
            all_args.append({'posi': _posi, 'label': arg_label, 'type': _type})
    # --
    # predicate
    cc['g_rel'] += 1
    h_index = inst.predid.find('-H')
    cc['g_rel_H'] += int(h_index >= 0)
    assert len(inst.predicate.select(tree).leaves())==1
    rel = (inst.wordnum, f'{inst.baseform}.{inst.sensenumber}')
    return rel, all_args

# --
def prep_sent(sent, tree, cc):
    valids = []
    # --
    # mark empty nodes and put valids!
    def _visit1(_n):
        _chs = list(_n)
        _label = _n.label()
        if len(_chs)==1 and isinstance(_chs[0], str):  # pre-terminal
            _vv = int(_label != "-NONE-")
            valids.append(_vv)
        else:
            _vv = sum([_visit1(_z) for _z in _chs])
        if _vv <= 0:
            _n.set_label("ZZZZZ")
        return _vv
    # --
    def _visit2(_n):
        _chs = list(_n)
        _label = _n.label()
        if _label == "ZZZZZ": return []  # deleted
        if len(_chs)==1 and isinstance(_chs[0], str):  # pre-terminal
            return [[_chs[0], _label, '*']]  # return the single node!
        else:
            assert all((not isinstance(z, str)) for z in _chs)  # plain non-terminal
            _rets = sum([_visit2(c) for c in _chs], [])
            if len(_rets) > 0:  # empty node?
                _rets[0][-1] = f"({_label}{_rets[0][-1]}"
                _rets[-1][-1] = _rets[-1][-1] + ")"
            return _rets
    # --
    _visit1(tree)
    assert len(tree.leaves()) == len(valids)
    res = _visit2(tree)
    assert sum(valids) == len(res)
    # --
    cc['tok_orig'] += len(tree.leaves())
    cc['tok_valid'] += len(res)
    sent.build_words([z[0] for z in res])
    sent.info['xpos'] = [z[1] for z in res]
    sent.info['parse'] = [z[2] for z in res]
    new_idxes = []
    _i = 0
    for v in valids:
        new_idxes.append(_i if v else -1)
        _i += int(v)
    return new_idxes

def setup_final_docs(docs, cc, arg_cc):
    # --
    # get a new span
    def _new_span(_o2n, _widx, _wlen):
        _n_idxes = [_o2n[z] for z in range(_widx, _widx+_wlen) if _o2n[z]>=0]
        assert _n_idxes == [_n_idxes[0]+i for i in range(len(_n_idxes))]
        if len(_n_idxes) == 0:
            return None
        else:
            return _n_idxes[0], len(_n_idxes)
    # --
    # try to merge spans
    def _merge_span(_posies):
        _cur_spans = [_posies[0]]
        for _widx, _wlen in _posies[1:]:
            _last0, _last1 = _cur_spans[-1][0], sum(_cur_spans[-1])
            if _widx > _last1:
                _cur_spans.append((_widx, _wlen))
            else:
                assert _widx>_last0 and _widx<=_last1
                if _widx != _last1:
                    zwarn(f"Overlapping spans: {_cur_spans} ({_widx},{_wlen})")
                    # breakpoint()
                _cur_spans[-1] = (_last0, _widx+_wlen-_last0)
        return _cur_spans
    # --
    for doc in docs.values():
        cc['doc'] += 1
        for sent in doc.sents:
            cc['sent'] += 1
            new_idxes = prep_sent(sent, sent._tree, cc)
            # put frames
            for frame_rel, frame_args in sent._frames:
                cc['frame'] += 1
                _evt_widx = new_idxes[frame_rel[0]]
                assert _evt_widx >= 0
                evt = sent.make_event(_evt_widx, 1, type=frame_rel[1])
                for arg_dict in frame_args:
                    cc[f"arg_T{arg_dict['type']}"] += 1
                    if arg_dict['type'] == 'p':  # plain
                        arg_posi = _new_span(new_idxes, *(arg_dict['posi']))
                        if arg_posi is None:
                            cc[f"arg_T{arg_dict['type']}_None"] += 1
                            continue
                        arg_ef = sent.make_entity_filler(arg_posi[0], arg_posi[1], type='UNK')
                        evt.add_arg(arg_ef, role=arg_dict['label'])
                        arg_cc[arg_dict['label']] += 1
                    elif arg_dict['type'] == 's':  # split
                        arg_posies = [_new_span(new_idxes, *zz) for zz in arg_dict['posi']]
                        arg_posies = sorted(set([zz for zz in arg_posies if zz is not None]))
                        merged_posies = _merge_span(arg_posies)
                        if len(merged_posies) is None:
                            cc[f"arg_T{arg_dict['type']}_None"] += 1
                        for _ii, _posi in enumerate(merged_posies):
                            arg_ef = sent.make_entity_filler(_posi[0], _posi[1], type='UNK')
                            _role_label = f"{'' if _ii==0 else 'C-'}{arg_dict['label']}"
                            evt.add_arg(arg_ef, role=_role_label)
                            arg_cc[_role_label] += 1
                    elif arg_dict['type'] == 'c':  # chain
                        _arg_posies = [_new_span(new_idxes, *zz) for zz in arg_dict['posi']]
                        arg_posies = []
                        for a in _arg_posies:
                            if a is not None and a not in arg_posies:
                                arg_posies.append(a)
                        if len(arg_posies) is None:
                            cc[f"arg_T{arg_dict['type']}_None"] += 1
                        if len(arg_posies) > 2:
                            zwarn(f"Long chain: {[sent.seq_word.vals[a[0]:sum(a)] for a in arg_posies]}")
                        # --
                        arg_xposes = [[sent.info['xpos'][z] for z in range(a0,a0+a1)] for a0, a1 in arg_posies]
                        if len(arg_posies) > 1:
                            if not all(('WDT' in _xs or 'WP' in _xs or 'WP$' in _xs or 'WRB' in _xs) for _xs in arg_xposes[1:]) \
                                    and arg_xposes[1:] != [['IN']]:
                                zwarn(f"Strange R-chain: {arg_xposes} {[sent.seq_word.vals[a[0]:sum(a)] for a in arg_posies]}")
                                cc['arg_SRC'] += 1  # strange r-chain?
                        # cannot merge chains
                        # todo(+N): there are some (less than 100) outlier cases, currently just ignored ...
                        for _ii, _posi in enumerate(arg_posies):
                            arg_ef = sent.make_entity_filler(_posi[0], _posi[1], type='UNK')
                            _role_label = f"{'' if _ii==0 else 'R-'}{arg_dict['label']}"
                            evt.add_arg(arg_ef, role=_role_label)
                            arg_cc[_role_label] += 1
                        # --
                    else:
                        raise RuntimeError()
            # --
# --

# --
def main(nombank_dir: str, wsj_dir: str, output_file=''):
    # --
    # first read all wsj
    cc0 = Counter()
    docs = read_wsj(wsj_dir, cc0)
    zlog(f"Read docs from {wsj_dir}: {OtherHelper.printd_str(cc0, sep=' || ')}")
    # --
    # read nombank
    root_pathp = FileSystemPathPointer(nombank_dir)
    reader = nombank_reader.NombankCorpusReader(
        root=root_pathp, nomfile='nombank.1.0', framefiles='.*\.xml', nounsfile='nombank.1.0.words')
    # get the instances
    cc1 = Counter()
    instances = reader.instances()
    # populate the instances
    for inst in instances:
        cc1['frame'] += 1
        _doc_id = inst.fileid.rsplit("/", 1)[-1].split(".", 1)[0]
        if _doc_id not in docs:
            zwarn(f"Doc {_doc_id} not found")
            continue
        _sent = docs[_doc_id].sents[inst.sentnum]
        _frame = get_frame(_sent._tree, inst, cc1)
        _sent._frames.append(_frame)
    zlog(f"Read nb from {nombank_dir}: {OtherHelper.printd_str(cc1, sep=' || ')}")
    # --
    # finally put things together!
    cc2 = Counter()
    arg_cc = Counter()
    setup_final_docs(docs, cc2, arg_cc)
    # write
    zlog(f"OK with {OtherHelper.printd_str(cc2, sep=' || ')}, args=\n=> {arg_cc}")
    if output_file:
        with WriterGetterConf.direct_conf(output_path=output_file).get_writer() as writer:
            writer.write_insts(docs.values())
    # --

# --
# python3 -m msp2.tasks.zmtl3.scripts.nombank.prep_nb nombank1.0 ptb3 OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
