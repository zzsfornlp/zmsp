#

# prepare amr

import sys
import os
from collections import Counter, OrderedDict
from msp2.data.inst import Doc, Sent
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zopen, zlog, OtherHelper, Conf, init_everything, zwarn, zglob1z

# --
def make_item(sent, v, is_ef: bool):
    idxes = v[2]
    if idxes is None:  # no aligns
        return None, 'NoAlign'
    idxes = sorted(idxes)
    widx, wlen = idxes[0], idxes[-1]-idxes[0]+1
    if wlen != len(idxes):
        ret_info = "Disc"
    else:
        ret_info = 'Ok'
    if is_ef:
        ret = sent.make_entity_filler(widx, wlen, type=v[1], id=v[0])
    else:
        ret = sent.make_event(widx, wlen, type=v[1], id=f"E{v[0]}")
    return ret, ret_info

def amr2sent(graph, onto):
    from penman import surface
    cc = Counter()
    amr_role_map = {
        'source': 'ARGM-DIR', 'destination': 'ARGM-DIR', 'path': 'ARGM-DIR', 'direction': 'ARGM-DIR',
        'beneficiary': 'ARGM-GOL', 'accompanier': 'ARGM-COM',
        'instrument': 'instrument',
        'manner': 'ARGM-MNR', 'purpose': 'ARGM-PRP', 'cause': 'ARGM-CAU', 'extent': 'ARGM-EXT',
        'location': 'ARGM-LOC', 'time': 'ARGM-TMP',
    }
    # --
    # create sentence
    tokens = [z.replace('@-@', '-') for z in graph.metadata['tok'].split()]
    sent = Sent.create(tokens)
    sent.info['orig_id'] = graph.metadata['id']
    # --
    # gather information
    info = {}
    alignments = surface.alignments(graph)
    all_inst_ids = []
    for inst in graph.instances():
        _key = tuple(inst)
        _id, _type, _concept = _key
        all_inst_ids.append(_id)
        assert _type == ':instance' and _id not in info
        cc['Gnode'] += 1
        if _key in alignments:
            cc['GnodeA'] += 1
            _idxes = sorted(alignments[_key].indices)
            if _idxes[-1]-_idxes[0]+1 != len(_idxes):
                cc['GnodeA_disc'] += 1
        else:
            cc['GnodeN'] += 1
            _idxes = None
        info[_id] = [_id, _concept, _idxes, [], None]  # id, concept, idxes, edges, ops
    for edge in graph.edges():
        cc['Gedge'] += 1
        info[edge[0]][3].append((edge[1], edge[2]))
    attrs = {}
    for attr in graph.attributes():
        _key = tuple(attr)
        _id, _role, _ = _key
        if _id not in attrs:
            attrs[_id] = []
        if _key in alignments:
            cc['GnodeA'] += 1
            _idxes = sorted(alignments[_key].indices)
        else:
            _idxes = []
        attrs[_id].append((_role, _idxes))
    # --
    # handle special nodes
    for v in info.values():
        # deal with NE
        names = [z[1] for z in v[3] if z[0]==':name']
        if len(names) > 0:
            if len(names) > 1:
                zwarn(f"Multiple :names of {v}")
            _idxes = []
            for _name in names:
                v2 = info[_name]
                assert v2[1] == 'name'
                ops = [z[1] for z in v2[3] if z[0].startswith(':op')]
                _idxes.extend(sum([info[z][2] for z in ops if info[z][2] is not None], []))
                _idxes.extend(sum([z[1] for z in attrs.get(_name, []) if z[0].startswith(':op')], []))
                # breakpoint()
            v[2] = sorted(set(_idxes))  # note: directly replace it!
            cc['Gnode_ne'] += 1
            if len(v[2]) == 0:
                # zwarn(f"Strange NE without align: {v}")
                cc['Gnode_ne?'] += 1
                v[2] = None
                # breakpoint()
        # deal with ops
        ops = [z[1] for z in v[3] if z[0].startswith(':op')]
        if len(ops) > 0 and v[1] != 'name':
            cc[f'Gnode_op'] += 1
            v[-1] = ops
    # --
    # put ef nodes
    map_ef = {}
    for k in reversed(all_inst_ids):
        if k in map_ef:
            continue
        # --
        v = info[k]
        _efs = []
        if v[-1] is not None:  # use ops instead
            for _op in v[-1]:
                v2 = info[_op]
                if v2[0] in map_ef:
                    _efs.extend(map_ef[v2[0]])
                else:
                    _ef, _ef_info = make_item(sent, v2, True)
                    cc['ef'] += 1
                    cc[f'ef_{_ef_info}'] += 1
                    if _ef is not None:
                        _efs.append(_ef)
                        map_ef[v2[0]] = [_ef]
        else:
            _ef, _ef_info = make_item(sent, v, True)
            cc['ef'] += 1
            cc[f'ef_{_ef_info}'] += 1
            if _ef is not None:
                _efs.append(_ef)
        map_ef[v[0]] = _efs
    # --
    # put evt nodes and args
    for v in info.values():
        label = v[1]
        label_fields = label.rsplit('-', 1)
        if len(label_fields) == 2 and str.isdigit(label_fields[-1]):
            cc['evtC'] += 1  # looks like a predicate
            pb_label = '.'.join(label_fields)  # use dot to join
            if label_fields[-1] in ['00', '91']:
                cc['evtC_spec'] += 1  # ignore special ones
                continue
            if onto is not None and onto.find_frame(pb_label) is None:  # extra filter
                cc['evtC_unk'] += 1  # filtered out
                continue
            cc['evtC_ok'] += 1
            cc['evtY'] += 1
            _evt, _evt_info = make_item(sent, v, False)
            cc[f'evtY_{_evt_info}'] += 1
            if _evt is not None:
                _evt.set_label(pb_label)
                # find args, todo(+N): to avoid potential problems, let it be flat
                for rr, aid in v[3]:
                    assert rr[0] == ':'
                    rr = rr[1:]
                    cc['arg'] += 1
                    if rr.startswith("ARG"):
                        role = rr  # core role!
                    else:  # pick certain
                        role = amr_role_map.get(rr, None)
                    if role is None:
                        cc[f'arg_unk'] += 1
                        continue
                    # find it
                    _efs = map_ef.get(aid, [])
                    cc[f'arg_N={min(2, len(_efs))}'] += 1
                    cc[f'argS'] += len(_efs)  # sum
                    for _ef in _efs:
                        _evt.add_arg(_ef, role=role)
    # --
    return sent, cc

def yield_sents(fd, onto):
    import penman
    # --
    cur_lines = []
    while True:
        line = fd.readline()
        if line.strip() == '' and len(cur_lines) > 0:  # one sent
            graph = penman.decode("\n".join(cur_lines))
            cur_lines = []
            sent, cc = amr2sent(graph, onto)
            cc['sent'] += 1
            yield sent, cc
        else:
            cur_lines.append(line)
        if line == '':  # EOS
            break
    # --

def yield_docs(fd, onto):
    cur_sents = []
    _gen = yield_sents(fd, onto)
    while True:
        try:
            sent, cc0 = next(_gen)
            docid = sent.info['orig_id'].rsplit('.', 1)[0]
        except StopIteration:
            sent = cc0 = docid = None
        if len(cur_sents)>0 and (cur_sents[-1][-1] != docid):  # make new doc
            new_docid = cur_sents[-1][-1]
            doc = Doc.create([z[0] for z in cur_sents], id=new_docid)
            cc = Counter()
            for z in cur_sents:
                cc += z[1]
            cc['doc'] += 1
            yield doc, cc
            cur_sents = []
        if sent is not None:
            cur_sents.append((sent, cc0, docid))
        else:
            break
    # --

# --
def add_msamr(sent_map, ff):
    import xml.etree.ElementTree as ET
    cc = Counter()
    cc['file'] += 1
    # --
    tree = ET.parse(ff)
    root = tree.getroot()  # <document>
    assert root.tag == 'document'
    # get sents
    sents = []
    for _node in root.findall("sentences"):
        for _node2 in _node.findall("amr"):
            _sid = _node2.attrib['id']
            if _sid in sent_map:
                copied_sent = Sent.cls_from_json(sent_map[_sid].to_json())  # copy as individual ones!
                copied_sent.deref()
                copied_sent.set_id(None)
                sents.append(copied_sent)
    if len(sents) == 0:
        return None, cc
    # --
    doc = Doc.create(sents)
    # read iargs
    cc['doc'] += 1
    cc['sent'] += len(doc.sents)
    sent_map = {s.info['orig_id']: s for s in doc.sents}
    for _node in root.findall("relations"):
        for _node2 in _node.findall("identity"):
            for _node3 in _node2.findall("identchain"):
                cc['identchain'] += 1
                # find efs
                efs = []
                for _mention in _node3.findall("mention"):
                    _sent = sent_map[_mention.attrib['id']]
                    _vid = _mention.attrib['variable']
                    _cands = [z for z in _sent.entity_fillers if z.id==_vid]
                    if len(_cands) > 0:
                        assert len(_cands) == 1 and _cands[0].label == _mention.attrib['concept']
                        efs.append(_cands[0])
                cc[f'identchain_N{min(5, len(efs))}'] += 1
                # iarg?
                for _iarg in _node3.findall("implicitrole"):
                    cc['implicitrole'] += 1
                    if len(efs) == 0:
                        cc['implicitrole_ef0'] += 1
                        continue  # ef not found
                    # find par
                    _sent = sent_map[_iarg.attrib['id']]
                    _vid = "E" + _iarg.attrib['parentvariable']
                    _cands = [z for z in _sent.events if z.id==_vid]
                    if len(_cands) > 0:
                        assert len(_cands) == 1 and _cands[0].label == '.'.join(_iarg.attrib['parentconcept'].rsplit("-", 1))
                        _evt = _cands[0]
                        for _ef in efs:
                            rr = _evt.add_arg(_ef, role=_iarg.attrib['argument'])
                            rr.info['implicit'] = True
                            rr.info['implicitL'] = len(efs)
                        cc['implicitrole_ok'] += 1
                    else:
                        cc['implicitrole_evt0'] += 1
                        # breakpoint()
        for _node2 in _node.findall("singletons"):
            for _node3 in _node2.findall("identchain"):
                s_mentions, s_iargs = list(_node3.findall("mention")), list(_node3.findall("implicitrole"))
                if len(s_mentions) != 1 or len(s_iargs) != 0:
                    zwarn(f"Strange singleton node: {s_mentions} {s_iargs}")
    # --
    return doc, cc

# --
class MainConf(Conf):
    def __init__(self):
        self.input_file = ''  # simply input it as shortcut
        self.input_dir = ''
        self.input_msamr_dir = ''
        self.output_file = ''
        self.onto = ''
        # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    onto = None
    if conf.onto:
        from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto
        _path = zglob1z(conf.onto)
        onto = zonto.Onto.load_onto(_path)
    # --
    if conf.input_file:
        all_docs = list(ReaderGetterConf().get_reader(input_path=conf.input_file))
        zlog(f"Read directly from {conf.input_file}")
    else:
        files = sorted(os.listdir(conf.input_dir))
        cc = Counter()
        all_docs = []
        for f in files:
            with zopen(os.path.join(conf.input_dir, f)) as fd:
                fd.readline()  # discard first line!!
                for doc, cc0 in yield_docs(fd, onto):
                    all_docs.append(doc)
                    cc += cc0
        zlog(f"Read amrs from {conf.input_dir}/{files}: {cc}")
        OtherHelper.printd(cc, try_div=True)
    # --
    if conf.input_msamr_dir:
        sent_map = {}
        for d in all_docs:
            for s in d.sents:
                sid = s.info['orig_id']
                if sid in sent_map:
                    zwarn(f"Repeated sent: {sid}")
                    continue
                sent_map[sid] = s
        ccM = Counter()
        all_docs = []
        for f in sorted(os.listdir(conf.input_msamr_dir)):
            if not f.endswith('.xml'): continue
            ff = os.path.join(conf.input_msamr_dir, f)
            doc, cc0 = add_msamr(sent_map, ff)
            if doc is not None:
                all_docs.append(doc)
            ccM += cc0
        # --
        zlog(f"Read msamrs from {conf.input_msamr_dir}: {ccM}")
        OtherHelper.printd(ccM, try_div=True)
    # --
    if conf.output_file:
        with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
            writer.write_insts(all_docs)
    # --

# --
# python3 -m msp2.tasks.zmtl3.scripts.misc.prep_amr input_dir:... output_file:...
if __name__ == '__main__':
    main(*sys.argv[1:])

# --
"""
# simply put all unsplit ones
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_amr input_dir:amr_annotation_3.0/data/alignments/unsplit/ onto:merged_f_pbAm0.json output_file:en.amr.all.json |& tee _log_amr
# sent: 59255, doc: 6367, 
# evtC: 262242, evtC_ok: 191583 (0.73), evtC_spec: 24691 (0.09), evtC_unk: 45968 (0.18)
# evtY: 191583, evtY_Disc: 6027 (0.03), evtY_NoAlign: 11676 (0.06), evtY_Ok: 173880 (0.91)
# arg_N=0: 24955 (0.07), arg_N=1: 273286 (0.81), arg_N=2: 12367 (0.04), arg_unk: 25987 (0.08)
# -> Counter({'arg': 301052, 'evt': 179907, 'sent': 59255, 'sent1': 51063, 'inst': 1})
# --
# further read msamr
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_amr input_msamr_dir:amr_annotation_3.0/data/multisentence/ms-amr-unsplit/ input_file:en.amr.all.json output_file:en.msamr.all.json |& tee _log_msamr
# sent: 8027, doc: 293, identchain: 3897
# implicitrole: 2463, implicitrole_ef0: 157 (0.06), implicitrole_evt0: 476 (0.19), implicitrole_ok: 1830 (0.74)
# ->
# group fl 'round(sum(1./a.info.get("implicitL") for a in d.gold.args if a.info.get("implicit")))'
# -> 1  ==  (0,)  22342(0.9327)[0.9327]  None; # 2  ==  (1,)   1415(0.0591)[0.9918]  None
# -> 3  ==  (2,)    176(0.0073)[0.9991]  None; 4  ==  (3,)     21(0.0009)[1.0000]  None
# fg al 'abs(d.gold.arg.sent.sid-d.gold.main.sent.sid)<=2' 'd.gold.info.get("implicit", False)'
# -> 39869(0.9392)[0.9392]; 2581(0.0608)[1.0000]
# --
# parse
for ff in en.*amr.all.json; do
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:${ff} "output_path:${ff%.json}.ud2.json"
done |& tee _log_parse
"""
