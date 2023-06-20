#

# brat format

__all__ = [
    "BratFormator", "BratFormatorConf",
]

from typing import List
import os
import datetime
import itertools
from collections import Counter, OrderedDict, defaultdict
from .base import DataFormator
from mspx.utils import Registrable, zlog, zopen, Conf, zwarn, ConfEntryCallback, auto_mkdir, ZHelper
from mspx.data.inst import DataInst, Doc, Sent, Frame, ArgLink, CharIndexer

# --
class BratFormatorConf(Conf):
    def __init__(self, default_toker='nltk'):
        self.evt_cate = "evt"
        self.ef_cate = "ef"
        self.char_map = ""  # char label mapping, simply 0->1,2->3,...
        # for brat2zjson
        self.toker = ConfEntryCallback((lambda s: self.callback_entry(s, ff=self.get_toker)), default_s=default_toker)  # by default, simple whitespace based!
        self.rm_text_positions = False  # whether removing text & positions after read_brat
        self.store_orig_ann_info = False  # store original ann info to the spans?

    def get_toker(self, key: str):
        from mspx.tools.annotate import AnnotatorConf, AnnotatorStanzaConf, AnnotatorNltkConf
        # --
        nltk_simple = (key == 'simple0')  # simply by whitespace and newline
        if nltk_simple:
            key = 'nltk'
        cls = AnnotatorConf.key2cls(key)
        ret = cls().direct_update(ann_input_mode='raw', ann_rm_text=False, _assert_exists=True)
        if isinstance(ret, AnnotatorStanzaConf):
            ret.stanza_processors = ['tokenize']  # by default only tokenize!
        if nltk_simple:
            ret.nltk_scheme = 'simple'  # by default simple scheme!
        return ret

# --
@DataFormator.rd("brat")
class BratFormator(DataFormator):
    def __init__(self, conf: BratFormatorConf, **kwargs):
        conf: BratFormatorConf = BratFormatorConf.direct_conf(conf, **kwargs)
        self.conf = conf
        # --
        _cm = conf.char_map
        self.char_map = {_cm[ii*2]: _cm[ii*2+1] for ii in range(len(_cm)//2)}
        self.char_map2 = {_cm[ii*2+1]: _cm[ii*2] for ii in range(len(_cm)//2)}
        # --
        self.toker = None
        if conf.toker is not None:
            self.toker = conf.toker.make_node()
        # --

    def label_z2b(self, t: str):  # change labels to acceptable ones
        _m = self.char_map
        if _m:
            t = ''.join([_m.get(c, c) for c in t])
        return t

    def label_b2z(self, t: str):  # convert back
        _m = self.char_map2
        if _m:
            t = ''.join([_m.get(c, c) for c in t])
        return t

    def mention2cspan(self, mention, posi_cache):
        _key = id(mention.sent)
        if _key not in posi_cache:
            posi_cache[_key] = mention.sent.get_word_positions(save=False)
        positions = posi_cache[_key]
        widx, wlen = mention.get_span()
        cstart, cend = positions[widx][0], sum(positions[widx + wlen - 1])
        return cstart, cend

    def to_obj(self, inst: Doc, cc=None) -> object:
        conf: BratFormatorConf = self.conf
        if cc is None:
            cc = Counter()
        # --
        doc_id = inst.id
        if conf.rm_text_positions:
            inst.remove_all_text()
        source_text = inst.get_text()
        # --
        _posi_cache = {}
        _norm_f = self.label_z2b
        id_maps = {}  # id(item) -> [names]
        lines = []
        cc['inst'] += 1
        cc['sent'] += len(inst.sents)
        # first all mentions
        curr_tid = 1  # start from 1
        for one_frame in inst.yield_frames(cates=[conf.evt_cate, conf.ef_cate]):
            cc[f"F{one_frame.cate}"] += 1
            assert id(one_frame) not in id_maps
            _tid = f"T{curr_tid}"
            curr_tid += 1
            c0, c1 = self.mention2cspan(one_frame.mention, _posi_cache)
            lines.append(f"{_tid}\t{_norm_f(one_frame.label)} {c0} {c1}\t{source_text[c0:c1]}")
            id_maps[id(one_frame)] = [_tid]
        # put events
        curr_eid = 1  # start from 1
        for one_frame in inst.yield_frames(cates=conf.evt_cate):
            _eid = f"E{curr_eid}"
            curr_eid += 1
            id_maps[id(one_frame)].append(_eid)  # update with an EID
        # put evts, args
        curr_rid = 1  # start from 1
        _evtrel_lines = []
        for one_frame in inst.yield_frames(cates=conf.evt_cate):  # evt and evt-arg (or evt-rel)
            _tid, _eid = id_maps[id(one_frame)]
            _line = f"{_eid}\t{_norm_f(one_frame.label)}:{_tid}"
            for aa in one_frame.args:
                _aaid = id_maps[id(aa.arg)][-1]
                if aa.info.get("is_rel"):  # note: relation on evts!
                    cc['Revt'] += 1
                    _evtrel_lines.append(f"R{curr_rid}\t{_norm_f(aa.label)} Arg1:{_eid} Arg2:{_aaid}")
                    curr_rid += 1
                else:
                    cc['Aevt'] += 1
                    _line += f" {_norm_f(aa.label)}:{_aaid}"
            lines.append(_line)
        lines.extend(_evtrel_lines)
        for one_frame in inst.yield_frames(cates=conf.ef_cate):  # relations
            _aid1 = id_maps[id(one_frame)][-1]
            for aa in one_frame.args:
                _aaid = id_maps[id(aa.arg)][-1]
                cc['Ref'] += 1
                lines.append(f"R{curr_rid}\t{_norm_f(aa.label)} Arg1:{_aid1} Arg2:{_aaid}")
                curr_rid += 1
        # put attributes
        curr_aid = 1  # start from 1
        for one_frame in inst.yield_frames(cates=[conf.evt_cate, conf.ef_cate]):
            _one_id = id_maps[id(one_frame)][-1]  # last one!
            attributes = one_frame.info.get("brat_attrs")
            if attributes is not None:
                for k, v in attributes.items():
                    if isinstance(v, bool):
                        if v:  # if True!
                            lines.append(f"A{curr_aid}\t{k} {_one_id}")
                            curr_aid += 1
                    else:
                        lines.append(f"A{curr_aid}\t{k} {_one_id} {v}")
                        curr_aid += 1
        # --
        ret_ann = "".join([z+"\n" for z in lines])
        return (doc_id, source_text, ret_ann)

    def from_obj(self, s, cc=None, toker=None) -> DataInst:
        conf: BratFormatorConf = self.conf
        _norm_f = self.label_b2z
        # --
        doc_id, str_text, str_ann = s  # doc_id, str_text, str_ann
        if str_ann is None:
            str_ann = ""  # no anns!
        ann_lines = [line.strip() for line in str_ann.split("\n") if line.strip()]
        # --
        # first read the items
        ann_items = OrderedDict()
        # mentions
        for line in ann_lines:
            fields = line.split("\t")
            tag = fields[0]
            if tag.startswith("T"):  # mention
                mtype, mposi = fields[1].split(" ", 1)
                mstart, mend = [int(z) for z in mposi.split(" ")]
                assert tag not in ann_items
                ann_items[tag] = (tag, mtype, mstart, mend)
        # events
        for line in ann_lines:
            fields = line.split("\t")
            tag = fields[0]
            if tag.startswith("E"):  # event
                s_evt, *s_args = fields[1].split()
                etype, e_tid = s_evt.split(":")
                assert e_tid in ann_items
                e_args = []  # List[(role, tid)]
                for s_arg in s_args:
                    arole, a_tid = s_arg.split(":")
                    e_args.append((arole, a_tid))
                assert tag not in ann_items
                ann_items[tag] = (tag, etype, e_tid, e_args)
        # relations
        for line in ann_lines:
            fields = line.split("\t")
            tag = fields[0]
            if tag.startswith("R"):  # relation
                rtype, a1, a2 = fields[1].split()
                assert a1.startswith("Arg1:") and a2.startswith("Arg2:")
                a1, a2 = a1[5:], a2[5:]
                assert a1 in ann_items and a2 in ann_items
                assert tag not in ann_items
                ann_items[tag] = (tag, rtype, a1, a2)
        # attributes
        for line in ann_lines:
            fields = line.split("\t")
            tag = fields[0]
            if tag.startswith("A"):  # attribute
                name, aa, value = (fields[1].split() + [True])[:3]  # by default True
                assert aa in ann_items
                ann_items[tag] = (tag, name, aa, value)
        # --
        # brat to doc
        if cc is None:
            cc = Counter()
        # build doc with raw text and do tokenize
        doc = Doc(text=str_text, id=doc_id)
        if toker is None:
            toker = self.toker
        toker.annotate([doc])
        cc['doc'] += 1
        cc['sent'] += len(doc.sents)
        cc['tok'] += sum(len(s) for s in doc.sents)
        # --
        # put items!
        char_indexer = CharIndexer.build_from_doc(doc, str_text)
        hit_items = {}  # TID -> Frame
        # first put events
        for kk, vv in ann_items.items():
            if not kk.startswith('E'): continue
            eid, etype, e_tid, e_args = vv
            tid, mtype, mstart, mend = ann_items[e_tid]
            assert etype == mtype
            cc['evt'] += 1
            cur_posi, cur_err = char_indexer.get_posi(mstart, mend - mstart)
            cc[f'evt_C={cur_err}'] += 1
            if cur_posi is not None:
                cc['evtV'] += 1
                evt = doc.sents[cur_posi[0]].make_frame(cur_posi[1], cur_posi[2], _norm_f(etype), 'evt', id=eid)
                if conf.store_orig_ann_info:
                    evt.info['brat_orig'] = ann_items[e_tid]
                hit_items[eid] = evt
                hit_items[tid] = evt  # no need to add for entities!
            else:
                zwarn(f"Cannot locate: {vv}")
                hit_items[eid] = None
                hit_items[tid] = None
        # then put remaining mentions as entities
        for kk, vv in ann_items.items():
            if not kk.startswith('T'): continue
            if kk in hit_items: continue
            tid, mtype, mstart, mend = vv
            cc['ef'] += 1
            cur_posi, cur_err = char_indexer.get_posi(mstart, mend - mstart)
            cc[f'ef_C={cur_err}'] += 1
            if cur_posi is not None:
                cc['efV'] += 1
                ef = doc.sents[cur_posi[0]].make_frame(cur_posi[1], cur_posi[2], _norm_f(mtype), 'ef', id=tid)
                if conf.store_orig_ann_info:
                    ef.info['brat_orig'] = vv
                hit_items[tid] = ef
            else:
                zwarn(f"Cannot locate: {vv}")
                hit_items[tid] = None
        # put evt arguments
        for kk, vv in ann_items.items():
            if not kk.startswith('E'): continue
            eid, etype, e_tid, e_args = vv
            cc['arg'] += 1
            _evt = hit_items[eid]
            if _evt is None:
                continue
            for arole, a_tid in e_args:
                _item = hit_items[a_tid]
                if _item is None:
                    zwarn(f"Cannot find evt-arg for {(arole, a_tid)}")
                    continue
                _evt.add_arg(_item, _norm_f(arole))
                cc['argV'] += 1
        # put relations
        for kk, vv in ann_items.items():
            if not kk.startswith('R'): continue
            tag, rtype, a1, a2 = vv
            cc['rel'] += 1
            e1, e2 = hit_items[a1], hit_items[a2]
            if e1 is None or e2 is None:
                zwarn(f"Cannot find rel for {vv}")
                continue
            cc['relV'] += 1
            alink = e1.add_arg(e2, _norm_f(rtype))  # note: simply from A1 -> A2
            alink.info['is_rel'] = 1  # note: special mark
            if conf.store_orig_ann_info:
                alink.info['brat_orig'] = vv
        # put attributes
        for kk, vv in ann_items.items():
            if not kk.startswith('A'): continue
            tag, name, aa, value = vv
            cc['attr'] += 1
            ee = hit_items[aa]
            if ee is None:
                zwarn(f"Cannot find attr for {vv}")
                continue
            cc['attrV'] += 1
            if 'brat_attrs' not in ee.info:
                ee.info['brat_attrs'] = {}
            ee.info['brat_attrs'][name] = value
        # --
        # clear all text and positions
        if conf.rm_text_positions:
            doc.remove_all_text()
        # --
        return doc

    # --
    # special bundled methods
    def write_brat(self, inst_stream, output_prefix: str):
        auto_mkdir(output_prefix, err_act='err')
        cc = Counter()
        for inst in inst_stream:
            doc_id, str_text, str_ann = self.to_obj(inst, cc=cc)
            with zopen(output_prefix + f"{doc_id}.txt", 'w') as fd:
                fd.write(str_text)
            with zopen(output_prefix + f"{doc_id}.ann", 'w') as fd:
                fd.write(str_ann)
        return cc

    def read_brat(self, inputs, cc=None, toker=None):
        if isinstance(inputs, str):
            inputs = [inputs]
        for one_file in inputs:
            if os.path.isdir(one_file):
                tfiles = [os.path.join(one_file, z) for z in sorted(os.listdir(one_file)) if z.endswith('.txt')]
            else:
                tfiles = [one_file]
            for tfile in tfiles:
                assert tfile.endswith(".txt")
                doc_id = os.path.basename(tfile)[:-4]
                afile = tfile[:-4] + ".ann"
                with zopen(tfile) as fd:
                    str_text = fd.read()
                if os.path.isfile(afile):
                    with zopen(afile) as fd:
                        str_ann = fd.read()
                else:
                    str_ann = None
                data = (doc_id, str_text, str_ann)
                doc = self.from_obj(data, cc=cc, toker=toker)
                yield doc

    # special reading the annotation logs (for annotation time)
    @staticmethod
    def read_time_log(ann_log: str, insts):
        # --
        # first read the full log!
        ann_span, ann_arc = defaultdict(list), defaultdict(list)
        ann_map = {'span': ann_span, 'arc': ann_arc}
        ann_specs = OrderedDict()
        with zopen(ann_log) as fd:
            last_doc = None  # start of a new doc
            last_fields = None  # last fields
            last_action = None  # last one write/new action!
            ending_line = "\t".join([datetime.datetime.now().isoformat(), "", "", "", "FINISH", "getDocument"])
            for line in itertools.chain(fd, [ending_line]):
                fields = line.strip().split('\t')
                f_time, f_user, f_dir, f_docid, f_sf, f_cmd, *f_args = fields
                if f_sf == 'START':
                    continue  # simply record the FINISH
                assert f_sf == 'FINISH'
                fields[0] = datetime.datetime.fromisoformat('.'.join(f_time.rsplit(',', 1)))
                # --
                if last_doc is None:
                    last_doc = fields  # the very first
                # --
                if f_cmd == 'getDocument':
                    # check/add overall time
                    if last_action is not None and last_doc is not None and last_fields != last_doc:  # note: only record until last line!
                        if last_doc[3] not in ann_specs:
                            ann_specs[last_doc[3]] = Counter()
                        _ds = (last_fields[0] - last_doc[0]).total_seconds()
                        ann_specs[last_doc[3]][f'timeD'] += _ds
                    last_doc = fields  # switch to the new doc!
                    last_action = fields
                else:
                    if last_action is None or last_action[3] != f_docid:
                        zwarn(f"Strange log without new doc: {fields}")
                        last_action = fields  # restart!
                    prev_dt, sel_dt, cur_dt = last_action[0], None, fields[0]
                    c_sig = None
                    if f_cmd in ['deleteSpan', 'deleteArc']:
                        pass  # todo(+N): currently ignore DEL, which may make some other dts inaccurate!
                    elif f_cmd == 'createSpan':
                        if last_fields[5] not in ['spanSelected', 'spanEditSelected']:
                            zwarn(f"Strange createSpan without selection: curr={fields} last={last_fields}")
                            sel_dt = fields[0]
                        else:
                            sel_dt = last_fields[0]
                        _span = eval(f_args[0])
                        assert len(_span) == 1 and len(_span[0]) == 2, "Currently assuming only cont. span!"
                        c_sig = _span[0] + [f_args[1]]  # [c0, c1, label]
                        last_action = fields
                    elif f_cmd == 'createArc':
                        if last_fields[5] not in ['arcSelected', 'arcEditSelected']:
                            zwarn(f"Strange createArc without selection: curr={fields} last={last_fields}")
                            sel_dt = fields[0]
                        else:
                            sel_dt = last_fields[0]
                        c_sig = f_args[:3]  # [H, T, label]
                        last_action = fields
                    elif f_cmd in ['spanSelected', 'spanEditSelected', 'arcSelected', 'arcEditSelected']:
                        pass
                    else:
                        zwarn(f"Strange log: {fields}")
                    # --
                    # add a new one
                    if c_sig is not None:
                        _dts = [prev_dt, sel_dt, cur_dt]
                        _cmd_key = f_cmd[6:].lower()
                        ann_map[_cmd_key][f_docid].append([tuple(c_sig), [z.isoformat() for z in _dts]])
                        if f_docid not in ann_specs:
                            ann_specs[f_docid] = Counter()
                        t0, t1, tA = [(_dts[a]-_dts[b]).total_seconds() for a,b in [(1,0), (2,1), (2,0)]]
                        ann_specs[f_docid][f'count_{_cmd_key}'] += 1
                        ann_specs[f_docid][f'time_{_cmd_key}0'] += t0
                        ann_specs[f_docid][f'time_{_cmd_key}1'] += t1
                        ann_specs[f_docid][f'time_{_cmd_key}A'] += tA
                        ann_specs[f_docid][f'timeA'] += tA
                    # --
                last_fields = fields
        # --
        for key, spec in ann_specs.items():
            for k, v in spec.items():
                if isinstance(v, float):
                    spec[k] = (round(v,3))
            zlog(f"Ann {key}: {ZHelper.resort_dict(spec)}")
        all_spec = sum(ann_specs.values(), Counter())
        zlog(f"# -- Ann ALL: {ZHelper.resort_dict(all_spec)}")
        # --
        # then store into insts
        cc = Counter()
        if insts:
            inst_map = {}
            for inst in insts:
                cc['doc'] += 1
                doc_id = inst.id
                assert doc_id not in inst_map
                item_map = {'': inst}
                inst_map[doc_id] = item_map
                # --
                for f_head in inst.get_frames():
                    cc['frame'] += 1
                    cc['alink'] += len(f_head.args)
                    if 'brat_orig' in f_head.info:
                        cc['frameB'] += 1
                        tid, mtype, mstart, mend = f_head.info['brat_orig']
                        _key = (mstart, mend, mtype)
                        if _key in item_map:
                            zwarn(f"Repeated key: {_key} {item_map[_key]} vs {f_head}")
                        item_map[_key] = f_head
                        for alink in f_head.args:
                            if 'brat_orig' in alink.info:
                                cc['alinkB'] += 1
                                tag, rtype, a1, a2 = alink.info['brat_orig']
                                _key2 = (a1, a2, rtype)
                                if _key2 in item_map:
                                    zwarn(f"Repeated key: {_key2} {item_map[_key2]} vs {alink}")
                                item_map[_key2] = alink
            for doc_id in ann_specs.keys():
                cc['ann_doc_all'] += 1
                cc['ann_doc_hit'] += int(doc_id in inst_map)
                item_map = inst_map.get(doc_id, {})
                if doc_id in inst_map:
                    info = item_map[''].info
                    if 'brat_ann_info' not in info:
                        info['brat_ann_info'] = []  # there can be multiple rounds!
                    info['brat_ann_info'].append(ZHelper.resort_dict(ann_specs[doc_id]))
                for _key, _dts in ann_span[doc_id]:
                    cc['ann_span_all'] += 1
                    cc['ann_span_hit'] += int(_key in item_map)
                    if _key in item_map:
                        item_map[_key].info['brat_ann_dts'] = _dts
                    # else:
                    #     breakpoint()
                for _key, _dts in ann_arc[doc_id]:
                    cc['ann_arc_all'] += 1
                    cc['ann_arc_hit'] += int(_key in item_map)
                    if _key in item_map:
                        item_map[_key].info['brat_ann_dts'] = _dts
                    # else:
                    #     breakpoint()
        zlog(f"# -- Assign ALL: {ZHelper.resort_dict(cc)}")
        # --
