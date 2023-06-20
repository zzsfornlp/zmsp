#

# utils for brat (visualization, converting of annotations, ...)

import pandas
import numpy as np
from collections import Counter
import os
from mspx.utils import Conf, zlog, zwarn, ZObject, zopen, auto_mkdir, init_everything, zglobs, default_pickle_serializer, ZHelper
from mspx.data.inst import Frame, yield_sents
from mspx.data.rw import BratFormatorConf, BratFormator, ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        self.cmd = ''  # task to do: conf, z2b, b2z
        self.input_path = []
        self.output_path = ""
        self.specific_ann_conf = ""  # mat?
        self.conf_add_nil = True  # add N* ones?
        # self.convert = BratFormatorConf('simple0').direct_update(char_map=":_", rm_text_positions=True)
        self.convert = BratFormatorConf('simple0').direct_update(rm_text_positions=True, store_orig_ann_info=True)
        self.z2b_filter = ''
        self.b2z_filter = ''
        self.query_add_score = False  # add query scores
        self.check_dpar = False  # add dpar mentions for easier checking
        self.pred_type_prefix = ''  # special prefix for the labels of predicted types (such as P-)?
        self.delete_nils = False  # delete all NILs
        self.convert_qs = ''  # convert sent-q for easier visualization
        # more specific options
        self.b2z_orig_query = ""  # auto fix Q and NIL with the original query
        self.b2z_time_log = ""  # read time info from the logs
        self.b2z_default_cate = 'ef'

# --
NIL_PREFIX = "N"
# --

# --
# special processings: is_pred, is_gold(past)(** as attribute), special NILs, ...
def process_docs(conf, docs, cc, is_z2b: bool, filter_method=''):
    query_add_score, check_dpar, pred_type_prefix, delete_nils, convert_qs = conf.query_add_score, conf.check_dpar, conf.pred_type_prefix, conf.delete_nils, conf.convert_qs
    # --
    def _query_label(_item):
        return f"Q({_item.score:.3f})" if query_add_score else "Q"
    # --
    ret = []
    for doc in docs:
        cc['doc0'] += 1
        cc['sent0'] += len(doc.sents)
        has_ann, has_query, has_plain = False, False, False
        _brat_attrs_key = 'brat_attrs'
        if check_dpar:
            assert is_z2b
            for sent in doc.sents:
                for widx, dlab in enumerate(sent.tree_dep.seq_label.vals):
                    if dlab == '_Q_':
                        has_ann = has_query = True
                        _dpar_topk = sent.arrs.get('qannKV')
                        if _dpar_topk is not None:
                            _tmp_arr = sorted(_dpar_topk[1+widx].tolist())
                            _dpar_score = 1 - (_tmp_arr[-1] - _tmp_arr[-2])  # 1-margin!
                        else:
                            _dpar_score = 'UNK'
                        _dpar_item = ZObject(score=_dpar_score)
                        sent.make_frame(widx, 1, _query_label(_dpar_item), "ef")  # note: simply ef!
        for frame in doc.get_frames():  # simply process all of them
            has_ann = True
            cc['frame'] += 1
            cc[f'frame_{frame.cate}'] += 1
            # -- frame
            if is_z2b:  # zjson to brat
                if frame.label == "_Q_":  # query
                    has_query = True
                    if convert_qs and frame.mention.get_span() == (0, len(frame.sent)):
                        if convert_qs == 'D':  # delete!
                            frame.del_self()
                        else:
                            frame.mention.set_span(0, 1)
                            frame.set_label('QS')
                        cc['frame_QS'] += 1
                    else:
                        frame.set_label(_query_label(frame))
                        cc['frame_Q'] += 1
                elif frame.label.startswith("**"):  # gold
                    if _brat_attrs_key not in frame.info:
                        frame.info[_brat_attrs_key] = {}
                    frame.set_label(frame.label[2:])  # change label!
                    frame.info[_brat_attrs_key]['Gold'] = True  # already annotated in the past!
                    cc['frame_**'] += 1
                elif frame.info.get('is_pred'):
                    frame.set_label(pred_type_prefix+frame.label)  # change label!
                    cc['frame_pred'] += 1
                else:
                    has_plain = True
                    cc['frame_plain'] += 1
                if frame.label.endswith("_NIL_"):
                    if delete_nils:
                        frame.del_self()
                    else:
                        trg = {'ef': f'{NIL_PREFIX}1', 'evt': f'{NIL_PREFIX}2'}.get(frame.cate)
                        if trg is not None:
                            frame.set_label(frame.label.replace("_NIL_", trg))
            else:  # brat to zjson
                if frame.label == "Q":  # query
                    has_query = True
                    frame.set_label("_Q_")
                    cc['frame_Q'] += 1
                elif frame.label == "QS":  # sent-query
                    has_query = True
                    frame.mention.set_span(0, len(frame.sent))
                    frame.set_label("_Q_")
                    cc['frame_QS'] += 1
                elif frame.info.get(_brat_attrs_key, {}).get('Gold'):  # gold
                    frame.set_label("**" + frame.label)  # change label!
                    del frame.info[_brat_attrs_key]['Gold']
                    cc['frame_G'] += 1
                elif pred_type_prefix and frame.label.startswith(pred_type_prefix):
                    frame.set_label(frame.label[2:])  # change label!
                    frame.info['is_pred'] = True
                    cc['frame_pred'] += 1
                else:
                    has_plain = True
                    cc['frame_plain'] += 1
                for one_lab in [f"{NIL_PREFIX}1", f"{NIL_PREFIX}2"]:
                    if frame.label.endswith(one_lab):
                        if delete_nils:
                            frame.del_self()
                            break
                        frame.set_label(frame.label.replace(one_lab, '_NIL_'))
            # --
            for alink in frame.get_args():
                cc['alink'] += 1
                if is_z2b:  # zjson to brat
                    if alink.label == '_Q_':
                        has_query = True
                        alink.set_label(_query_label(alink))
                        cc['alink_Q'] += 1
                    elif alink.label.startswith("**"):
                        alink.set_label("G-" + alink.label[2:])  # change label!
                        cc['alink_**'] += 1
                    else:
                        has_plain = True
                        cc['alink_plain'] += 1
                    if alink.label.endswith("_NIL_"):
                        if delete_nils:
                            alink.del_self()
                        else:
                            alink.set_label(alink.label.replace("_NIL_", f"{NIL_PREFIX}3"))
                else:  # brat to zjson
                    if alink.label == 'Q':
                        has_query = True
                        alink.set_label('_Q_')
                        cc['alink_Q'] += 1
                    elif alink.label.startswith("G-"):
                        alink.set_label("**" + alink.label[2:])  # change label!
                        cc['alink_G'] += 1
                    else:
                        has_plain = True
                        cc['alink_plain'] += 1
                    if alink.label.endswith(f"{NIL_PREFIX}3"):
                        if delete_nils:
                            alink.del_self()
                        else:
                            alink.set_label(alink.label.replace(f"{NIL_PREFIX}3", '_NIL_'))
                # --
        # --
        adding = {'ann': has_ann, 'query': has_query, 'plain': has_plain}.get(filter_method, True)
        if not adding:
            continue
        cc['doc'] += 1
        cc['sent'] += len(doc.sents)
        ret.append(doc)
    # --
    return ret
# --

# --
def process_ann_with_queries(conf, ann_docs, query_docs, cc):
    qmap = {}
    for dd in query_docs:
        for ss in dd.sents:
            _key = ss.dsids
            assert _key not in qmap
            qmap[_key] = ss
    # --
    LAB_QUERY, LAB_NIL = '_Q_', '_NIL_'
    _CATE = conf.b2z_default_cate
    ret = []
    for doc in ann_docs:
        ret.append(doc)
        for sent_a in yield_sents(doc):
            # --
            # note: mainly using query for checking purposes!
            cc['sent'] += 1
            sent_q = qmap[sent_a.dsids]  # must be there
            assert sent_q.seq_word.vals == sent_a.seq_word.vals
            qq_spans = [z.mention.get_span() for z in sent_q.get_frames() if z.label == LAB_QUERY]
            full_mode = any(z for z in qq_spans if z == (0, len(sent_q)))
            if len(qq_spans) == 0:
                continue
            # --
            cc['sentQ'] += 1
            cc['sentQ_full'] += int(full_mode)
            if full_mode:  # for full mode, only one query!!
                if len(qq_spans) != 1:
                    zwarn(f"Strange full mode: {qq_spans}")
            # --
            # check Q frames
            old_queries = set()
            flag_aug_nil = [0] * len(sent_q)  # whether adding new nil?
            for _widx, _wlen in qq_spans:
                cc['q_frame'] += 1
                cc['q_frameLen'] += _wlen
                old_queries.add((_widx, _wlen))
                flag_aug_nil[_widx:_widx+_wlen] = [1] * _wlen
            for a_frame in sent_a.get_frames():
                _widx, _wlen = a_frame.mention.get_span()
                if a_frame.label == LAB_QUERY:
                    if (_widx, _wlen) not in old_queries:
                        zwarn(f"Strange new Q in the ann: {a_frame} of {sent_a}")
                    a_frame.del_self()  # delete it!
                else:  # mark annotated
                    cc['ann_frameLen'] += _wlen
                    flag_aug_nil[_widx:_widx+_wlen] = [0] * _wlen
            for widx, flag in enumerate(flag_aug_nil):
                if flag:
                    cc['ann_frameLen'] += 1
                    cc['ann_frame_augnil'] += 1
                    sent_a.make_frame(widx, 1, LAB_NIL, _CATE)
            # check Q alinks
            for af1 in sent_a.get_frames():
                for alink in af1.args:
                    cc['ann_alink'] += 1
                    if alink.label == LAB_QUERY:
                        cc['ann_alinkQ'] += 1
                        alink.set_label(LAB_NIL)  # this simply means NIL!
                if full_mode:  # add more explicit NIL link!
                    if af1.label == LAB_NIL: continue  # no considerations of NIL itself
                    _amap = {id(z.arg): z for z in af1.args}
                    for af2 in sent_a.get_frames():
                        if af2.label == LAB_NIL: continue
                        if af1 is af2: continue  # no self-link
                        cc['ann_alink_pairALL'] += 1
                        if id(af2) not in _amap:
                            cc['ann_alink_pairAugnil'] += 1
                            af1.add_arg(af2, LAB_NIL)
    return ret
# --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)  # real init!
    converter = BratFormator(conf.convert)
    # --
    if conf.cmd == 'conf':  # setup confs
        label_zjson2brat = converter.label_z2b  # converting labels
        # --
        _conf_add_nil = conf.conf_add_nil
        _paths = zglobs(conf.input_path)
        assert len(_paths) == 1
        vpack = default_pickle_serializer.from_file(_paths[0])
        zlog(f"Read {vpack} from {_paths}")
        entities, events, rels = [set() for z in range(3)]
        if len(vpack) > 1:  # first one is rel!
            rels.update([label_zjson2brat(z) for z in vpack[0].real_i2w])
            vpack = vpack[1:]
        for voc in vpack:  # others are mentions
            for lab in voc.real_i2w:
                _cate, _label = Frame.parse_cate_label(lab)
                if _cate == 'evt':
                    events.add(label_zjson2brat(_label))
                else:
                    entities.add(label_zjson2brat(_label))
        entities, events, rels = [sorted(z) for z in [entities, events, rels]]
        # todo(+N): to add arg/rel constraints
        lines_ann, lines_visual = [], []
        path_log = os.path.join(os.path.abspath(conf.output_path), 'ann.log')
        lines_tools = [f"[options]",
                       "Validation\tvalidate:all",
                       f"Tokens\ttokenizer:whitespace",
                       f"Sentences\tsplitter:newline",
                       f"Annotation-log\tlogfile:{path_log}"]
        # lines_tools = []  # no need!
        lines_visual.append("\n[labels]\n\n[drawing]\n")
        # by default, mark them as UNN
        # --
        # SPAN_DRAWING_ATTRIBUTES = ['fgColor', 'bgColor', 'borderColor']
        # ARC_DRAWING_ATTRIBUTES = ['color', 'dashArray', 'arrowHead', 'labelArrow']
        # ATTR_DRAWING_ATTRIBUTES = ['glyphColor', 'box', 'dashArray', 'glyph', 'position']
        # --
        lines_visual.append("SPAN_DEFAULT\tfgColor:black, bgColor:lightgreen, borderColor:darken")
        lines_visual.append("ARC_DEFAULT\tcolor:black, arrowHead:triangle-5")
        # special ones
        lines_visual.append("Q\tfgColor:black, bgColor:tomato, borderColor:darken, color:red, dashArray:3-3")
        lines_visual.append("QS\tfgColor:black, bgColor:tomato, borderColor:darken, color:red, dashArray:3-3")
        # entities
        lines_ann.append("\n[entities]\n")
        for ii, ent in enumerate(([f"{NIL_PREFIX}1"] if _conf_add_nil else []) + entities):
            lines_ann.append(ent)
            if ent == f"{NIL_PREFIX}1":
                lines_visual.append(f"{ent}\tfgColor:black, bgColor:lightgrey, borderColor:darken")
            else:
                pass
                # lines_visual.append(f"{ent}\tfgColor:black, bgColor:lightgreen, borderColor:darken")
        # events
        lines_ann.append("\n[events]\n")
        if len(events) > 0:
            for ii, evt in enumerate(([f"{NIL_PREFIX}2"] if _conf_add_nil else []) + events):
                lines_ann.append(evt)
                if evt == f"{NIL_PREFIX}2":
                    lines_visual.append(f"{evt}\tfgColor:black, bgColor:lightgrey, borderColor:darken")
                else:
                    pass
                    # lines_visual.append(f"{evt}\tfgColor:black, bgColor:lightblue, borderColor:darken")
        # rels
        _rel_args = "Arg1:<EVENT>, Arg2:<ENTITY>" if len(events)>0 else "Arg1:<ENTITY>, Arg2:<ENTITY>"
        lines_ann.append("\n[relations]\n")
        for ii, rel in enumerate(([f"{NIL_PREFIX}3"] if _conf_add_nil else []) + rels):
            lines_ann.append(f'{rel}\t{_rel_args}')
            if rel == f"{NIL_PREFIX}3":
                lines_visual.append(f"{rel}\tcolor:lightgrey, dashArray:-")
                lines_visual.append(f"G-{rel}\tcolor:lightgrey, dashArray:-")
            else:
                pass
                # lines_visual.append(f"{rel}\tcolor:black, dashArray:-")
                # lines_visual.append(f"G-{rel}\tcolor:GoldenRod, dashArray:-")
        # attributes
        lines_ann.append("\n[attributes]\n")
        lines_ann.append("Gold\tArg:<EVENT>|<ENTITY>")
        lines_visual.append("Gold\tposition:left, dashArray:-, bgColor:yellow, glyphColor:yellow, glyph:G\n")
        # --
        # output
        if conf.specific_ann_conf == 'mat':
            zlog("Replace ann.conf with special one!")
            from mspx.tasks.others.mat.brat.ann_conf import get_mat_conf
            lines_ann = get_mat_conf(conf.conf_add_nil).split('\n')
        if conf.output_path:
            auto_mkdir(conf.output_path)
            for one_fname, one_lines in zip(["annotation.conf", "visual.conf", "tools.conf"],
                                            [lines_ann, lines_visual, lines_tools]):
                with zopen(os.path.join(conf.output_path, one_fname), 'w') as fd:
                    fd.write("".join([z+"\n" for z in one_lines]))
    # --
    elif conf.cmd == 'z2b':
        cc0 = Counter()
        all_insts = list(ReaderGetterConf().get_reader(input_path=conf.input_path))
        all_insts = process_docs(conf, all_insts, cc0, is_z2b=True, filter_method=conf.z2b_filter)
        zlog(f"[z2b]Read from {conf.input_path}: {ZHelper.resort_dict(cc0)}")
        if conf.output_path:
            cc1 = converter.write_brat(all_insts, conf.output_path)
            zlog(f"[z2b]Write to {conf.output_path}: {ZHelper.resort_dict(cc1)}")
    elif conf.cmd == 'b2z':
        cc0 = Counter()
        input_files = zglobs(conf.input_path)
        all_insts = list(converter.read_brat(input_files, cc=cc0))
        zlog(f"#--\n[b2z]Read from {conf.input_path}: {ZHelper.resort_dict(cc0)}")
        if conf.output_path:
            cc1 = Counter()
            all_insts = process_docs(conf, all_insts, cc1, is_z2b=False, filter_method=conf.b2z_filter)
            zlog(f"#--\n[b2z]Process ann-docs: {ZHelper.resort_dict(cc1)}")
            if conf.b2z_orig_query:  # further adjust with the original queries
                cc2 = Counter()
                query_insts = list(ReaderGetterConf().get_reader(input_path=conf.b2z_orig_query))
                all_insts = process_ann_with_queries(conf, all_insts, query_insts, cc2)
                zlog(f"#--\nProcess ann-docs with queries: {ZHelper.resort_dict(cc2)}")
            if conf.b2z_time_log:  # add annotation time info
                zlog(f"#--\nStart to read time info")
                converter.read_time_log(conf.b2z_time_log, all_insts)
            with WriterGetterConf().get_writer(output_path=conf.output_path) as writer:
                writer.write_insts(all_insts)
    else:
        raise NotImplemented(f"UNK cmd of {conf.cmd}")
    # --

# PYTHONPATH=../src/ python3 -m mspx.tools.al.utils_brat ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
# --

# --
# install & setup brat
"""
# 220920: 44ecd825810167eed2a5d8ad875d832218e734e8
git clone https://github.com/nlplab/brat
cd brat
conda create -n brat22 python=3.8
conda activate brat22
./install.sh -u
"""
"""
git clone https://github.com/nlplab/brat
cd brat
conda create -n brat23 python=3.8
conda activate brat23
./install.sh -u
"""
# --
# run
"""
# test matching!
python3 -m mspx.tools.al.utils_brat cmd:z2b input_path:__data/evt/data/en.ace05.dev.json output_path:./ace/ z2b_filter:
python3 -m mspx.tools.al.utils_brat cmd:b2z input_path:./ace output_path:ace.dev0.json
python3 -m mspx.tasks.others.mat.filter_by_id ../../../../data/evt/data/en.ace05.dev.json ace.dev0.json ace.dev.json
python3 -m pdb -m mspx.cli.analyze frame gold:__data/evt/data/en.ace05.dev.json preds:./ace.dev.json frame_cate:evt
# --
python3 -m mspx.tools.al.utils_brat cmd:conf input_path:__vocabs/evt_en/v_rel0.pkl output_path:./
python3 -m mspx.tools.al.utils_brat cmd:z2b input_path:__go_al0/run_try0922evt_3/iter07/data.query.json output_path:./alQ/
rm -rf *.{ann,txt}
"""
RUN_VISUAL="""
# ZIN=?? ZOUT=?? ZVOC=?? RUN_VISUAL
function RUN_VISUAL () {
if [[ -z $ZOUT ]]; then
ZOUT=$(basename $ZIN)
fi
mkdir -p $ZOUT
python3 -m mspx.tools.al.utils_brat cmd:conf input_path:$ZVOC output_path:$ZOUT/
for dii in $ZIN/iter*; do
echo RUN ITER $dii
iter_name=$(basename $dii)
python3 -m mspx.tools.al.utils_brat cmd:z2b input_path:$dii/data.query.json output_path:$ZOUT/${iter_name}Q/ z2b_filter:query
python3 -m mspx.tools.al.utils_brat cmd:z2b input_path:$dii/data.ann.json output_path:$ZOUT/${iter_name}A/ z2b_filter:plain
done |& tee $ZOUT/_log.visual
ZOUT=""
}
# ZIN=../../../../go_al0/run_try1002ner_0/ ZVOC=__vocabs/ner_nl/v_ext0.pkl RUN_VISUAL
# ZIN=../../../../go_al0/_debug/run_try1002evt_0/ ZVOC=__vocabs/evt_en/v_rel0.pkl RUN_VISUAL
"""
# --
