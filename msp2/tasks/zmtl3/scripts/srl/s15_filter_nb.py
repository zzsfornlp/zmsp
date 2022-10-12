#

# specific filtering/converting for nombank frames

import os
from collections import Counter, defaultdict
from msp2.data.inst import yield_sents, yield_frames
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, mkdir_p
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

class MainConf(Conf):
    def __init__(self):
        self.onto_nb = ""  # nb onto frames
        self.onto_pb = ""  # pb onto frames
        self.input_files = []   # data to convert
        self.output_onto = ""  # output nb onto (adding a field of "pb_frame")
        self.output_dir = "."  # output dir (use the same filename)
        self.output_sub = []  # 1 -> 2, for example: nb. -> nb_filter2.
        self.only_ann = False  # annotate only
        self.rm_reuse = False  # rm reused ones

# --
# modify inplace
def add_nb2pb(onto_nb: zonto.Onto, onto_pb: zonto.Onto):
    cc = Counter()
    # --
    def _seems_n4v(_n: str, _v: str):
        for _e in ["ment", "ion", "er", "or"]:
            if _n.endswith(_e):
                _n = _n[:-len(_e)]
        _c = 0
        for a,b in zip(_n, _v):
            if a==b: _c+=1
            else: break
        return _c/len(_n)
    # --
    # first build maps of pb
    frame_map = {f.name: f for f in onto_pb.frames}
    alias_map = defaultdict(set)
    for f in onto_pb.frames:
        for a in f.frame_aliases:
            alias_map[a].add(f.name)
    # then check nb frames
    nb_splits = defaultdict(list)
    for f in onto_nb.frames:
        cc['nb_all'] += 1
        _pb_fname = None
        _lemma = f.name.rsplit(".", 1)[0]
        source_fname = f.info.get("frame_source")
        cand_by_aliases = alias_map.get(_lemma, {})
        seems_n4v = False if source_fname is None else (_seems_n4v(_lemma, source_fname.rsplit(".",1)[0])>=0.6)
        # --
        # check frame
        if source_fname is not None:
            if source_fname in frame_map:
                if source_fname in cand_by_aliases:
                    _kind = 'nb_good'
                elif seems_n4v:
                    _kind = 'nb_fine'
                else:
                    _kind = 'nb_reuse'
            else:
                _kind = 'nb_badsrc'
                source_fname = None  # simply discard ...
        else:
            _kind = 'nb_nosrc'
        # check roles
        if source_fname is not None:
            _pbf = frame_map[source_fname]
            _r1, _r2 = sorted([r.name for r in f.core_roles]), sorted([r.name for r in _pbf.core_roles])
            if _r1 == _r2:
                _rkind = 'ok'
            elif all(_r in _r2 for _r in _r1):
                _rkind = 'smaller'
            else:
                _rkind = 'differ'
                # zwarn(f"Different: {f.name}({_r1}) vs {_pbf.name}({_r2})")
            cc[f'rr_{_rkind}'] += 1
        cc[_kind] += 1
        nb_splits[_kind].append(f)
        f.info['pb_map'] = (source_fname, _kind)
    # --
    # pp [(f,f.frame_source) for f in nb_splits['nb_reuse']]
    # breakpoint()
    zlog(f"Process {onto_nb}: {cc}")
    # --

# change data (inplace)
def convert_data(insts, onto: zonto.Onto, conf: MainConf):
    frame_map = {f.name: f for f in onto.frames}
    ok_kinds = {'nb_good', 'nb_fine', 'nb_reuse'}
    if conf.rm_reuse:
        ok_kinds.remove('nb_reuse')
    # --
    cc = Counter()
    for sent in yield_sents(insts):
        cc['all_sent'] += 1
        for frame in list(sent.events):
            cc['all_frame'] += 1
            _f = frame_map.get(frame.label)
            _name, _kind = _f.pb_map if _f is not None else (None, 'unk')
            if _kind not in ok_kinds:
                cc[f'frame_rm_{_kind}'] += 1
                _rm = True
            else:
                cc[f'frame_kept_{_kind}'] += 1
                _rm = False
            # --
            if conf.only_ann:  # only annotate
                frame.info['pb_map'] = (_name, _kind)
            else:
                if _rm:
                    sent.delete_frame(frame, 'evt')
                else:
                    if frame.label != _name:
                        cc[f'frame_change'] += 1
                        frame.set_label(_f.pb_map[0])
            # --
    # --
    return cc

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # add map info to onto
    onto_nb = zonto.Onto.load_onto(conf.onto_nb)
    onto_pb = zonto.Onto.load_onto(conf.onto_pb)
    add_nb2pb(onto_nb, onto_pb)
    if conf.output_onto:
        default_json_serializer.to_file(onto_nb.to_json(), conf.output_onto, indent=2)
    # --
    # change dataset
    if conf.input_files:
        mkdir_p(conf.output_dir)
        for f in conf.input_files:
            reader = ReaderGetterConf().get_reader(input_path=f)
            insts = list(reader)
            cc = convert_data(insts, onto_nb, conf)
            zlog(f"Read {f}: {cc}")
            _output_file0 = os.path.basename(f)
            if conf.output_sub:
                a, b = conf.output_sub[:2]
                _output_file0 = _output_file0.replace(a, b)
            _output_file = os.path.join(conf.output_dir, _output_file0)
            assert not os.path.exists(_output_file), f"For safety, no overwriting of {_output_file}"
            with WriterGetterConf().get_writer(output_path=_output_file) as writer:
                writer.write_insts(insts)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.s15_filter_nb onto_nb:f_nb.json onto_pb:f_pb.json
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
