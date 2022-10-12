#

# try to align frames according to reprs

import os
import sys
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
from msp2.data.inst import yield_sents, yield_frames, set_ee_heads, DataPadder, Sent
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, default_pickle_serializer, OtherHelper, zglob, ZObject
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto
from msp2.tasks.zmtl3.mod.pretrained import ZBmodConf
from msp2.nn import BK
from msp2.nn.l3 import SubPoolerLayer

class MainConf(Conf):
    def __init__(self):
        # --
        # collect repr & align (src is English, trg is foreign!)
        self.bconf = ZBmodConf()
        # self.bconf.b_model = "bert-base-multilingual-cased"
        self.bconf.b_model = 'xlm-roberta-base'
        self.b_hlayers = [8]
        self.b_alayers = list(range(12))
        # --
        self.src_onto = ""
        self.src_files = []
        self.src_repr = ""
        self.trg_onto = ""
        self.trg_files = []
        self.trg_repr = ""
        # --
        # align
        self.trig_ws = [0., 0., 1.]
        self.role_ws = [1., 0.]  # 786**0.5/144**0.5~2.34
        self.csls = 1.
        self.fixed_map_roles = ["ARG0", "ARGM-LOC"]  # these are fixingly mapped
        self.aligned_onto = ""  # file for aligned one
        self.add_amloc = True  # always add amloc if the aligned one has this!
        # --
        # final map
        self.map_input = ""
        self.map_output = ""
        # --

# --
class ReprGetter:
    def __init__(self, conf: MainConf):
        self.conf = conf
        self.bmod = conf.bconf.make_node()
        self.bmod.to(BK.DEFAULT_DEVICE)
        zlog("Finished loading bmod.")
        self.sub_pooler = SubPoolerLayer(conf=None, pool_hid_f='mean2', pool_att_f='mean4')
        # --

    def forward_seqs(self, sents):
        conf = self.conf
        _MAX_CLEN = 128
        sub_toker = self.bmod.sub_toker
        _tokenizer = self.bmod.tokenizer
        _pad_id, _cls_id, _sep_id = _tokenizer.pad_token_id, _tokenizer.cls_token_id, _tokenizer.sep_token_id
        # --
        arr_ids, arr_lens = [], []
        for sent in sents:
            sub_vals, sub_idxes, sub_info = sub_toker.sub_vals(sent.seq_word.vals)
            _cur_ids, _cur_lens = [], []
            for a, b in zip(sub_info.orig2begin, sub_info.orig2end):
                if len(_cur_ids) + (b-a) <= _MAX_CLEN:
                    _cur_ids.extend(sub_idxes[a:b])
                    _cur_lens.append(b-a)
                else:  # truncate!
                    break
            arr_ids.append([_cls_id] + _cur_ids + [_sep_id])
            arr_lens.append([1] + _cur_lens + [1])
        # [bs, ext_len], [bs, orig_len]
        arr_ids, arr_lens = DataPadder.go_batch_2d(arr_ids, _pad_id), DataPadder.go_batch_2d(arr_lens, 0)
        t_ids, t_lens = BK.input_idx(arr_ids), BK.input_idx(arr_lens)
        bert_out = self.bmod.forward_enc(t_ids, (t_ids != _pad_id).float())
        # get results (h and att)
        t_h0 = BK.stack(
            [bert_out.hidden_states[_ii] for _ii in conf.b_hlayers], -1).mean(-1)  # simply average all layers, [bs,L0,D]
        t_att0 = BK.concat(
            [bert_out.attentions[_ii].transpose(-2,-3).transpose(-1,-2) for _ii in conf.b_alayers], -1)  # [bs,Q0,K0,L*H]
        # back to original seq
        t_h = self.sub_pooler.forward_hid(t_h0, t_lens)[:,1:-1]  # [bs,L,D]
        t_att = self.sub_pooler.forward_att(t_att0, t_lens)[:,1:-1,1:-1]  # [bs,Q,K,L*H]
        return t_h, t_att

class ReprList(list):
    def get_value(self):
        ret = np.stack(self, 0).mean(0)
        return ret

def yield_batches(stream, bsize: int):
    cache = []
    for inst in stream:
        if len(cache) >= bsize:
            yield cache
            cache = []
        cache.append(inst)
    if len(cache) > 0:
        yield cache
    # --

# --
def get_onto_reprs(onto_file: str, data_files: list, repr_file: str, conf: MainConf):
    if not onto_file:
        return None, None
    # --
    onto = zonto.Onto.load_onto(onto_file)
    inc_noncore = {'Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC'}
    for ff in onto.frames:
        ff.build_role_map(nc_filter=(lambda _name: _name in inc_noncore), force_rebuild=True)
    # --
    if os.path.exists(repr_file):
        zlog(f"Read repr_file from {repr_file}", timed=True)
        onto_repr = default_pickle_serializer.from_file(repr_file)
    else:  # collect from data
        _BSIZE = 16
        repr_getter = ReprGetter(conf)
        frame2data = OrderedDict()  # frame-type: {T:[lemma, vp, [hs...]], role-name: [[atts...], [hs...], ]}}
        zlog(f"Collect repr-data from frames", timed=True)
        for frames in yield_batches(onto.frames, _BSIZE):
            sents_lemma = [Sent.create([b for a in _f.name.split('.')[0].split() for b in a.split("_")]) for _f in frames]
            sents_vp = [Sent.create([b for a in _f.vp.split() for b in a.split("_")]) for _f in frames]
            h_lemma, _ = repr_getter.forward_seqs(sents_lemma)
            h_vp, _ = repr_getter.forward_seqs(sents_vp)
            for _ii, _ff in enumerate(frames):
                # note: simply get the first sub-token since these are usually short phrases!
                _dd = {'T': [BK.get_value(h_lemma[_ii, 0]), BK.get_value(h_vp[_ii, 0]), ReprList()]}
                _dd.update({k: [ReprList(), ReprList()] for k in _ff.role_map.keys()})
                frame2data[_ff.name] = _dd
        # --
        cc = Counter()
        zlog(f"Collect repr-data from {data_files}", timed=True)
        for data_file in data_files:
            reader = ReaderGetterConf().get_reader(input_path=data_file)
            for sents in yield_batches(yield_sents(reader), _BSIZE):
                t_h, t_att = repr_getter.forward_seqs(sents)  # [bs, L, DH], [bs, L, L, DA]
                _mlen = BK.get_shape(t_h, -2)  # [L]
                set_ee_heads(sents)
                for _ii, _ss in enumerate(sents):
                    cc['sent'] += 1
                    for _ee in yield_frames(_ss):
                        _ftype = _ee.label
                        _ewidx = _ee.mention.shead_widx
                        if onto.find_frame(_ftype) is None:
                            continue
                        cc['evt'] += 1
                        if _ewidx >= _mlen:
                            cc['evt0'] += 1
                            continue  # get truncated!
                        frame2data[_ftype]['T'][2].append(BK.get_value(t_h[_ii, _ewidx]))
                        for _aa in _ee.args:
                            _arole = _aa.label
                            if _arole not in frame2data[_ftype] or _aa.mention.sent is not _ee.sent:
                                continue  # non-active or non-local role
                            cc['arg'] += 1
                            _awidx = _aa.mention.shead_widx
                            if _awidx >= _mlen:
                                cc['arg0'] += 1
                                continue  # get truncated!
                            frame2data[_ftype][_arole][0].append(BK.get_value(t_att[_ii, _ewidx, _awidx]))
                            frame2data[_ftype][_arole][1].append(BK.get_value(t_h[_ii, _awidx]))
        zlog(f"Read and repr finish with: {cc}", timed=True)
        # --
        # filter only for those with data!
        cc.clear()
        onto_repr = OrderedDict()
        for k, v in frame2data.items():
            cc['frame'] += 1
            cc['role'] += len(v) - 1
            if len(v['T'][2]) > 0:
                # breakpoint()
                cc['frameV'] += 1
                new_v = {}
                new_v['T'] = v['T'][:2] + [v['T'][2].get_value()]
                for role, v2 in v.items():
                    if role != 'T' and len(v2[0])>0:
                        cc['roleV'] += 1
                        new_v[role] = [z.get_value() for z in v2]
                onto_repr[k] = new_v
        zlog(f"Prepare onto_repr with: {cc}", timed=True)
        if repr_file:
            default_pickle_serializer.to_file(onto_repr, repr_file)
    # --
    return onto, onto_repr
    # --

def align_onto(onto_src, repr_src, onto_trg, repr_trg, conf: MainConf):
    _trig_ws = conf.trig_ws
    _role_ws = conf.role_ws
    # --
    def _do_norm(_t):
        _t = BK.input_real(_t)
        return _t / _t.norm(2, -1, keepdim=True).clamp_min(1e-5)
    # --
    # first align frames
    # note: only lemma is not good!
    src_fs, src_ts = [], []
    trg_fs, trg_ts = [], []
    for _fs, _ts, _repr in zip([src_fs, trg_fs], [src_ts, trg_ts], [repr_src, repr_trg]):
        for k, v in _repr.items():
            _fs.append(k)
            _ts.append(np.concatenate([a*b for a,b in zip(v['T'], _trig_ws)]))
    t_src, t_trg = _do_norm(src_ts), _do_norm(trg_ts)
    # ignore certain src ones
    _sim = BK.matmul(t_trg, t_src.T)  # [T, S]
    _sim = _sim * 2 - conf.csls * _sim.topk(10, dim=0)[0].mean(0)  # [T, S]
    for _ii, _ss in enumerate(src_fs):
        if _ss.endswith(".LV") or len([z for z in onto_src.find_frame(_ss).role_map if not z.startswith('ARGM')])==0:
            _sim[:, _ii] = 0.
    # --
    # note: currently simple use top-1!
    _aligns = BK.get_value(_sim.argmax(-1))  # [T]
    # --
    # pp (lambda x,y,z: [(x[ii], y[aa]) for ii,aa in enumerate(z)])(trg_fs,src_fs,_aligns)
    # from pprint import pprint as pp
    # pp((lambda x,y,z: [(x[ii], y[aa]) for ii,aa in enumerate(z)])(trg_fs,src_fs,_aligns))
    # breakpoint()
    # --
    # then align args
    ret_frames, ret_roles = [], []
    cc = Counter()
    confusion_matrix = np.zeros([7,7], dtype=np.int32)  # [T, S] A0-A5,AM
    for ii, aa in enumerate(_aligns):
        sname, tname = src_fs[aa], trg_fs[ii]
        s_ff, t_ff = onto_src.find_frame(sname), onto_trg.find_frame(tname)
        s_roles, t_roles = sorted(s_ff.role_map.keys()), sorted(t_ff.role_map.keys())
        if conf.add_amloc:
            _al = 'ARGM-LOC'
            if _al in s_roles and _al not in t_roles:
                t_roles.append(_al)
        s_rr, t_rr = repr_src[sname], repr_trg[tname]
        s_reprs, t_reprs = \
            [(np.concatenate([a*b for a,b in zip(s_rr[z], _role_ws)]) if z in s_rr else None) for z in s_roles], \
            [(np.concatenate([a*b for a,b in zip(t_rr[z], _role_ws)]) if z in t_rr else None) for z in t_roles]
        _dim = [len(z) for z in (s_reprs + t_reprs) if z is not None]
        _dim = _dim[0] if len(_dim)>0 else 1
        _arr_zero = np.zeros(_dim)  # pad zero for those without stats
        ta_src, ta_trg = _do_norm([(_arr_zero if z is None else z) for z in s_reprs]), \
                         _do_norm([(_arr_zero if z is None else z) for z in t_reprs])  # [SR, D'], [ST, D']
        # --
        cc['frameAll'] += 1
        _simA = BK.get_value(BK.matmul(ta_trg, ta_src.T))  # [T, S]
        for tii, t_role in enumerate(t_roles):
            cc['roleA'] += 1
            hit_fixes = [z for z in conf.fixed_map_roles if t_role.startswith(z)]
            if len(hit_fixes) > 0:
                assert len(hit_fixes) == 1
                if hit_fixes[0] in s_ff.role_map:
                    sii = s_roles.index(hit_fixes[0])
                    cc['roleFix'] += 1
                    _simA[tii, sii] += 10.  # make it preferred
        # todo(+N): simply greedily pick!
        _amap = {'T': sname}
        _hit_sroles = set()
        t2s = [None] * len(_simA)
        for _ in range(len(t2s)):
            _max_tii = _simA.max(-1).argmax()
            _max_sii = _simA[_max_tii].argmax()
            t2s[_max_tii] = _max_sii
            _simA[_max_tii, :] -= 1000.
            _simA[:, _max_sii] -= 5.
        for tii, t_role in enumerate(t_roles):
            s_role = s_roles[t2s[tii]]
            _amap[t_role] = s_role
            _hit_sroles.add(s_role)
            _cm0, _cm1 = -1 if t_role.startswith("ARGM") else int(t_role[3]), \
                         -1 if s_role.startswith("ARGM") else int(s_role[3])
            confusion_matrix[_cm0, _cm1] += 1
            cc['roleHitn'] += (s_role.split("-")[0] == t_role.split("-")[0])
        # --
        # make a new frame
        new_roles = [deepcopy(s_ff.role_map[k][0]) for k in sorted(_hit_sroles)]
        new_frame = zonto.Frame(tname, s_ff.vp, core_roles=new_roles, a_map=_amap, **s_ff.info)
        new_frame.template = [z for z in s_ff.template if z[0] is None or z[0] in _hit_sroles]
        ret_frames.append(new_frame)
        ret_roles.extend(new_roles)
        # --
    ret_onto = zonto.Onto(ret_frames, ret_roles)
    # --
    cm_tags = [f"A{z}" for z in range(6)] + ["AM"]
    zlog(f"Align roles and get {ret_onto}: {OtherHelper.printd_str(cc)}\n and Confusion matrix =\n{pd.DataFrame(confusion_matrix, index=cm_tags, columns=cm_tags)}")
    # --
    return ret_onto

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # read reprs
    onto_src, repr_src = get_onto_reprs(conf.src_onto, conf.src_files, conf.src_repr, conf)
    onto_trg, repr_trg = get_onto_reprs(conf.trg_onto, conf.trg_files, conf.trg_repr, conf)
    # --
    # align things
    onto_aligned_trg = align_onto(onto_src, repr_src, onto_trg, repr_trg, conf)
    if conf.aligned_onto:
        default_json_serializer.to_file(onto_aligned_trg.to_json(), conf.aligned_onto, indent=2)
        # also print pp_str format
        inc_noncore = {'Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC'}
        pp_s = onto_aligned_trg.pp_str(inc_noncore)
        with zopen(conf.aligned_onto + ".txt", 'w') as fd:
            fd.write(pp_s)
    # --
    if conf.map_input and conf.map_output:
        cc = Counter()
        reader = ReaderGetterConf().get_reader(input_path=conf.map_input)
        with WriterGetterConf().get_writer(output_path=conf.map_output) as writer:
            for inst in reader:
                cc['inst'] += 1
                for frame in yield_frames(inst):
                    cc['frame'] += 1
                    _ff = onto_aligned_trg.find_frame(frame.label)
                    if _ff is not None:
                        cc['frame_hit'] += 1
                        _a_map = _ff.info['a_map']
                        for arg in frame.args:
                            _old_label = arg.label
                            arg.set_label(_a_map.get(_old_label, _old_label))
                            cc['arg'] += 1
                            cc['arg_changed'] += (arg.label != _old_label)
                writer.write_insts(inst)
        zlog(f"Map {conf.map_input} => {conf.map_output}: {cc}")
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.sz_align_onto ...
if __name__ == '__main__':
    main(*sys.argv[1:])
