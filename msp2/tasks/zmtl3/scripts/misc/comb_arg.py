#

# combine results

import sys
import math
from collections import Counter, defaultdict
from msp2.utils import init_everything, zglob1z, zlog, OtherHelper, Conf, zwarn
from msp2.tools.analyze import FrameAnalyzer, FrameAnalyzerConf, AnnotationTask
from msp2.data.inst import set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.proc.eval import *
from msp2.proc import ResultRecord

# --
class MaincConf(Conf):
    def __init__(self):
        super().__init__()
        # --
        self.input0 = ""
        self.input1 = ""
        self.output = ""
        self.onto = 'ace'
        self.econf = FrameEvalConf()
        self.filter_noncore = ['Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC']
        self.add_n = 1.  # rate if <=1., else num
        self.add_thr = 1e-5  # thr on score
        self.exc_cand = False  # also exclude by cands?
        # --

def main(*args):
    conf: MaincConf = init_everything(MaincConf(), args)
    # --
    if conf.onto:
        from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto
        _path = zglob1z(conf.onto)
        onto = zonto.Onto.load_onto(_path)
    else:
        onto = None
    evaler = FrameEvaler(conf.econf)
    insts0 = list(ReaderGetterConf().get_reader(input_path=conf.input0))
    insts1 = list(ReaderGetterConf().get_reader(input_path=conf.input1))
    eres = evaler.eval(insts0, insts1)
    zlog(f"#=====\nEval with {conf.input0} vs. {conf.input1}: res = {eres}\n{eres.get_detailed_str()}")
    # --
    # combine
    _add_thr = conf.add_thr if conf.add_thr<=0. else math.log(conf.add_thr)
    _filter_noncore = set(conf.filter_noncore)
    cc = Counter()
    all_cands = []
    for pp in eres.frame_pairs:
        f0, f1 = pp.gold, pp.pred
        cc['frame'] += 1
        if f0 is None or f1 is None or f0.label != f1.label or (onto is not None and onto.find_frame(f0.label) is None):
            cc['frame_nope'] += 1
            continue
        cc['frame_match'] += 1
        # check args
        if onto is not None:
            frame = onto.find_frame(f0.label)
            frame.build_role_map(nc_filter=(lambda _name: _name in _filter_noncore), force_rebuild=True)
            all_arg_names = list(frame.role_map.keys())
        else:
            all_arg_names = list(set([z.label for z in f0.args] + [z.label for z in f1.args]))
        # --
        cc['aname'] += len(all_arg_names)
        args0, args1 = {z:[] for z in all_arg_names}, {z:[] for z in all_arg_names}
        for _f, _as, _ai in zip([f0, f1], [args0, args1], [0,1]):
            for z in _f.args:
                if z.label not in _as:
                    zwarn(f"Unfound name: {z}")
                    continue
                _as[z.label].append(z)
                cc[f'arg{_ai}'] += 1
        # --
        set_ee_heads([f0.sent, f1.sent])
        a0_cands = set(v.mention.shead_token.get_indoc_id() for vs in args0.values() for v in vs)
        for aname in all_arg_names:
            cc['aname_sys0'] += int(len(args0[aname])>0)
            cc['aname_sys1'] += int(len(args1[aname])>0)
            if len(args1[aname])>0 and len(args0[aname])==0:
                cc['aname_sys1p'] += 1
                for aa in args1[aname]:
                    if not (conf.exc_cand and aa.mention.shead_token.get_indoc_id() in a0_cands):
                        all_cands.append((aa.score, aa, f0))
    # --
    cc['cand'] = len(all_cands)
    all_cands.sort(key=(lambda x: x[0]), reverse=True)
    add_n = int(conf.add_n if conf.add_n>1 else len(all_cands) * conf.add_n)
    all_cands = all_cands[:add_n]
    cc['cand_n'] = len(all_cands)
    all_cands = [z for z in all_cands if z[0]>=_add_thr]
    cc['cand_final'] = len(all_cands)
    for _ss, _aa, _ff in all_cands:
        # find sent
        sid = _aa.mention.sent.sid
        if sid is None:  # simply put it at the same sent!
            _sent = _ff.sent
        else:
            _sent = _ff.doc.sents[sid]
        # add it!
        _widx, _wlen = _aa.mention.get_span()
        _ef = _sent.make_entity_filler(_widx, _wlen, type=_aa.arg.label)
        _ff.add_arg(_ef, role=_aa.label)
        # breakpoint()
        # --
    # --
    zlog(f"Results of cc:\n{OtherHelper.printd_str(cc, try_div=True)}")
    # breakpoint()
    # --
    if conf.output:
        with WriterGetterConf().get_writer(output_path=conf.output) as writer:
            writer.write_insts(insts0)
    # --

# --
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
for thr in 0.0001 0.5 0.75 0.8 0.9 0.95 0.97 0.99; do
python3 -m msp2.tasks.zmtl3.scripts.misc.comb_arg input0:zout_srl2/_zout.ee_en.ace2.dev.json_test.json input1:zout_qa1/_zout.ee_en.ace2.dev.json_test.json output:tmp.json add_thr:$thr
python3 -m msp2.tasks.zmtl3.scripts.misc.ana_arg gold_set_ee_heads:1 gold:../../events/data/data21f/en.ace2.dev.json "preds:zout_srl2/_zout.ee_en.ace2.dev.json_test.json,zout_qa1/_zout.ee_en.ace2.dev.json_test.json,tmp.json" do_loop:0
done |& tee _log_comb
grep -E "lab-arg:|cand_final" _log_comb
# note: not as good if using qa1r!
# --
base0 = 375.0/611.0=0.6137; 375.0/759.0=0.4941; 0.5474 [srl]
base1 = 375.0/975.0=0.3846; 375.0/759.0=0.4941; 0.4325 [qa]
0 vs 1 = 311.0/975.0=0.3190; 311.0/611.0=0.5090; 0.3922
thr=0: 527 (1.00) 486.0/1138.0=0.4271; 486.0/759.0=0.6403; 0.5124
thr=0.5: 348 (0.66) 470.0/959.0=0.4901; 470.0/759.0=0.6192; 0.5471
thr=0.75: 223 (0.42) 445.0/834.0=0.5336; 445.0/759.0=0.5863; 0.5587
thr=0.8: 205 (0.39) 443.0/816.0=0.5429; 443.0/759.0=0.5837; 0.5625
thr=0.9: 165 (0.31) 436.0/776.0=0.5619; 436.0/759.0=0.5744; 0.5681 (*)
thr=0.95: 116 (0.22) 419.0/727.0=0.5763; 419.0/759.0=0.5520; 0.5639
thr=0.97: 86 (0.16) 408.0/697.0=0.5854; 408.0/759.0=0.5375; 0.5604
thr=0.99: 41 (0.08) 392.0/652.0=0.6012; 392.0/759.0=0.5165; 0.5556
# +exc_cand: (slightly better(
thr=0.8: 158 (0.36) 434.0/769.0=0.5644; 434.0/759.0=0.5718; 0.5681
thr=0.9: 127 (0.29) 427.0/738.0=0.5786; 427.0/759.0=0.5626; 0.5705
thr=0.95: 84 (0.19) 410.0/695.0=0.5899; 410.0/759.0=0.5402; 0.5640
"""
