#

# stat and possibly modify frame and arg labels

import sys
import re
from collections import Counter, defaultdict
from msp2.utils import init_everything, Conf, zlog, zwarn
from msp2.data.inst import yield_sents, yield_frames
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.tasks.zmtl3.mod.extract.evt_arg.onto import Onto

# --
class MainConf(Conf):
    def __init__(self):
        self.input = ''
        self.output = ''
        self.print_onto = False  # init onto?
        self.check_onto = ""
        self.filter_noncore = ['Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC']
        self.repl_pats = []  # pat -> sub
        self.repl_repls = []  # pat -> sub
        # --

def stat_insts(insts, print_frames: bool, onto=None, filter_noncore=None):
    cc0, cc_evt, cc_arg = Counter(), Counter(), Counter()
    cc_frames = defaultdict(Counter)
    cc0['inst'] += 1
    # --
    if onto is not None:
        m = {f.name: set([z.name for z in f.core_roles] + ([] if filter_noncore is None else filter_noncore)) for f in onto.frames}
    # --
    for sent in yield_sents(insts):
        cc0['sent'] += 1
        cc0['sent1'] += int(len(sent.events) > 0)
        for evt in sent.events:
            cc0['evt'] += 1
            if onto is not None and m.get(evt.label) is None:
                continue
            cc0['evt1'] += 1
            # --
            cc_evt[evt.label] += 1
            for arg in evt.args:
                cc0['arg'] += 1
                cc_arg[arg.label] += 1
                cc_frames[evt.label][arg.label] += 1
                if onto is not None and arg.label not in m.get(evt.label, {}):
                    zwarn(f"Cannot find {evt}/{arg} in the onto!")
                else:
                    cc0['arg1'] += 1
    cc0['aPe'] = cc0['arg'] / cc0['evt']
    cc0['aPe1'] = cc0['arg1'] / cc0['evt1']
    # --
    zlog("#--\nStat for insts:")
    for ccc in [cc_evt, cc_arg, cc0]:
        zlog(f"=>[{len(ccc)}/{sum(ccc.values())}] {ccc}")
    if print_frames:
        zlog("#--\nFrames:")
        for kk in sorted(cc_frames.keys()):
            zlog(f"'{kk}': {cc_frames[kk]},")
    # --
    return cc0, cc_evt, cc_arg, cc_frames

# modify the labels
def mod_insts(insts, pats, repls):
    cc0 = Counter()
    # --
    def _mod(_one, _what):
        _lab = _one.label
        for pp, rr in zip(pats, repls):
            _lab = re.sub(pp, rr, _lab)
        if _lab != _one.label:
            _one.set_label(_lab)
            cc0[_what+"_sub"] += 1
        else:
            cc0[_what+"_nosub"] += 1
    # --
    for frame in yield_frames(insts):
        _mod(frame, 'evt')
        for arg in frame.args:
            _mod(arg, 'arg')
    # --
    zlog(f"Stat for subs: {cc0}")

def main(*args):
    conf = MainConf()
    conf: MainConf = init_everything(conf, args)
    # --
    # first read all insts
    reader = ReaderGetterConf().get_reader(input_path=conf.input)
    insts = list(reader)
    stat_insts(insts, False)
    pats = [re.compile(z) for z in conf.repl_pats]
    mod_insts(insts, pats, conf.repl_repls)
    # --
    onto = None
    if conf.check_onto:
        onto = Onto.load_onto(conf.check_onto)
    stat_res = stat_insts(insts, False, onto=onto, filter_noncore=conf.filter_noncore)
    # --
    def _name2phrase(_n: str):
        _r = _n.lower()
        for sep in "_:.":
            _r = _r.split(sep)[-1]
        return _r
    # --
    if conf.print_onto:
        cc0, cc_evt, cc_arg, cc_frames = stat_res
        zlog("# --\n#frames:")
        for ff in sorted(cc_evt.keys()):
            _args = sorted(cc_frames[ff])
            _argsL, _argsR = _args[:1], _args[1:]
            d = {"name": ff, "vp": _name2phrase(ff), "core_roles": _args}
            d['template'] = " ".join([f"<{z}>" for z in _argsL] + [d["vp"]] + [f"<{z}>" for z in _argsR])
            zlog(f"{d},")
        zlog("# --\n#roles:")
        for rr in sorted(cc_arg.keys()):
            d = {"name": rr, "np": _name2phrase(rr)}
            zlog(f"{d},")
        zlog("# --")
    # --
    if conf.output:
        with WriterGetterConf().get_writer(output_path=conf.output) as writer:
            writer.write_insts(insts)
    # --

# --
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# --
python3 -m msp2.tasks.zmtl3.scripts.misc.stat_and_modify input:
# --
python3 -m msp2.tasks.zmtl3.scripts.misc.stat_and_modify check_onto:pbfn input:??
ZRUN ../../events/data/data21f/en.ewt.train.ud.json
=>[9/268601.210417398] Counter({'arg': 114765, 'arg1': 52260, 'evt': 40486, 'evt1': 37987, 'sent': 12543, 'sent1': 10555, 'aPe': 2.834683594328904, 'aPe1': 1.3757338036696765, 'inst': 1})
ZRUN ../../events/data/data21f/en.ontoC.train.ud.json
=>[9/1583213.551459229] Counter({'arg': 679474, 'arg1': 322233, 'evt': 221011, 'evt1': 218157, 'sent': 75187, 'sent1': 67146, 'aPe': 3.074389962490555, 'aPe1': 1.477069266629079, 'inst': 1})
ZRUN ../../events/data/data21f/en.nb_f0.dev.ud.json
=>[9/16932.54013721119] Counter({'arg': 5248, 'arg1': 3695, 'evt': 2527, 'evt1': 2525, 'sent': 1700, 'sent1': 1233, 'aPe': 2.076770874554808, 'aPe1': 1.4633663366336633, 'inst': 1})
ZRUN ../../events/data/data21f/en.fn17.exemplars.ud2.json
=>[9/1095321.28429441] Counter({'arg': 253085, 'arg1': 208511, 'evt': 173391, 'sent': 173029, 'sent1': 173028, 'evt1': 114273, 'aPe1': 1.8246742450097573, 'aPe': 1.4596201648297777, 'inst': 1})
ZRUN ../../events/data/data21f/en.amr.all.ud2.json
=>[9/1050554.2262113201] Counter({'arg': 301052, 'arg1': 279366, 'evt': 179907, 'evt1': 179907, 'sent': 59255, 'sent1': 51063, 'aPe': 1.6733756885501954, 'aPe1': 1.5528356317430672, 'inst': 1})
ZRUN ../../events/data/data21f/en.msamr.all.ud2.json
=>[9/162670.1593888286] Counter({'arg': 51307, 'arg1': 48327, 'evt': 23954, 'evt1': 23954, 'sent': 8027, 'sent1': 7096, 'aPe': 2.1418969691909493, 'aPe1': 2.017491859397178, 'inst': 1})
"""
