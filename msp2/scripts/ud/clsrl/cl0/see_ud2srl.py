#

# see the links from ud relations to srl relations

#

# stat for ud treebanks

from msp2.utils import zlog, zwarn, zopen, default_json_serializer, OtherHelper
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.data.vocab import SimpleVocab
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from collections import Counter

# --
def do_stat(insts):
    cc = Counter()
    u2s_maps = {}  # UD-lab -> SRL-lab
    set_ee_heads(insts)
    for sent in yield_sents(insts):
        cc["sent"] += 1
        cc["tok"] += len(sent)
        cc["frame"] += len(sent.events)
        # --
        _tree = sent.tree_dep
        for evt in sent.events:
            evt_widx = evt.mention.shead_widx
            ch_labs = {z: _tree.seq_label.vals[z] for z in _tree.chs_lists[evt_widx+1]}
            hit_ch_widxes = set()
            for arg in evt.args:
                arg_lab = arg.label
                arg_lab = arg_lab.lstrip("C-").lstrip("R-")  # simplify it
                if arg_lab == "V": continue
                # --
                cc['arg'] += 1
                arg_widx = arg.mention.shead_widx
                ud_lab = ch_labs.get(arg_widx, "HO")  # otherwise, make it "high-order"
                hit_ch_widxes.add(arg_widx)
                if ud_lab not in u2s_maps:
                    u2s_maps[ud_lab] = {}
                if arg_lab not in u2s_maps[ud_lab]:
                    u2s_maps[ud_lab][arg_lab] = 0
                u2s_maps[ud_lab][arg_lab] += 1
            for ci, cv in ch_labs.items():
                if ci not in hit_ch_widxes:
                    if cv in ['punct']: continue
                    if cv not in u2s_maps:
                        u2s_maps[cv] = {}
                    u2s_maps[cv]["NIL"] = u2s_maps[cv].get("NIL",0) + 1
    # --
    # get more stat
    cc2 = dict(cc)
    cc2.update({"t/s": f"{cc['tok']/cc['sent']:.2f}", "f/s": f"{cc['frame']/cc['sent']:.2f}", "a/f": f"{cc['arg']/cc['frame']:.2f}"})
    zlog(f"CC: {cc2}")
    # look at the mappings
    from msp2.tools.analyze import Analyzer, AnalyzerConf
    ana = Analyzer(AnalyzerConf())
    vv = []
    for a in u2s_maps.keys():
        for b in u2s_maps[a].keys():
            c = u2s_maps[a][b]
            if c>20:
                vv.extend([(a,b)]*c)
    ana.set_var('zz', vv)
    ana.do_group('zz', 'd')
    # --
    # breakpoint()
    # --

# --
def main(*input_files: str):
    reader_conf = ReaderGetterConf()
    reader_conf.validate()
    # --
    all_insts = []
    for ff in input_files:
        one_insts = list(reader_conf.get_reader(input_path=ff))
        zlog(f"Read from {ff}: {len(one_insts)} instances.")
        all_insts.extend(one_insts)
    do_stat(all_insts)
    # --

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# PYTHONPATH=../src/ python3 see_ud2srl.py
