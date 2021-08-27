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
    voc = SimpleVocab.build_empty()
    for sent in yield_sents(insts):
        cc["sent"] += 1
        cc["tok"] += len(sent)
        cc["tok_pair"] += len(sent)**2
        _tree = sent.tree_dep
        _deplabs = _tree.seq_label.vals
        _slen = len(sent)
        for i0 in range(_slen):
            for i1 in range(_slen):
                if abs(i0-i1) > 5:
                    continue
                path1, path2 = _tree.get_path(i0, i1)
                labs1, labs2 = sorted([[_deplabs[z].split(":")[0] for z in path1], [_deplabs[z].split(":")[0] for z in path2]])
                _len = len(labs1) + len(labs2)
                # if _len<=0 or _len>2 or "punct" in labs1 or "punct" in labs2:
                if _len != 2 or "punct" in labs1 or "punct" in labs2:
                    continue
                _k = (tuple(labs1), tuple(labs2))
                voc.feed_one(_k)
    # --
    zlog(cc)
    voc.build_sort()
    d = voc.get_info_table()
    print(d[:100].to_string())
    # breakpoint()
    # --

# --
def do_stat_srl(insts):
    cc = Counter()
    cc_narg = Counter()
    voc = SimpleVocab.build_empty()
    # set_ee_heads(insts)
    voc_pred, voc_arg = SimpleVocab.build_empty(), SimpleVocab.build_empty()
    voc_deplab = SimpleVocab.build_empty()
    for sent in yield_sents(insts):
        cc["sent"] += 1
        cc["tok"] += len(sent)
        cc["frame"] += len(sent.events)
        # --
        _tree = sent.tree_dep
        if _tree is not None:
            voc_deplab.feed_iter(_tree.seq_label.vals)
        for evt in sent.events:
            voc_pred.feed_one(evt.label)
            evt_widx = evt.mention.shead_widx
            cc_narg[f"NARG={len(evt.args)}"] += 1
            for arg in evt.args:
                voc_arg.feed_one(arg.label)
                cc["arg"] += 1
                # check arg overlap
                for a2 in evt.args:
                    if a2 is arg: continue  # not self
                    if not (arg.mention.widx >= a2.mention.wridx or a2.mention.widx >= arg.mention.wridx):
                        cc["arg_overlap"] += 1
                    else:
                        cc["arg_overlap"] += 0
    # --
    voc.build_sort()
    voc_pred.build_sort()
    voc_arg.build_sort()
    voc_deplab.build_sort()
    # --
    # get more stat
    cc2 = dict(cc)
    cc2.update({"t/s": f"{cc['tok']/cc['sent']:.2f}", "f/s": f"{cc['frame']/cc['sent']:.2f}", "a/f": f"{cc['arg']/cc['frame']:.2f}"})
    zlog(f"CC: {cc2}")
    zlog(cc_narg)
    zlog(voc_arg.counts)
    # --
    MAX_PRINT_ITEMS = 20
    d_pred = voc_pred.get_info_table()
    print(d_pred[:MAX_PRINT_ITEMS].to_string())
    d_arg = voc_arg.get_info_table()
    print(d_arg[:MAX_PRINT_ITEMS].to_string())
    d_deplab = voc_deplab.get_info_table()
    print(d_deplab[:MAX_PRINT_ITEMS].to_string())
    d = voc.get_info_table()
    print(d[:MAX_PRINT_ITEMS].to_string())
    # --
    # breakpoint()
    # --

# --
def main(input_format, *input_files: str):
    reader_conf = ReaderGetterConf().direct_update(input_format=input_format)
    reader_conf.validate()
    # --
    all_insts = []
    for ff in input_files:
        one_insts = list(reader_conf.get_reader(input_path=ff))
        zlog(f"Read from {ff}: {len(one_insts)} instances.")
        all_insts.extend(one_insts)
    # --
    if input_format == "conllu":
        do_stat(all_insts)
    do_stat_srl(all_insts)
    # --

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# PYTHONPATH=../src/ python3 stat_udsrl.py [format] *files
