#
import sys

# break full annotated sentences into partial ones

from collections import Counter
from mspx.data.inst import yield_sents, Sent
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.utils import zlog

def main(fin: str, fout: str):
    reader = ReaderGetterConf().get_reader(input_path=fin)
    insts = []
    cc = Counter()
    for sent in yield_sents(reader):
        _len = len(sent)
        cc['sent'] += 1
        cc['tok'] += _len
        _hs, _ls = sent.tree_dep.seq_head.vals, sent.tree_dep.seq_label.vals
        for ii in range(_len):
            new_sent = Sent(sent.seq_word.vals)
            new_hs, new_ls = [-1] * _len, [""] * _len
            new_hs[ii] = _hs[ii]
            new_ls[ii] = _ls[ii]
            new_sent.build_dep_tree(new_hs, new_ls)
            insts.append(new_sent.make_singleton_doc())
    zlog(f"Process {fin} => {fout}: {cc}")
    with WriterGetterConf().get_writer(output_path=fout) as writer:
        writer.write_insts(insts)
    # --

# python3 -m mspx.scripts.data.ud.utils_break_pa
if __name__ == '__main__':
    main(*sys.argv[1:])
