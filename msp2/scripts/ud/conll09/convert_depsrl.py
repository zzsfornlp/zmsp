#

# convert dep-srl based on different dep trees

from collections import Counter
from msp2.utils import Conf, zlog, init_everything, OtherHelper
from msp2.data.inst import yield_sents, Sent
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
class MainConf(Conf):
    def __init__(self):
        self.src_input = ReaderGetterConf()  # src (dep + srl)
        self.trg_input = ReaderGetterConf()  # trg (dep)
        self.output = WriterGetterConf()
        # --
        self.method = "path"  # span/path

# --
# converting methods
class Converter:
    def __init__(self, conf: MainConf):
        self.conf = conf
        self.ff = {"path": self.convert_by_path, "span": self.convert_by_span}[conf.method]
        # --

    def get_desc(self, tree):
        _heads = tree.seq_head.vals
        ret = [set() for _ in range(len(_heads))]  # widx -> set(desc), note: including self!
        for m in range(len(_heads)):
            cur = m
            _p = set()
            while cur>=0:
                ret[cur].add(m)
                # --
                # check loop
                assert cur not in _p
                _p.add(cur)
                # --
                cur = _heads[cur]-1
        return ret

    def convert(self, src_sent: Sent, trg_sent: Sent, cc: Counter):
        cc["sent"] += 1
        assert len(src_sent) == len(trg_sent)
        src_tree = src_sent.tree_dep
        trg_tree = trg_sent.tree_dep
        # --
        # copy trg sent
        ret = Sent.create(trg_sent.seq_word.vals.copy())
        if trg_sent.seq_upos is not None:
            ret.build_uposes(trg_sent.seq_upos.vals)
        ret.build_dep_tree(trg_tree.seq_head.vals, trg_tree.seq_label.vals)
        # --
        # map items
        # first get everyone's desc set
        src_desc = self.get_desc(src_tree)
        trg_desc = self.get_desc(trg_tree)
        for src_evt in src_sent.events:
            cc["evt"] += 1
            _ewidx, _ewlen = src_evt.mention.get_span()
            assert _ewlen == 1
            trg_evt = ret.make_event(_ewidx, _ewlen, type=src_evt.type)
            for src_arg in src_evt.args:
                cc["arg"] += 1
                _awidx, _awlen = src_arg.mention.get_span()
                assert _awlen == 1
                _new_awidx = self.ff(_ewidx, _awidx, src_tree, trg_tree, src_desc, trg_desc, cc)
                trg_ef = ret.make_entity_filler(_new_awidx, 1, type=src_arg.arg.type)
                trg_evt.add_arg(trg_ef, src_arg.role)
        # --
        return ret

    def convert_by_path(self, ewidx: int, awidx: int, src_tree, trg_tree, src_desc, trg_desc, cc: Counter):
        # --
        if ewidx == awidx:
            cc["conv_0|0"] += 1
            return awidx
        # --
        spine_e, spine_a = trg_tree.get_path(ewidx, awidx, inc_common=1)  # trg_tree! include one common!
        assert spine_e[0]==ewidx and spine_a[0]==awidx
        for ii in list(reversed(spine_a)):  # only with spine_a!
            if ii in src_desc[awidx] and ii != ewidx:  # not ewidx itself!
                _s1, _s2 = trg_tree.get_path(awidx,ii)  # compare old and new one!
                cc[f"conv_{len(_s1)}|{len(_s2)}"] += 1
                return ii
        raise RuntimeError("Should not reach here!")
        # --

    def convert_by_span(self, ewidx: int, awidx: int, src_tree, trg_tree, src_desc, trg_desc, cc: Counter):        # --
        # --
        if ewidx == awidx:
            cc["conv_0|0"] += 1
            cc["span_0|0"] += 1
            return awidx
        # --
        def _score(s1, s2):
            return len(s1 & s2) - len(s1 - s2)
        # --
        a_set = src_desc[awidx]
        best_ii, best_score = awidx, _score(trg_desc[awidx], a_set)
        for ii, b_set in enumerate(trg_desc):
            if ii == ewidx: continue  # cannot be itself!
            ss = _score(b_set, a_set)
            if ss > best_score:
                best_ii, best_score = ii, ss
        # --
        b_set = trg_desc[best_ii]
        cc[f"span_{len(a_set-b_set)}|{len(b_set-a_set)}"] += 1  # compare old and new span!
        _s1, _s2 = trg_tree.get_path(awidx, best_ii)  # compare old and new one!
        cc[f"conv_{len(_s1)}|{len(_s2)}"] += 1
        return best_ii

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # first read them all
    src_sents, trg_sents = list(yield_sents(conf.src_input.get_reader())), \
                           list(yield_sents(conf.trg_input.get_reader()))
    assert len(src_sents) == len(trg_sents)
    cc = Counter()
    conv = Converter(conf)
    # --
    outputs = []
    for src_sent, trg_sent in zip(src_sents, trg_sents):
        res = conv.convert(src_sent, trg_sent, cc)
        outputs.append(res)
    zlog("Stat:")
    OtherHelper.printd(cc)
    # --
    with conf.output.get_writer() as writer:
        writer.write_insts(outputs)
    # --

# --
# python3 -m pdb convert_depsrl.py src_input.input_path:?? trg_input.input_path:?? output.output_path:??
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
"""
# convert it forw and back and compare
zgo () {
python3 convert_depsrl.py src_input.input_path:${SRC} trg_input.input_path:${TRG} output.output_path:_tmp1.json method:${METHOD}
python3 convert_depsrl.py src_input.input_path:_tmp1.json trg_input.input_path:${SRC} output.output_path:_tmp2.json method:${METHOD}
python3 -m msp2.cli.analyze frame gold:${SRC} preds:_tmp1.json,_tmp2.json auto_save_name: econf:pb no_join_c:1 </dev/null
# zz = filter fl "(lambda x: [z.mention.widx for z in x])(d.gold.args) != (lambda x: [z.mention.widx for z in x])(d.pred.args)"
}
# --
zback () {
python3 -m msp2.scripts.ud.conll09.convert_depsrl src_input.input_path:${SRC} trg_input.input_path:${TRG} output.output_path:${OUT}
python3 -m msp2.cli.analyze frame gold:${TRG} preds:${OUT} auto_save_name: econf:pb no_join_c:1 </dev/null
}
# SRC=output TRG=orig OUT=_tmp.json zback
"""
