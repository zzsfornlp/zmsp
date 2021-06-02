#

import sys
from msp.zext.ana import AnalyzerConf, Analyzer, ZRecNode, AnnotationTask

try:
    from .ann import *
except:
    from ann import *

def main(args):
    conf = AnalysisConf(args)
    # =====
    if conf.load_name == "":
        # recalculate them
        # read them
        zlog("Read them all ...")
        gold_parses = list(yield_ones(conf.gold))
        sys_parses = [list(yield_ones(z)) for z in conf.fs]
        # use vocab?
        voc = Vocab.read(conf.vocab) if len(conf.vocab)>0 else None
        # =====
        # stat them
        zlog("Stat them all ...")
        all_sents, all_tokens = get_objs(gold_parses, sys_parses, conf.getter)
        analyzer = ParsingAnalyzer(conf.ana, all_sents, all_tokens, conf.labeled, vocab=voc)
        if conf.save_name != "":
            analyzer.do_save(conf.save_name)
    else:
        analyzer = ParsingAnalyzer(conf.ana, None, None, conf.labeled)
        analyzer.do_load(conf.load_name)
    # =====
    # analyze them
    zlog("Analyze them all ...")
    analyzer.loop()
    return analyzer

if __name__ == '__main__':
    main(sys.argv[1:])

"""
# example (pdb can use readline)
PYTHONPATH=../src/ python3 -m pdb run.py gold:en_dev.conllu fs:zout.g1,zout.ef vocab:
x1 = filter sents "sum(z.s1.ucorr for z in d.rtoks)<d.len"
x2 = filter sents "sum(z.s1.ucorr for z in d.rtoks)<d.len or sum(z.s2.ucorr for z in d.rtoks)<d.len"
a = ann_start x2
aj
aj 1
# y1 = sort x2 "-np.average([z.s1.g1s for z in d.rtoks if z.s1.ucorr<1])"  # rank by error confidence
# group tokens
group tokens "(d.g.label, d.s1.ucorr)"
group tokens "(get_coarse_type(d.g.label), d.s1.ucorr)"
group tokens "(get_coarse_type(d.g.label), tuple(sorted([get_coarse_type(z) for z in d.g.childs_labels])))"
# with back_gap: this token is the underlooked node
group tokens "(d.s1.ucorr, d.s1.edge.back_gap)"
group tokens "(d.s1.edge.back_gap, d.s1.ucorr)"
group tokens "(d.s1.edge.back_gap, d.s1.label)"
group tokens "(d.s1.edge.back_gap, d.g.label0)"
group tokens "(d.s1.edge.back_gap, d.g.upos)"
# group fixes
j = join sents "d.ptrees[0].fixes"
group j "(d.type, d.category[0])"
# group tokens
group tokens "(get_coarse_type(d.g.label), (d.s1.ucorr, d.s2.ucorr))"
group tokens "(get_coarse_type(d.g.label), d.s1.ucorr)"
group tokens "(get_coarse_type(d.g.label), d.s2.ucorr)"
group tokens "(get_coarse_type(d.g.label), )"
#
group tokens "(min(vs.voc.get(d.g.word, -10) // 10, 100), )"
group tokens "(1 if vs.voc.get(d.g.word, 1000000)<10 else 0, )"
# freq
group tokens "(get_coarse_type(d.g.label), )"
group tokens "(val2bin(vs.voc.get(d.g.word, 1000000), [10, 100, 1000, 10000]), )"
group tokens "(val2bin(vs.voc.get(d.g.word, 1000000), [50, 100, 500, 1000, 10000, 20000]), )"
group tokens "(val2bin(vs.voc.get(d.g.word, 1000000), [50, 100, 500, 1000, 10000, 20000]), len(d.g.childs))"
# freq and #children
group tokens "(1 if (vs.voc.final_vals[vs.voc.get(d.g.word)] if vs.voc.get(d.g.word) is not None else 0)>0 else 0, len(d.g.childs))"
"""
