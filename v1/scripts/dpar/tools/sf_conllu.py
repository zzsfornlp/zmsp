#

# stat and filter (filter the sentences)

import sys
from msp.utils import zopen, Helper, zlog
from msp.zext.dpar.conllu_reader import ConlluReader, write_conllu

#
def main(args):
    in_file, out_file, filter_code = (args + [None]*3)[:3]
    #
    if filter_code:
        filter_ff = compile(filter_code, "", "eval")
    else:
        filter_ff = compile("False", "", "eval")
    stat = {}
    survives = []
    # loop
    with zopen(in_file) as fd:
        rr = ConlluReader()
        idx = 0
        for p in rr.yield_ones(fd):
            # list the usable names
            tokens = p.get_tokens()
            length = len(tokens)
            word = p.get_props("word")
            upos = p.get_props("upos")
            xpos = p.get_props("xpos")
            head = p.get_props("head")
            label = p.get_props("label")
            label0 = p.get_props("label0")
            unproj = p.crossed
            num_cross = len([z for z in unproj if z])
            # stat
            Helper.stat_addone(stat, "Nsent")
            Helper.stat_addone(stat, "Ntok", length)
            Helper.stat_addone(stat, "Ctok", num_cross)
            if num_cross > 0:
                Helper.stat_addone(stat, "Csent")
            # filter
            if eval(filter_ff):
                survives.append(p)
            idx += 1
    # print stat
    stat["Rsent"] = stat.get("Csent", 0) / stat.get("Nsent")
    stat["Rtok"] = stat.get("Ctok", 0) / stat.get("Ntok")
    Helper.printd(stat)
    # output file
    if out_file:
        with open(out_file, "w") as fd:
            for onep in survives:
                onep.write(fd)
    #
    zlog("Output to %s: len=%d" % (out_file, len(survives)))


# python3 ~ {in_file} {out_file} {filter_code}
if __name__ == '__main__':
    main(sys.argv[1:])

# examples:
"""
# only stat
PYTHONPATH=../src python3 sf_conllu.py pc/ptb_train.conllu
# first 10k unproj
PYTHONPATH=../src python3 sf_conllu.py pc/ptb_train.conllu pc/ptb_train.10k.conllu "num_cross==0 and idx<10009"
"""
