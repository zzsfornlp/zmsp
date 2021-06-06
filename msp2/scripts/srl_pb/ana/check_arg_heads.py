#

# check arg heads

from collections import Counter
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import ZObject, OtherHelper

#
def main(input_path):
    insts = list(ReaderGetterConf().get_reader(input_path=input_path))  # read from stdin
    all_sents = list(yield_sents(insts))
    set_ee_heads(insts)
    # --
    cc = Counter()
    for sent in all_sents:
        cc["sent"] += 1
        arg_maps = [[] for _ in range(len(sent))]
        for evt in sent.events:
            cc["evt"] += 1
            for arg in evt.args:
                # --
                # no VERB
                if arg.role in ["V", "C-V"]:
                    cc["argV"] += 1
                    continue
                # --
                cc["arg"] += 1
                ef = arg.arg
                shidx = ef.mention.shead_widx
                span = ef.mention.get_span()
                arg_maps[shidx].append(ZObject(evt=evt, ef=ef, span=span))
        # check for all tokens
        cc["tok"] += len(arg_maps)
        for one_objs in arg_maps:
            cc[f"tok_N{len(one_objs)}"] += 1
            all_spans = set(z.span for z in one_objs)
            cc[f"tok_N{len(one_objs)}S{len(all_spans)}"] += 1
            # --
            if len(one_objs) > 0:
                cc[f"tok_diff={len(all_spans)>1}"] += 1
            if len(all_spans) > 1:
                breakpoint()
                pass
        # --
    # --
    OtherHelper.printd(cc)

# PYTHONPATH=../src/ python3 check_arg_heads.py [input]
# note: (around 5% in 05-dev); 1) argV, 2) NP inside PP, 3) modifying predicates inside, ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
