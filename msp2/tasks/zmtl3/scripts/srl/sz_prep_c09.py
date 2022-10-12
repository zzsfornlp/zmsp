#

# pre-processing c09 (handle the "_" in es and remove no-used args)

import sys
from collections import Counter, OrderedDict
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
def norm_arg_name(role: str):
    ret = role.upper()
    if ret == 'LOC':
        ret = 'ARGM-LOC'
    if ret[0]=='A' and str.isdigit(ret[1]):
        ret = 'ARG' + ret[1:]
    if ret == 'ARGM-LOC' or (ret.startswith("ARG") and str.isdigit(ret[3])):
        return ret
    else:
        return None  # ignore others!
# --

def main(input_file: str, output_file: str):
    inc_arg_set = {
        'ARGM-LOC',  # en
        'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'LOC',  # zh
        'argM-loc',  # es
    }
    # --
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    all_insts = list(reader)
    cc = Counter()
    arg_cc = Counter()
    for inst in all_insts:
        cc['inst'] += 1
        for sent in yield_sents(inst):
            cc['sent'] += 1
            cc['tok'] += len(sent)
            # --
            words = list(sent.seq_word.vals)
            for ii, ww in enumerate(words):
                if ww == '_':
                    cc['tok_F'] += 1
                elif '_' in ww:
                    cc['tok_C'] += 1
                    words[ii] = " ".join(ww.split("_"))  # split it!
            sent.build_words(words)
            # --
            # delete strange args
            set_ee_heads(sent)
            for evt in sent.events:
                cc['evt'] += 1
                for arg in list(evt.args):
                    cc['arg'] += 1
                    arg_shead_tok = arg.mention.shead_token
                    if not any(str.isalnum(c) for c in arg_shead_tok.word):
                        cc['arg_D'] += 1
                        arg.delete_self()
                    else:
                        rr = norm_arg_name(arg.role)
                        if rr is None:
                            cc['arg_No'] += 1
                            arg.delete_self()
                        else:
                            cc['arg_OK'] += 1
                            arg.set_label(rr)
                            arg_cc[rr] += 1
    # --
    zlog(f"Prep es {input_file} -> {output_file}: {cc}")
    zlog(f"Arg: {arg_cc}")
    OtherHelper.printd(cc, try_div=True)
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(all_insts)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.sz_prep_c09 IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
