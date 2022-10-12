#

# create syn frames (v2)

from collections import Counter, OrderedDict
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

class MainConf(Conf):
    def __init__(self):
        self.input_file = ""  # input data
        self.output_file = ""  # output data
        self.language = 'UNK'
        # --
        self.syn_pos = ['VERB', 'NOUN']  # pos for pred
        self.syn_deplab = ['nsubj', 'obj', 'iobj', 'obl', 'nmod', 'compound']  # dep args
        # self.syn_deplabR = ['acl']  # reversed ones
        self.syn_deplabR = []  # reversed ones
        # --

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # first read data
    zlog(f"Read from {conf.input_file}")
    reader = ReaderGetterConf().get_reader(input_path=conf.input_file)
    all_insts = list(reader)
    # --
    # then process
    _syn_pos_set = set(conf.syn_pos)
    _syn_deplab_set = set(conf.syn_deplab)
    _syn_deplabR_set = set(conf.syn_deplabR)
    # --
    cc = Counter()
    for inst in all_insts:
        cc['inst'] += 1
        for sent in yield_sents(inst):
            cc['sent'] += 1
            cc['tok'] += len(sent)
            # --
            # simply clean previous ones
            sent.clear_entity_fillers()
            sent.clear_events()
            # --
            for widx, tok in enumerate(sent.tokens):
                if tok.upos in _syn_pos_set:
                    args = []
                    for ch in tok.ch_toks:
                        _lab = ch.deplab.split(":")[0]
                        if _lab in _syn_deplab_set:
                            args.append((ch, _lab))
                    _labR = tok.deplab.split(":")[0]
                    if _labR in _syn_deplabR_set:
                        args.append((tok.head_tok, "R_"+_labR))
                    # --
                    if len(args) > 0:
                        cc['pred'] += 1
                        cc['args'] += len(args)
                        # --
                        evt = sent.make_event(widx, 1, type=f'_{conf.language}_{tok.upos}')  # note: put dummy ones
                        for ch, dlab in args:
                            dlab2 = "_".join(dlab.split(":"))
                            ef = sent.make_entity_filler(ch.widx, 1, type='UNK')
                            evt.add_arg(ef, role=dlab2)
                    # --
    # --
    zlog(f"Create syn frames {conf.input_file} -> {conf.output_file}: {cc}")
    if conf.output_file:
        with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
            writer.write_insts(all_insts)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn2 ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
