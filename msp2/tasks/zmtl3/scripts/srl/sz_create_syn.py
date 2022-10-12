#

# create syn frames

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
        self.syn_deplab = ['nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'obl:prep',
                           'nmod', 'nmod:assmod', 'compound']  # dep args
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
    # --
    def _get_hit_args(_toks):
        _rets = []
        for _tok in _toks:
            _r = None
            if not any(str.isalnum(c) for c in _tok.word):
                continue
            if _tok.deplab in _syn_deplab_set:
                _r = _tok.deplab
            else:
                continue
            # note: filter with l2-type but discard it afterwards
            _rets.append((_tok, _r.split(":")[0]))
        return _rets
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
                    args = _get_hit_args(tok.ch_toks)
                    # --
                    # note: ignore subjs for COP
                    has_cop = (tok.upos == "NOUN") and any(z.deplab.split(":")[0] == 'cop' for z in tok.ch_toks)
                    if has_cop:
                        args = [z for z in args if not z[1].startswith('nsubj')]
                    # --
                    # note: looking for extra one: like xcomp-subj, conj-subj
                    if (not has_cop) and (not any(z[1].startswith('nsubj') for z in args)):
                        tok2 = tok
                        while tok2.deplab in ['xcomp', 'conj']:
                            tok2 = tok2.head_tok
                            if tok2 is not tok:
                                subjs = [z for z in _get_hit_args(tok2.ch_toks) if z[1].startswith('nsubj')]
                                if len(subjs) > 0:  # note: should be only one!
                                    args.append(subjs[0])
                                    break  # find one, enough!
                    # --
                    if len(args) > 0:
                        cc['pred'] += 1
                        cc['args'] += len(args)
                        # --
                        # note: simply use word form!
                        # from .s2_aug_onto import get_word_lemma
                        # _lemma = get_word_lemma(tok.word, conf.language)
                        _word = str.lower(tok.word)
                        _dummy_id = {'VERB': '01', 'NOUN': '02'}[tok.upos]
                        evt = sent.make_event(widx, 1, type=f'{_word}.{conf.language}{_dummy_id}')  # note: put dummy ones
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

# python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
