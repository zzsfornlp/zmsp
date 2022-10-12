#

# extend srl to msent
# (dir @qadistill)

from collections import Counter, OrderedDict
from msp2.data.inst import yield_sents, set_ee_heads, Doc, Sent
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

class MainConf(Conf):
    def __init__(self):
        self.onto = 'pbfn'
        self.inc_noncore = ['Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC']
        self.input_file = ""  # input data
        self.output_file = ""  # output data
        # --
        self.sdist = 2  # sdist <= this
        self.max_cands = 2  # choose the nearby ones
        self.match_lemma = False  # match lemma rather than word
        self.ignore_deplabs = []  # ignore these
        self.repl_orig = ""   # replace what for the original arg? '' means simply delete!
        # --
        self.allow_same_sent = False  # allow same sent?
        self.use_match = True  # use string matching
        self.use_coref = False  # use coref
        # --

# --
def get_conjs(tok, span_range):
    rets = []
    for tok2 in tok.ch_toks:
        if tok2.deplab in ['conj', 'appos'] and tok2.widx >= span_range[0] and tok2.widx < span_range[1]:
            rets.append(tok2)
    return rets

def create_msarg_insts(arg_link, conf: MainConf, cc):
    # --
    from .s2_aug_onto import get_word_lemma
    _word_feat = (lambda x: get_word_lemma(x, 'en')) if conf.match_lemma else (lambda x: x)
    # --
    evt, ef = arg_link.main, arg_link.arg
    ef_widx, ef_wlen = ef.mention.get_span()
    ef_wridx = ef_widx + ef_wlen
    evt_hwidx = evt.mention.shead_widx
    c_sent = evt.sent
    cc['tl'] += 1
    # c1: already diff sent or evt inside arg
    if ef.sent is not c_sent or (evt_hwidx>=ef_widx and evt_hwidx<ef_wridx):
        cc['tl_c1diff'] += 1
        return None
    # c2: only NN POS
    shead_token = ef.mention.shead_token
    is_pron = (shead_token.upos == "PRON")
    if (shead_token.upos not in ["NOUN", "PROPN"]) and not (conf.use_coref and is_pron):
        cc['tl_c2nnn'] += 1
        return None
    # c3: should not be a predicate
    _pred_rels = {'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp',
                  'obl', 'vocative', 'expl', 'dislocated', 'advcl', 'aux', 'cop', 'mark'}
    if any((ch.deplab.split(":")[0] in _pred_rels and ch.widx>=ef_widx and ch.widx<ef_wridx) for ch in shead_token.ch_toks):
        cc['tl_c3pred'] += 1
        return None
    # c4: ignore certain relations
    if shead_token.deplab.split(":")[0] in conf.ignore_deplabs:
        cc['tl_c4rel'] += 1
        return None
    # --
    _mwe_rels = {'fixed', 'flat', 'compound'}
    ef_sigs = []
    for _ef_tok in [shead_token] + get_conjs(shead_token, (ef_widx, ef_wridx)):
        if _ef_tok.upos in ["NOUN", "PROPN"]:  # check again!
            _ef_sig0 = {'h': _word_feat(_ef_tok.word), 'p': _ef_tok.upos}
            _ef_sig1 = [_word_feat(z.word) for z in sorted(
                [ch for ch in _ef_tok.ch_toks if ch.deplab.split(":")[0] in _mwe_rels], key=lambda x: x.widx)]
            ef_sigs.append((_ef_sig0, _ef_sig1))
    # --
    # get cand sents
    _c_sent_idx = c_sent.doc.sents.index(c_sent)
    ctx_sents = c_sent.doc.sents[max(0, _c_sent_idx-conf.sdist):_c_sent_idx+conf.sdist+1]
    candidates = []
    if conf.use_match and (not is_pron):  # should not allow pronouns here!
        for _s in ctx_sents:
            if _s is c_sent and (not conf.allow_same_sent):
                continue
            for _t in _s.tokens:
                if _s is c_sent and _t.widx >= ef_widx and _t.widx < ef_wridx:
                    continue  # should not be inside!
                for _ef_sig0, _ef_sig1 in ef_sigs:
                    _sig0 = {'h': _word_feat(_t.word), 'p': _t.upos}
                    if _sig0 == _ef_sig0:
                        _sig1 = [_word_feat(z.word) for z in sorted(
                            [ch for ch in _t.ch_toks if ch.deplab.split(":")[0] in _mwe_rels], key=lambda x: x.widx)]
                        if _sig1 == _ef_sig1:  # matched!
                            candidates.append(_t)
                            break
    if conf.use_coref:
        _coref_tids = set(c_sent.info['info_coref'].get(shead_token.get_indoc_id(True), []))
        if len(_coref_tids) > 0:
            for _s in ctx_sents:
                if _s is c_sent and (not conf.allow_same_sent):
                    continue
                for _t in _s.tokens:
                    if _s is c_sent and _t.widx >= ef_widx and _t.widx < ef_wridx:
                        continue  # should not be inside!
                    if _t.get_indoc_id(True) in _coref_tids:
                        if _t not in candidates and _t.upos in ["NOUN", "PROPN"]:  # filter UPOS here again!
                            candidates.append(_t)
    # c5: nomatch
    if len(candidates) == 0:
        cc['tl_c5nomatch'] += 1
        return None
    # --
    # filter cands (sort by dist)
    candidates.sort(key=lambda x: (abs(x.sent.sid - c_sent.sid), x.sent.sid, x.widx * (-1 if x.sent.sid<c_sent.sid else 1)))
    candidates = candidates[:conf.max_cands]
    # --
    # we have matched items!
    cc[f'tl_c6m'] += 1
    cc[f'tl_c6m_N={len(candidates)}'] += 1
    # cc[f'tl_c6m_R={shead_token.deplab}'] += 1
    # --
    # make a new inst (also copy various info!)
    tok_map = {}
    new_sents = []
    for _s in ctx_sents:
        _tmp_seq = [[id(_t), _t.word, _t.upos, _t.lemma, _t] for _t in _s.tokens]
        _tmp_orig_str = " ".join([z[1] for z in _tmp_seq[ef_widx:ef_wridx]])
        if _s is c_sent:  # replace the original arg span with ??
            _tmp_seq = _tmp_seq[:ef_widx] + ([[None, conf.repl_orig.replace("[ORIG]", _tmp_orig_str), "X", "X", None]] if conf.repl_orig else []) + _tmp_seq[ef_wridx:]
        _new_sent = Sent.create([z[1] for z in _tmp_seq])
        _new_sent.build_uposes([z[2] for z in _tmp_seq])
        # --
        for _ii, z in enumerate(_tmp_seq):
            if z is not None:
                tok_map[z[0]] = _new_sent.tokens[_ii]
        new_sents.append(_new_sent)
        # --
        _new_dp_heads, _new_dp_labs = [], []
        for _tmp_one in _tmp_seq:
            if _tmp_one[-1] is not None and _tmp_one[-1].head_idx < 0:
                _head, _lab = 0, 'root'
            else:
                try:
                    _tmp_new_tok = tok_map[id(_tmp_one[-1])]
                    _tmp_new_head = tok_map[id(_tmp_one[-1].head_tok)]
                    _head, _lab = _tmp_new_head.widx+1, _tmp_one[-1].deplab
                except:  # note: simply ...
                    _head, _lab = 0, 'dep'
            _new_dp_heads.append(_head)
            _new_dp_labs.append(_lab)
        _new_sent.build_dep_tree(_new_dp_heads, _new_dp_labs)
        # --
    new_inst = Doc.create(new_sents)
    # add the frame
    new_evt_token = tok_map[id(evt.mention.shead_token)]
    new_evt = new_evt_token.sent.make_event(new_evt_token.widx, 1, type=evt.label)
    for arg in evt.args:
        if arg is not arg_link:
            new_ef_token = tok_map.get(id(arg.arg.mention.shead_token), None)
            if new_ef_token is not None:
                new_ef = new_ef_token.sent.make_entity_filler(new_ef_token.widx, 1, type=arg.arg.label)
                new_evt.add_arg(new_ef, role=arg.role)
                # --
                for _old_conj in get_conjs(arg.arg.mention.shead_token, (ef_widx, ef_wridx)):
                    _new_conj = tok_map.get(id(_old_conj), None)
                    if _new_conj is not None:  # add conj here!
                        new_ef = new_ef_token.sent.make_entity_filler(_new_conj.widx, 1, type=arg.arg.label)
                        new_evt.add_arg(new_ef, role=arg.role)
                # --
            else:
                cc['warn_nesting_arg'] += 1
    for ms_cand in candidates:  # note: add "fake" dist ones
        new_ef_token = tok_map[id(ms_cand)]
        new_ef = new_ef_token.sent.make_entity_filler(new_ef_token.widx, 1, type='DistEF')
        new_evt.add_arg(new_ef, role=arg_link.role)
    # put information
    if 'arg_scores' in evt.info:
        _map2 = {_t.get_indoc_id(True): tok_map.get(id(_t)) for _s in ctx_sents for _t in _s.tokens}
        new_arg_scores = {k1: {_map2[k2].get_indoc_id(True): v2 for k2, v2 in v1.items() if _map2.get(k2) is not None}
                          for k1, v1 in evt.info['arg_scores'].items()}
        new_evt.info['arg_scores'] = new_arg_scores
    return new_inst
# --

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # first read data
    onto = zonto.Onto.load_onto(conf.onto)
    _inc_noncore = set(conf.inc_noncore)
    for ff in onto.frames:
        ff.build_role_map(nc_filter=(lambda _name: _name in _inc_noncore), force_rebuild=True)
    # --
    zlog(f"Read from {conf.input_file}")
    reader = ReaderGetterConf().get_reader(input_path=conf.input_file)
    all_insts = list(reader)
    # --
    rets = []
    cc = Counter()
    for inst in all_insts:
        cc['inst'] += 1
        set_ee_heads(inst)
        for sent in yield_sents(inst):
            cc['sent'] += 1
            for evt in list(sent.events):
                cc['evt'] += 1
                # --
                # filtering!
                _ff = onto.find_frame(evt.label)
                if _ff is None:
                    cc['evt_N'] += 1
                    sent.delete_frame(evt, 'evt')
                    continue
                # --
                # args
                cc['evt_Y'] += 1
                cc['evtY'] += 1
                for arg in list(evt.args):
                    cc['evtY_arg'] += 1
                    _label = arg.label
                    # --
                    # note: norm for fn
                    if _label.endswith("_1") or _label.endswith("_2"):
                        arg.set_label((_label[:-3] + 'ies') if _label[-3] == 'y' else (_label[:-2] + 's'))
                        _label = arg.label
                    # --
                    if _label not in _ff.role_map:
                        arg.delete_self()
                        cc['evtY_arg_N'] += 1
                    else:
                        cc['evtY_arg_Y'] += 1
                # --
                # find msent candidates
                # note: individually create for each of them!
                hit_this_evt = False
                for arg in list(evt.args):
                    _new = create_msarg_insts(arg, conf, cc)
                    if _new is not None:
                        hit_this_evt = True
                        rets.append(_new)
                cc[f'evtY_new={int(hit_this_evt)}'] += 1
    # --
    zlog(f"Create ms-insts {conf.input_file} -> {conf.output_file}: {cc}")
    OtherHelper.printd(cc, try_div=True)
    if conf.output_file:
        with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
            writer.write_insts(rets)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

"""
for ff in *.q2.json; do
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl 'repl_orig:<mask>' input_file:$ff output_file:${ff%.q2.json}.q2Mmask.json
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl repl_orig: input_file:$ff output_file:${ff%.q2.json}.q2Mempty.json
done |& tee _log_ms
# --
# note: first run without considering conj
# -- evtY=>evtY_new=1; tl=>tl_c5m
# ewtD: 4636=>466(0.10); 6210=>496(0.08)
# ewtT: 37987=>4809(0.13); 52260=>5174(0.10)
# nomb: 56939=>12654(0.22); 86760=>13340(0.15)
# onto: 218157=>38493(0.18); 322233=>42354(0.13)
# --
# note: run with conj -> slightly more
# ewtD: 4636=>479(0.10); 6210=>511(0.08)
# ewtT: 37987=>4920(0.13); 52260=>5313(0.10)
# nomb: 56939=>12861(0.23); 86760=>13564(0.16)
# onto: 218157=>39108(0.18); 322233=>43114(0.13)
# --
# further with extra options (use "empty")
for ff in *.q2.json; do
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl match_lemma:0 ignore_deplabs: input_file:$ff output_file:${ff%.q2.json}.q2Me2.json
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl match_lemma:0 ignore_deplabs:nsubj,obj input_file:$ff output_file:${ff%.q2.json}.q2Me3.json
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl match_lemma:1 ignore_deplabs: input_file:$ff output_file:${ff%.q2.json}.q2Me4.json
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl match_lemma:1 ignore_deplabs:nsubj,obj input_file:$ff output_file:${ff%.q2.json}.q2Me5.json
done |& tee _log_ms2
# => Me2/3/4/5
# ewtD: 511 (0.08) / 173 (0.03) / 609 (0.10) / 210 (0.03)
# ewtT: 5313 (0.10) / 1976 (0.04) / 6184 (0.12) / 2314 (0.04)
# nomb: 13564 (0.16) / 10174 (0.12) / 15348 (0.18) / 11561 (0.13)
# onto: 43114 (0.13) / 13432 (0.04) / 47908 (0.15) / 14947 (0.05)
# --
# + coref
# python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl use_coref:1 input_file:_tmp.json output_file:_tmp2.json
for ff in *.q2.json; do
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_coref input_file:$ff output_file:${ff%.q2.json}.q3C.json
done |& tee _log_ms3C
for ff in *.q3C.json; do
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl use_coref:1 use_match:0 input_file:$ff output_file:${ff%.q3C.json}.q3Me0.json
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl use_coref:1 use_match:1 input_file:$ff output_file:${ff%.q3C.json}.q3Me1.json
python3 -m msp2.tasks.zmtl3.scripts.srl.zz_extend_srl use_coref:1 use_match:1 match_lemma:1 input_file:$ff output_file:${ff%.q3C.json}.q3Me2.json
done |& tee _log_ms3M
# => Me0/1/2
# ewtD: 432 (0.07) / 753 (0.12) / 839 (0.14)
# ewtT: 4559 (0.09) / 7676 (0.15) / 8497 (0.16)
# nomb: 11242 (0.13) / 19058 (0.22) / 20778 (0.24)
# onto: 45199 (0.14) / 67115 (0.21) / 71673 (0.22)
"""
