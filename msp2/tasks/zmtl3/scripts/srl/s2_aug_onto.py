#

# prepare aug info for onto based on the corpus

import os
import re
from copy import deepcopy
from collections import Counter, OrderedDict
from shlex import split as sh_split
import numpy as np
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob
from msp2.data.rw import ReaderGetterConf
from msp2.data.inst import yield_sents, set_ee_heads
import xml.etree.ElementTree as ET
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

class MainConf(Conf):
    def __init__(self):
        self.input_onto = ""
        self.output_onto = ""
        self.input_files = []  # corpus for statistics
        self.debug = False
        self.language = 'en'
        # --
        # aug options
        # add 'who' to qw?
        self.qw_who_smooth = 10  # add for smoothing
        self.qw_who_thr = 0.1  # thr to add 'who'
        # decide tpl (order and prep)
        self.tpl_core_nameorder = False  # simply use ARG? names for the core roles
        self.tpl_core_mixratio = 0.2  # depends more on specific ones
        self.tpl_noncore_mixratio = 0.8  # depends more on shared ones
        self.tpl_prep_thr = 0.25  # thr to add a prep
        self.tpl_prep_trans = False  # whether translate preps
        self.tpl_in_amloc = False  # simply use 'in' for ARGM-LOC
        self.tpl_frozen_dist = False  # pre-defined distance (mainly for syn)
        # remove low-count frames
        self.rm_frame_thr = 0  # remove if <=this
        self.rm_frame_noverb = False  # remove if no verbs
        self.rm_frame_norole = False  # remove if no roles

# --
class TplRecord:
    def __init__(self):
        self.count = 0
        self.count_preps = Counter()
        self.count_dists = Counter()

    def record_one(self, prep, dist):
        self.count += 1
        self.count_preps[prep] += 1
        self.count_dists[dist] += 1

    def mix_one(self, other: 'TplRecord', other_ratio: float):
        ret = TplRecord()
        ret.count = 1.  # normed!
        assert other_ratio>=0 and other_ratio<=1
        # --
        for rec, ratio in zip([self, other], [1-other_ratio, other_ratio]):
            for k, v in rec.count_preps.items():
                ret.count_preps[k] += ratio * (v/rec.count)
            for k, v in rec.count_dists.items():
                ret.count_dists[k] += ratio * (v/rec.count)
        return ret

    def get_preps(self, thr: float, translate_src: str):
        # note: empty if no counts!
        rets = [k for k,v in self.count_preps.most_common() if k is not None and v/self.count>=thr]
        if translate_src is not None:
            rets = [get_word_translation(z, translate_src, 'en', False) for z in rets]
        return rets

    def get_dist(self, df: float, use_dir='max'):
        c_left = sum(v for k,v in self.count_dists.items() if k<0)
        c_right = sum(v for k,v in self.count_dists.items() if k>=0)
        v_left = sum(v*k for k,v in self.count_dists.items() if k<0)
        v_right = sum(v*k for k,v in self.count_dists.items() if k>=0)
        c_all = c_left + c_right
        v_all = v_left + v_right
        # --
        if use_dir == 'max':
            if c_left > c_right:
                cc, vv = c_left, v_left
            else:
                cc, vv = c_right, v_right
        elif use_dir == 'left':
            cc, vv = c_left, v_left
        elif use_dir == 'right':
            cc, vv = c_right, v_right
        elif use_dir == 'all':
            cc, vv = c_all, v_all
        else:
            raise NotImplementedError(f"UNK dir of {use_dir}")
        # --
        ret = df if cc==0 else vv/cc
        return ret  # return averaged dist

# --
def get_language_specific_sets(language: str):
    if language == 'en':
        pronoun_set = {
            'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        }
        prep_set = {
            'of', 'in', 'to', 'for', 'on', 'with', 'at', 'from', 'by', 'as', 'about', 'into', 'like', 'over', 'after',
            'off', 'through', 'between', 'under', 'against', 'during', 'before', 'within', 'without',
        }  # note: these already cover 95%
    elif language == 'zh':
        pronoun_set = {'他', '我', '我们', '他们', '她', '自己', '双方', '你', '大家', '谁', '您', '你们', '她们'}
        prep_set = {
            '在', '对', '以', '从', '为', '与', '向', '由', '因为', '据', '于', '到', '比', '为了', '根据',
            '就', '对于', '除了', '用', '通过', '经过', '随着', '针对', '自', '有关',
            # '将', '把',
        }  # cover around 90% before-ADP
    elif language == 'es':
        pronoun_set = {'le', 'me', 'quien', 'nos', 'él', 'ellos', 'ella', 'yo', 'nosotros', 'quienes', 'te'}
        # note: not super sure about this set, nevertheless not quite important though ...
        prep_set = {
            'de', 'en', 'a', 'del', 'con', 'por', 'para', 'al', 'entre', 'sobre', 'desde', 'hasta', 'sin', 'ante',
            'contra', 'durante', 'tras', 'según', 'hacia', 'bajo',
        }  # almost 95%
    elif language == 'NOPE':
        pronoun_set = set()
        prep_set = set()
    else:
        raise NotImplementedError(f"Unknown language: {language}")
    return pronoun_set, prep_set

# word lemma
def get_word_lemma(word: str, language: str):
    from pattern.en import lemma as lemma_en
    from pattern.es import lemma as lemma_es
    # --
    if language == 'en':
        lemma_f = lemma_en
    elif language == 'zh':  # simply skip for zh
        lemma_f = lambda x: x
    elif language == 'es':
        lemma_f = lemma_es
    else:
        raise NotImplementedError(f"Unknown language: {language}")
    # --
    ret = word.lower()
    for ii in range(5):
        try:  # note: sometimes this raises error
            ret = lemma_f(ret)
            break
        except:
            continue
    return ret

# word translation
def get_word_translation(word: str, src_lang: str, trg_lang: str, do_lemma: bool, use_w2w=False):
    if use_w2w:
        from word2word import Word2word
    dict0, dict1 = None, None
    if trg_lang == 'en':
        if src_lang == 'en':
            dict0 = {}
            dict1 = (lambda x: [x])
        elif src_lang == 'zh':
            dict0 = {
                '在': 'in', '对': 'to', '以': 'by', '从': 'from', '为': 'for', '与': 'with', '向': 'towards', '由': 'due to',
                '因为': 'because of', '据': 'according to', '于': 'at', '到': 'to', '比': 'than', '为了': 'for', '根据': 'according to',
                '就': 'on', '对于': 'upon', '除了': 'expect', '用': 'with', '通过': 'through', '经过': 'through', '随着': 'with',
                '针对': 'to', '自': 'from', '有关': 'about',
                # '将', '把',
            }
            if use_w2w:
                dict1 = Word2word('zh_cn', 'en')
        elif src_lang == 'es':
            dict0 = {
                'de': 'of', 'en': 'in', 'a': 'at', 'del': 'of', 'con': 'with', 'por': 'for', 'para': 'for', 'al': 'at',
                'entre': 'between', 'sobre': 'on', 'desde': 'since', 'hasta': 'until', 'sin': 'without', 'ante': 'before',
                'contra': 'against', 'durante': 'during', 'tras': 'after', 'según': 'according to', 'hacia': 'towards', 'bajo': 'under',
            }
            if use_w2w:
                dict1 = Word2word('es', 'en')
    assert dict0 is not None, f"Currently not supported for {src_lang} -> {trg_lang}"
    # --
    ret = word.lower()
    if ret in dict0:
        ret = dict0[ret]
    else:
        try:
            ret = dict1(ret)[0]
        except:
            ret = None
    if ret is not None:
        ret = ret.lower()
        if do_lemma:
            ret = get_word_lemma(ret, trg_lang)
    return ret
    # --

class OntoRecorder:
    def __init__(self, onto: zonto.Onto, conf: MainConf):
        self.onto = onto
        self.conf = conf
        self.pronoun_set, self.prep_set = get_language_specific_sets(conf.language)
        # --
        # collect stats: 1. frame hit, 2. arg with pronoun, 3. arg dist/prep
        self.frame_stat = {}  # Frame.name -> [All, is_verb, is_verb_passive]
        self.role_stat = {}  # id(Role) -> [All, is_pron]; note: this is role-specific
        self.overall_role_stat = {}  # Role.name -> ...
        self.frame_role_stat = {}  # (Frame.name, Role.name) -> ...
        # --

    def record_frame(self, fname: str, is_verb: bool, is_verb_passive: bool):
        _stat = self.frame_stat
        if fname not in _stat:
            _stat[fname] = [0, 0, 0]
        _stat[fname][0] += 1
        _stat[fname][1] += int(is_verb)
        _stat[fname][2] += int(is_verb_passive)
        # --

    def record_role(self, id_role, is_pron: bool):
        _stat = self.role_stat
        if id_role not in _stat:
            _stat[id_role] = [0, 0]
        _stat[id_role][0] += 1
        _stat[id_role][1] += int(is_pron)
        # --

    def record_frs(self, fname: str, rrs):
        for _dist, rr, _prep in rrs:
            for _stat, _key in zip([self.overall_role_stat, self.frame_role_stat], [rr.name, (fname, rr.name)]):
                if _key not in _stat:
                    _stat[_key] = TplRecord()
                _stat[_key].record_one(_prep, _dist)
        # --

    def record(self, inst_stream):
        onto = self.onto
        cc = Counter()
        # --
        # note: special ones for fn
        def _change_fn_label(_label: str):
            if _label.endswith("_2"): return "_zz_ignore_this_one"
            elif _label.endswith("_1"): return (_label[:-3]+'ies') if _label[-3]=='y' else (_label[:-2]+'s')
            else: return _label
        # --
        for inst in inst_stream:
            cc['all_inst'] += 1
            for sent in yield_sents(inst):
                set_ee_heads(sent)
                cc['all_sent'] += 1
                for evt in sent.events:
                    cc['all_frame'] += 1
                    for arg in evt.args:
                        cc['all_arg'] += 1
                    # get information from this inst
                    ff = onto.find_frame(evt.label)
                    if ff is None:
                        cc['all_frame_c0notfound'] += 1
                        cc['all_arg_c0evtnotfound'] += len(evt.args)
                        # zwarn(f"UNK evt: {evt}")  # allow filtered ones
                        continue
                    # --
                    is_verb = (evt.mention.shead_token.upos in ["VERB", "AUX"])
                    is_verb_passive = is_verb and any(z.deplab.endswith(":pass") for z in evt.mention.shead_token.ch_toks)
                    cc['all_frame_c1noverb'] += int(not is_verb)
                    cc['all_frame_c2verb'] += int(is_verb)
                    cc['all_frame_c2verb_passive'] += int(is_verb_passive)
                    self.record_frame(ff.name, is_verb, is_verb_passive)
                    # --
                    # record role's pron info
                    for arg in evt.args:
                        _role, _ = ff.find_role(_change_fn_label(arg.label))
                        if _role is not None:
                            _arg_tok = arg.mention.shead_token
                            _is_pron = (_arg_tok.upos == 'PRON' and str.lower(_arg_tok.word) in self.pronoun_set)
                            if _is_pron:
                                cc['all_arg_pronoun'] += 1
                            self.record_role(id(_role), _is_pron)
                    # --
                    if not is_verb:
                        cc['all_arg_c0evtnotverb'] += len(evt.args)
                        continue  # only get templates from verb ones
                    # --
                    # roles
                    _spec_rr = (0, )  # V itself
                    rrs = [_spec_rr]
                    evt_widx = evt.mention.shead_widx
                    for arg in evt.args:
                        if arg.mention.sent is not evt.mention.sent:
                            cc['all_arg_c1diffsent'] += 1
                            zwarn("Evt & arg not same sentence (this is rare)!")
                            continue
                        _role, _iscore = ff.find_role(_change_fn_label(arg.label))
                        if _role is None:
                            cc['all_arg_c2notfound'] += 1  # note: currently ignore C-*, R-*, V, etc
                            continue
                        _wdist = arg.mention.shead_widx - evt_widx
                        _arg_tok = arg.mention.shead_token
                        _all_preps = [str.lower(z.word) for z in _arg_tok.ch_toks
                                      if (z.deplab=='case' and str.lower(z.word) in self.prep_set)]
                        _prep = _all_preps[0] if len(_all_preps)>0 else None
                        # --
                        if is_verb_passive:
                            if _arg_tok.deplab.endswith("subj:pass"):  # *subj:pass
                                _wdist = 0.95  # place it after!
                            if _prep == 'by':  # very likely to be the original 'agent'
                                cc['all_arg_c3by'] += 1  # simply discard this one!!
                                continue
                        # --
                        cc[f'all_arg_c4inc'] += 1
                        cc[f'all_arg_c4inc_IC{_iscore}'] += 1
                        rrs.append([_wdist, _role, _prep])
                    # --
                    # change to relative ordered positions
                    rrs.sort(key=lambda x: x[0])
                    v_idx = rrs.index(_spec_rr)
                    for ii, rr in enumerate(rrs):
                        if rr != _spec_rr:
                            rr[0] = ii - v_idx
                    rrs.remove(_spec_rr)
                    # --
                    self.record_frs(ff.name, rrs)
                    # --
        # --
        return cc

    # inplace add extra info based on the counts
    def aug_onto(self, onto):
        conf = self.conf
        assert onto is self.onto
        cc = Counter()
        # --
        _DIST_MAP = {
            'nsubj': -2, 'iobj': 2, 'obj': 3, 'obl': 4,
            'compound': -1, 'nmod': 1,
        }  # distance map for syn
        # --
        # stat on frames
        survived_frames = []
        for ff in onto.frames:
            cc['all_frame'] += 1
            counts = self.frame_stat.get(ff.name, [0,0,0])
            cc['all_frame_nocount'] += int(counts[0] == 0)
            cc['all_frame_noverb'] += int(counts[1] == 0)
            cc['all_frame_noverbpass'] += int(counts[2] == 0)
            # --
            # filters
            if counts[0] < conf.rm_frame_thr:
                cc['all_frame_frmthr'] += 1
                continue
            if conf.rm_frame_noverb and counts[1] == 0:
                cc['all_frame_frmnoverb'] += 1
                continue
            if conf.rm_frame_norole and len(ff.core_roles)==0:
                cc['all_frame_frmnorole'] += 1
                continue
            # --
            survived_frames.append(ff)
            cc['all_frame_fok'] += 1
        onto.frames = survived_frames
        onto.refresh_cache()  # reset!!
        # on qw
        for rr in onto.roles:
            cc['all_role'] += 1
            _idr = id(rr)
            counts = self.role_stat.get(_idr, [0,0])
            cc['all_role_nocount'] += int(counts[0] == 0)
            cc['all_role_nopron'] += int(counts[1] == 0)
            if counts[1] / (conf.qw_who_smooth + counts[0]) > conf.qw_who_thr:
                cc['all_role_yeswho'] += 1
                rr.qwords = ['who']  # based on pron; note: assign qwords!
        # on tpl
        _df_dist = 100.  # default dist
        _arg0_dist = -1.33  # average arg0 distance (with ontoC&ewt)
        _trans_src = conf.language if conf.tpl_prep_trans else None
        for ff in onto.frames:
            # first decide core roles
            info_core, info_noncore = [], []
            for cr in ff.core_roles:
                rec0 = self.frame_role_stat.get((ff.name, cr.name), TplRecord())
                rec1 = self.overall_role_stat.get(cr.name, TplRecord())
                rec = rec0.mix_one(rec1, conf.tpl_core_mixratio)
                _dist = _DIST_MAP[cr.name] if conf.tpl_frozen_dist else rec.get_dist(_df_dist)
                _preps = ['in'] if (conf.tpl_in_amloc and cr.name=='ARGM-LOC') else rec.get_preps(conf.tpl_prep_thr, _trans_src)
                info_core.append([_dist, cr.name, _preps])
            # then add noncore roles
            for ncr in ff.noncore_roles:
                rec0 = self.frame_role_stat.get((ff.name, ncr.name), TplRecord())
                rec1 = self.overall_role_stat.get(ncr.name, TplRecord())
                rec = rec0.mix_one(rec1, conf.tpl_noncore_mixratio)
                _dist = _DIST_MAP[ncr.name] if conf.tpl_frozen_dist else rec.get_dist(_df_dist)
                _preps = ['in'] if (conf.tpl_in_amloc and ncr.name=='ARGM-LOC') else rec.get_preps(conf.tpl_prep_thr, _trans_src)
                info_noncore.append([_dist, ncr.name, _preps])
            # --
            info_core.sort()
            info_noncore.sort()
            info_core2 = sorted(info_core, key=lambda x: x[1])  # sort by name!
            if info_core != info_core2:
                cc['all_frame_ok_disagreename'] += 1
            if conf.tpl_core_nameorder:
                info_core = info_core2
                if len(info_core) > 0:
                    info_core[0][0] = -1  # make it the A0!
            # --
            # note: allow multiple cores to be before V!
            a0 = min(info_core, key=(lambda x: abs(-1 - x[0]) if x[0]<0 else (x[0]+100.))) \
                if len(info_core)>0 else None  # closest to '-1'
            if a0 is None:
                zwarn(f"No arg frames?: {ff}:{info_core}")
            if a0 is not None and a0[0] < 0:
                a0_idx = info_core.index(a0)
                if len(a0[-1]) > 0:
                    zwarn(f"Strange A0 with preps: {ff}:{info_core}")
                    a0[-1] = []
            else:
                a0_idx = -1
                zwarn(f"Cannot find a reasonable a0: {ff}:{info_core}")
            noncore0 = [z[1:] for z in info_noncore if z[0]<_arg0_dist]
            noncore1 = [z[1:] for z in info_noncore if z[0]>=_arg0_dist and z[0]<0]
            noncore2 = [z[1:] for z in info_noncore if z[0]>=0]
            # put template: nc0... A* nc1... V A*... nc2...
            template = noncore0 + [aa[1:] for aa in info_core[:a0_idx+1]] + noncore1 \
                       + [[None, []]] + [aa[1:] for aa in info_core[a0_idx+1:]] + noncore2
            ff.template = template
            # --
            # breakpoint()
        # --
        zlog(f"# --\nAfter aug_onto:")
        OtherHelper.printd(cc, try_div=True)
        if conf.debug:
            breakpoint()
        # --

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    onto = zonto.Onto.load_onto(conf.input_onto)
    recorder = OntoRecorder(onto, conf)
    all_cc = Counter()
    _input_files = sum([zglob(z) for z in conf.input_files], [])
    zlog(f"Read from {_input_files}")
    for f in _input_files:
        reader = ReaderGetterConf().get_reader(input_path=f)
        one_cc = recorder.record(reader)
        zlog(f"Read one file {f}: {one_cc}")
        all_cc += one_cc
    zlog(f"# --\nRead all files:")
    OtherHelper.printd(all_cc, try_div=True)
    # --
    recorder.aug_onto(onto)
    # --
    if conf.output_onto:
        default_json_serializer.to_file(onto.to_json(), conf.output_onto, indent=2)
        # also print pp_str format
        inc_noncore = {'Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC'}
        pp_s = onto.pp_str(inc_noncore)
        with zopen(conf.output_onto + ".txt", 'w') as fd:
            fd.write(pp_s)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_onto:... input_files:... output_onto:
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
