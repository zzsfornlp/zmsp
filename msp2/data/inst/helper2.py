#

# other helpers (like inst-yielder, pprinter, ...)

__all__ = [
    "yield_sents", "yield_sent_pairs", "yield_frames",
    "MyPrettyPrinterConf", "MyPrettyPrinter", "set_ee_heads",
    "QuestionAnalyzer", "SimpleSpanExtender", "SentTpGraph",
]

from typing import List, Union, Tuple
from msp2.utils import wrap_color, Conf, zwarn
from collections import Counter
from .doc import Sent, Doc, Token
from .frame import Mention, Frame, ArgLink
from .tree import HeadFinder, DepSentStruct

# =====
# general treating of doc&sent

# yield sents
def yield_sents(insts: List[Union[Doc, Sent]]):
    if isinstance(insts, (Doc, Sent, Frame)):
        insts = [insts]
    for inst in insts:
        if isinstance(inst, Doc):
            yield from inst.sents
        elif isinstance(inst, Sent):
            yield inst
        elif isinstance(inst, Frame):
            yield inst.sent  # go up!!
        else:
            raise NotImplementedError(f"Error: bad input {type(inst)}")

# check and iter sent pairs
def yield_sent_pairs(gold_insts: List[Union[Doc, Sent]], pred_insts: List[Union[Doc, Sent]]):
    # assert len(gold_insts) == len(pred_insts), "Err: Input size mismatch!"
    if len(gold_insts) != len(pred_insts):
        zwarn(f"Input size mismatch: {len(gold_insts)} vs {len(pred_insts)}")
    for one_g, one_p in zip(gold_insts, pred_insts):
        if isinstance(one_g, Sent) and isinstance(one_p, Sent):
            yield (one_g, one_p)
        elif isinstance(one_g, Doc) and isinstance(one_p, Doc):
            # assert one_g.id == one_p.id, "Err: DocID mismatch!"
            if one_g.id != one_p.id:
                zwarn(f"DocID mismatch: {one_g.id} vs {one_p.id}")
            assert len(one_g.sents) == len(one_p.sents), "Err: Doc sent-num mismatch!!"
            for one_gs, one_ps in zip(one_g.sents, one_p.sents):
                yield (one_gs, one_ps)
        else:
            raise RuntimeError(f"Err: Different/UNK types to eval {type(one_g)} vs {type(one_p)}")

# yield frames
def yield_frames(insts):
    if isinstance(insts, (Doc, Sent, Frame)):
        insts = [insts]
    for inst in insts:
        if isinstance(inst, Doc):
            for sent in inst.sents:
                if sent.events is not None:
                    yield from sent.events
        elif isinstance(inst, Sent):
            if inst.events is not None:
                yield from inst.events
        elif isinstance(inst, Frame):
            yield inst
        else:
            raise NotImplementedError(f"Error: bad input {type(inst)}")
    # --

# pretty printer
class MyPrettyPrinterConf(Conf):
    def __init__(self):
        # frame printing
        self.color_frame = "blue"
        self.color_arg = "red"
        self.color_highlight = 'MAGENTA'
        # sent printing
        self.sent_evt = False
        self.sent_ef = False
        self.sent_win = 0  # check surrounding sents
        self.sent_hlspan0 = 0  # highlight span left [)
        self.sent_hlspan1 = 0  # highlight span right [)
        # tree printing


class MyPrettyPrinter:
    @staticmethod
    def str_frame(f: Frame, conf: MyPrettyPrinterConf = None, **kwargs):
        if f is None:
            return "[None]"
        # --
        conf = MyPrettyPrinterConf.direct_conf(conf, **kwargs)
        # headline = f"## Frame `{f.mention.text}'[{wrap_color(str(f.type), bcolor=conf.color_frame)}]\n--[S{f.sent.sid}] "
        # headline = f"## Frame `{f.mention.text}'[{str(f.type)}] ({f.info.get('items')})\n"
        headline = f"## Frame `{f.mention.text}'[{str(f.type)}] ({f.info}) ({f.args})\n"
        # --
        lines = []
        all_sents = []
        ss = f.sent
        for ii in range(conf.sent_win):
            ss = ss.prev_sent
            if ss is None: break
            all_sents.append(ss)
        all_sents.reverse()
        all_sents.append(f.sent)
        ss = f.sent
        for ii in range(conf.sent_win):
            ss = ss.next_sent
            if ss is None: break
            all_sents.append(ss)
        # --
        for ss in all_sents:
            toks = list(ss.seq_word.vals)
            all_anns = ([(f.mention, f.type, conf.color_frame)] if f.sent is ss else []) + \
                       [(a.arg.mention, a.role, conf.color_arg) for a in f.args if a.arg.sent is ss]
            MyPrettyPrinter.add_anns(toks, all_anns)
            # --
            # further add res_cand and res_split
            res_cand, res_split = f.info.get("res_cand"), f.info.get("res_split")
            if res_cand is not None and res_split is not None:
                res_split = res_split + [1.]  # put an ending
                cidx = 1
                for ii, vv in enumerate(res_cand):
                    if vv>0:
                        pp = "~C" + ("|" if res_split[cidx]>0 else "-")
                        toks[ii] = toks[ii] + pp
                        cidx += 1
            lines.append(" ".join(toks))
        # --
        return headline + "\n".join(lines)

    @staticmethod
    def str_alink(a: ArgLink, conf: MyPrettyPrinterConf = None, **kwargs):
        if a is None:
            return "[None]"
        # --
        conf = MyPrettyPrinterConf.direct_conf(conf, **kwargs)
        headline = f"## ArgLink {a.role} {a.main.type}->{a.arg.type}"
        str_main = MyPrettyPrinter.str_frame(a.main, conf)
        str_arg = MyPrettyPrinter.str_frame(a.arg, conf)
        return "\n".join([headline, str_main, str_arg])

    @staticmethod
    def str_token(t: Token, conf: MyPrettyPrinterConf = None, **kwargs):
        if t is None:
            return "[None]"
        # --
        conf = MyPrettyPrinterConf.direct_conf(conf, **kwargs)
        # --
        return MyPrettyPrinter.str_sent(t.sent, conf, sent_hlspan0=t.widx, sent_hlspan1=t.widx+1)

    @staticmethod
    def str_mention(m: Mention, conf: MyPrettyPrinterConf = None, **kwargs):
        if m is None:
            return "[None]"
        # --
        conf = MyPrettyPrinterConf.direct_conf(conf, **kwargs)
        # --
        return MyPrettyPrinter.str_sent(m.sent, conf, sent_hlspan0=m.widx, sent_hlspan1=m.wridx)

    @staticmethod
    def str_sent(s: Sent, conf: MyPrettyPrinterConf = None, **kwargs):
        if s is None:
            return "[None]"
        # --
        conf = MyPrettyPrinterConf.direct_conf(conf, **kwargs)
        # --
        def _str_sent(_s: Sent, _add_trgs: bool):
            _toks = list(_s.seq_word.vals)
            # highlight span
            for _tidx in range(conf.sent_hlspan0, conf.sent_hlspan1):
                _toks[_tidx] = wrap_color(_toks[_tidx], bcolor=conf.color_highlight)
            # others
            if _add_trgs:
                for _items, _flag, _color in zip(
                        [_s.events, _s.entity_fillers], [conf.sent_evt, conf.sent_ef], [conf.color_frame, conf.color_arg]):
                    if not _flag: continue
                    MyPrettyPrinter.add_anns(_toks, [(_e.mention, _e.type, _color) for _e in _items])
            return " ".join(_toks)
        # --
        if s.doc is not None:
            sid = s.sid
            pre_sents = s.doc.sents[max(0,sid-conf.sent_win):sid]
            post_sents = s.doc.sents[sid+1:sid+conf.sent_win+1]
        else:
            pre_sents = post_sents = []
        cur_sid = -len(pre_sents)
        all_ss = []
        for s2 in pre_sents+[s]+post_sents:
            one_ss = f"[S{cur_sid}]({s2.sid}) {_str_sent(s2, True)}"
            all_ss.append(one_ss)
            cur_sid += 1
        return "\n".join(all_ss)

    # another pretty printer for sent
    @staticmethod
    def str_sent2(s: Sent, conf: MyPrettyPrinterConf = None, **kwargs):
        if s is None:
            return "[None]"
        # --
        conf = MyPrettyPrinterConf.direct_conf(conf, **kwargs)
        item_map = {}
        # --
        def _str_sent(_s: Sent, _is_center: bool):
            _toks = list(_s.seq_word.vals)
            _lines = []
            # highlight span
            for _tidx in range(conf.sent_hlspan0, conf.sent_hlspan1):
                _toks[_tidx] = wrap_color(_toks[_tidx], bcolor=conf.color_highlight)
            _ef_toks = _toks.copy()
            _evt_toks = _toks.copy()
            # check efs
            if _s.entity_fillers is not None:
                for _e in _s.entity_fillers:
                    assert id(_e) not in item_map
                    item_map[id(_e)] = f"E{len(item_map)}"
                    MyPrettyPrinter.add_anns(_ef_toks, [(_e.mention, item_map[id(_e)], conf.color_arg)])
                    _lines.append(f"[{wrap_color(item_map[id(_e)], bcolor=conf.color_arg)}]: {_e} {_e.info}")
            # check evts
            if _is_center:
                if _s.events is not None:
                    for _e in _s.events:
                        assert id(_e) not in item_map
                        item_map[id(_e)] = f"E{len(item_map)}"
                        MyPrettyPrinter.add_anns(_evt_toks, [(_e.mention, item_map[id(_e)], conf.color_frame)])
                        _lines.append(f"[{wrap_color(item_map[id(_e)], bcolor=conf.color_frame)}]: {_e} {_e.info}")
                        _tmps = [f"{a.label}->[{wrap_color(item_map.get(id(a.arg), '??'), bcolor=conf.color_arg)}]"
                                 for a in _e.args]
                        _lines.append("[args]=>" + "  ".join(_tmps))
            # --
            _lines = [" ".join(_ef_toks), " ".join(_evt_toks)] + _lines
            return "\n".join(_lines)
        # --
        if s.doc is not None:
            sid = s.sid
            pre_sents = s.doc.sents[max(0,sid-conf.sent_win):sid]
            post_sents = s.doc.sents[sid+1:sid+conf.sent_win+1]
        else:
            pre_sents = post_sents = []
        cur_sid = -len(pre_sents)
        all_ss = []
        for s2 in pre_sents+[s]+post_sents:
            one_ss = f"[S{cur_sid}]({s2.sid}) {_str_sent(s2, s2 is s)}"
            all_ss.append(one_ss)
            cur_sid += 1
        return "\n".join(all_ss)

    # some helpers
    @staticmethod
    def add_anns(toks: List[str], anns: List[Tuple[Mention, str, str]]):
        for one_mention, one_name, one_color in sorted(anns, key=lambda x: x[0].get_span()):
            # --
            shidx = one_mention.shead_widx
            if shidx is not None:
                toks[shidx] = toks[shidx] + "~H"
            info = one_mention.info
            # if info.get("widxes0"):
            #     for ii in info["widxes0"]:
            #         toks[ii] = toks[ii] + "~0"
            if info.get("widxes1"):
                for ii in info["widxes1"]:
                    toks[ii] = toks[ii] + "~R"
            # --
            widx, wridx = one_mention.widx, one_mention.wridx
            toks[widx] = wrap_color("[", bcolor=one_color) + toks[widx]
            toks[wridx - 1] = toks[wridx - 1] + wrap_color(f"]{one_name}", bcolor=one_color)

    @staticmethod
    def str_anns(toks: List[str], anns: List[Tuple[Mention, str, str]]):
        toks2 = toks.copy()
        MyPrettyPrinter.add_anns(toks, anns)
        return " ".join(toks2)

    @staticmethod
    def str_fnode(s: Sent, fnode, conf: MyPrettyPrinterConf = None, **kwargs):
        if s is None:
            return "[None]"
        conf = MyPrettyPrinterConf.direct_conf(conf, **kwargs)
        # --
        _lines = []
        _state = [0]  # current layer?
        _hit = Counter()
        def _tmp_visit(_node):
            _cl = _state[0]
            _cc = _hit[id(_node)]
            if _cc == 0:
                _state[0] += 1
                _hit[id(_node)] += 1
                # current line
                _node_widx, _node_wlen = _node.full_span
                _toks = list(s.seq_word.vals)  # copy the tokens
                if _node.widx >= 0:
                    _toks[_node.widx] = wrap_color(_toks[_node.widx], bcolor=conf.color_frame)
                for _ch in _node.chs:  # annotate chs
                    _ch_widx, _ch_wlen = _ch.full_span
                    _wridx = _ch_widx + _ch_wlen
                    _toks[_ch_widx] = wrap_color('[', bcolor=conf.color_arg) + _toks[_ch_widx]
                    _toks[_wridx-1] = _toks[_wridx-1] + wrap_color(f"]{_ch.d_label}", bcolor=conf.color_arg)
                _mid = "| " * (_cl-1) + "|-" if _cl>0 else ""
                _line = f"L{_cl:02d}" + f" ({_node.l_type:5s}) " + _mid + " ".join(_toks[_node_widx:_node_widx+_node_wlen])
                _lines.append(_line)
            else:
                _state[0] -= 1
        # --
        fnode.visit(_tmp_visit, do_pre=True, do_post=True)
        return "\n".join(_lines)

# shortcut: set heads for doc/sent's ef and evt
def set_ee_heads(insts: List):
    hf_ef, hf_evt = HeadFinder("NOUN"), HeadFinder("VERB")
    for sent in yield_sents(insts):
        if sent.entity_fillers is not None:
            for ef in sent.entity_fillers:
                hf_ef.set_head_for_mention(ef.mention)
        if sent.events is not None:
            for evt in sent.events:
                hf_evt.set_head_for_mention(evt.mention)

# --
# analyze question
class QuestionAnalyzer:
    def __init__(self):
        self.q_words = {'what', 'when', 'where', 'who', 'which', 'how', 'why', 'whose', 'whom'}
        self.q_replacements = {
            '_default': [None],
            'when': ['at time', None],
            'where': ['in place', None],
            'how': ['by', None],
            'why': ['because', None],
            'whose': [None, "'s"],
        }
        self.ordering = {'mark': -1, 'csubj': 0, 'nsubj': 0, 'aux': 1, 'cop': 1, 'head': 2,
                         'obj': 3, 'iobj': 3, 'ccomp': 3, 'xcomp': 3, 'obl': 4}

    def locate_qword(self, qsent: Sent):
        # locate the first question word
        lower_toks = [z.lower() for z in qsent.seq_word.vals]
        q_widx = -1
        for ii, vv in enumerate(lower_toks):
            if vv in self.q_words:
                q_widx = ii
                break  # note: simply find the first one!
        return q_widx

    def analyze_question(self, qsent: Sent):
        q_widx = self.locate_qword(qsent)
        if q_widx < 0:  # not found!
            return {'q_widx': -1, 'q_word': "UNK", 'q_sig': "UNK", 'q_sigD': "UNK"}
        q_word = qsent.seq_word.vals[q_widx].lower()
        tree = qsent.tree_dep
        struct = DepSentStruct(tree)
        # --
        # go above
        q_path = []
        _cur = struct.widx2nodes[q_widx][0]
        while _cur.l_type != 'pred' and _cur.par is not None:
            q_path.append(_cur)
            _cur = _cur.par
        # --
        q_sig = q_word + "|" + '|'.join([z.l_label for z in q_path])
        q_sigD = q_word + "|" + '|'.join([tree.seq_label.vals[z] for z in tree.get_spine(q_widx)])
        return {'q_widx': q_widx, 'q_word': q_word, 'q_sig': q_sig, 'q_sigD': q_sigD}

    def question2template(self, qsent: Sent, repl='X'):
        q_widx = self.locate_qword(qsent)
        if q_widx < 0:  # cannot convert!
            return None
        # --
        tree = qsent.tree_dep
        struct = DepSentStruct(tree)
        q_node = struct.widx2nodes[q_widx][0]
        sent_toks = list(qsent.seq_word.vals)  # copy it
        sent_toks[0] = sent_toks[0][0].lower() + sent_toks[0][1:]  # todo(+N): lowercase anyway ...
        # --
        hit_pred, hit_nom = False, False
        cur_toks = [(repl if z is None else z) for z in self.q_replacements.get(sent_toks[q_widx], [None])]
        cur_node = q_node
        while cur_node.par is not None:
            _par = cur_node.par
            _chs = list(_par.chs)  # copy
            # --
            new_toks = None
            if not hit_nom and _par.l_type == 'nom':
                # move case marker
                first_case = None
                _new_chs = []
                for ch in _chs:
                    if first_case is None and ch.l_label == 'case':
                        first_case = ch
                    else:
                        _new_chs.append(ch)
                _chs = ([first_case] if first_case is not None else []) + _new_chs
                hit_nom = True
            elif not hit_pred and _par.l_type == 'pred':
                # might need to transform
                # pick the first subj out
                all_subjs = [c for c in _chs if c is not cur_node and c.l_label in ['csubj', 'nsubj']]
                if len(all_subjs) > 0:  # otherwise no need to change!
                    first_subj = all_subjs[0]
                    # --
                    reordered_chs = [c for c in _chs if c not in [first_subj, cur_node]]
                    # insert subj before the first certain ...
                    ins_subj = [_ii for _ii, _cc in enumerate(reordered_chs) if _cc.l_label in ['aux', 'cop', 'head']]
                    ins_subj = 0 if len(ins_subj)==0 else ins_subj[0]
                    reordered_chs.insert(ins_subj, first_subj)
                    # insert q_node after last certain but before certain
                    _check_types = ['aux', 'cop', 'head'] if cur_node.l_label in ['obj', 'iobj', 'head'] \
                        else ['aux', 'cop', 'obj', 'iobj', 'head', 'ccomp', 'xcomp', 'obl']
                    ins_cur = [_ii for _ii, _cc in enumerate(reordered_chs) if _cc.l_label in _check_types]
                    if len(ins_cur) == 0:
                        ins_cur = [_ii for _ii in reversed(range(len(reordered_chs))) if reordered_chs[_ii].l_label != 'punct']
                    ins_cur = len(reordered_chs) if len(ins_cur)==0 else (ins_cur[-1]+1)  # insert after
                    reordered_chs.insert(ins_cur, cur_node)
                    # --
                    # also change aux / main-verb
                    new_toks = []
                    first_aux_word = None
                    for _ch in reordered_chs:
                        _span = _ch.full_span
                        if first_aux_word is None and _ch.l_label == 'aux':
                            first_aux_word = sent_toks[_span[0]]
                            if first_aux_word in ['do', 'does', 'did']:
                                continue  # delete this one
                        if _ch is cur_node:
                            new_toks.extend(cur_toks)
                        else:  # todo(+N): also need to change the verb!!
                            new_toks.extend(sent_toks[_span[0]:sum(_span)])
                # --
                hit_pred = True
            if new_toks is None:  # simply use original ones!
                new_toks = []
                for _ch in _chs:
                    _span = _ch.full_span
                    new_toks.extend(cur_toks if _ch is cur_node else sent_toks[_span[0]:sum(_span)])
            # --
            cur_toks = new_toks
            cur_node = cur_node.par
        # --
        if len(cur_toks) > 0:
            cur_toks[0] = cur_toks[0][0].upper() + cur_toks[0][1:]  # Uppercase the first one!
            if cur_toks[-1] == '?':
                cur_toks[-1] = '.'
        return cur_toks

    def template2question(self, tsent: Sent, x_widx: int, q_word='what'):
        raise NotImplementedError()  # todo(+N)

# --
# simple span extender (according to certain ud types)
class SimpleSpanExtender:
    @staticmethod
    def get_extender(item_type: str, **kwargs):
        if item_type == 'evt':
            kwargs0 = {'ch_types': {'compound:prt'}}
        elif item_type == 'ef':
            kwargs0 = {'ch_types': {'fixed', 'flat'}, 'par_types': {'fixed', 'flat'}, 'upper_coumpound': True}
        elif item_type == 'ef2':
            kwargs0 = {'ch_types': {"fixed", "flat", "compound", "nummod", "amod"},  # more types
                       'par_types': {'fixed', 'flat'}, 'upper_coumpound': True}
        else:
            raise NotImplementedError()
        kwargs0.update(kwargs)
        ret = SimpleSpanExtender(**kwargs0)
        return ret

    def __init__(self, ch_types=None, par_types=None, upper_coumpound=False, allow_discontinuous=False, include_one_hyphen=False):
        self.ch_types = set() if ch_types is None else ch_types
        self.par_types = set() if par_types is None else par_types
        self.upper_coumpound = upper_coumpound
        self.allow_discontinuous = allow_discontinuous
        self.include_one_hyphen = include_one_hyphen

    def check_type(self, t: str, w: str, upper_coumpound: bool, types):
        t0 = t.split(":")[0]
        if t in types or t0 in types:
            return True
        if upper_coumpound and t0=='compound' and (str.isupper(w[:1]) or str.isdigit(w[:1])):
            return True  # note: uppercase or digit!
        return False

    def extend(self, sent: Sent, widx: int, wlen: int):
        # --
        if wlen <= 0:  # nothing to expand
            return widx, wlen
        # --
        # extend it?
        words = sent.seq_word.vals
        dep_tree = sent.tree_dep
        dep_heads, dep_labs, dep_chs, dep_range = \
            dep_tree.seq_head.vals, dep_tree.seq_label.vals, dep_tree.chs_lists[1:], dep_tree.ranges  # [m]
        # --
        extra_widxes = set()
        extra_puncts = set()  # used for checking continuous!
        for ww in range(widx, widx+wlen):
            _upper_coumpound = self.upper_coumpound and str.isupper(words[ww][:1])  # first this one should be upper!
            # check par type?
            if dep_heads[ww]>0 and self.check_type(
                    dep_labs[ww], words[dep_heads[ww]-1], _upper_coumpound, self.par_types):  # include par
                ww = dep_heads[ww] - 1
                extra_widxes.add(ww)
            # check ch type?
            for ch in dep_chs[ww]:
                if self.check_type(dep_labs[ch], words[ch], _upper_coumpound, self.ch_types):  # include ch
                    _ch0, _ch1 = dep_range[ch]
                    extra_widxes.update(range(_ch0, _ch1+1))
                    # extra_widxes.add(ch)
                if dep_labs[ch] == 'punct':
                    extra_puncts.add(ch)
            # check one hyphen
            if self.include_one_hyphen:
                # previous one?
                if ww>=2 and words[ww-1]=='-' and (ww-2) in dep_chs[ww]:
                    extra_widxes.update([ww-1, ww-2])
                # after one?
                if ww+2<len(sent) and words[ww+1]=='-' and (ww+2) in dep_chs[ww]:
                    extra_widxes.update([ww+1, ww+2])
        # --
        extra_widxes.update(range(widx, widx+wlen))
        min_left, max_right = min(extra_widxes), max(extra_widxes)
        if self.allow_discontinuous:
            _left, _right = min_left, max_right
        else:
            _left = widx
            while _left > min_left:
                if _left not in extra_widxes and _left not in extra_puncts:
                    _left += 1
                    break
                _left -= 1
            _right = widx + wlen - 1
            while _right < max_right:
                if _right not in extra_widxes and _right not in extra_puncts:
                    _right -= 1
                    break
                _right += 1
        return (_left, _right-_left+1)

    def extend_mention(self, mention: Mention, inplace=True):
        _widx, _wlen = mention.get_span()
        if not inplace:  # copy another one!
            mention = Mention.create(mention.sent, _widx, _wlen)
        _new_widx, _new_wlen = self.extend(mention.sent, _widx, _wlen)
        if (_new_widx, _new_wlen) != (_widx, _wlen):
            mention.set_span(_new_widx, _new_wlen)
        return mention

# --
# token pair graph (syn + sem arg)
class SentTpEdge:
    def __init__(self, sent: Sent, i0: int, i1: int, label: str, priority: float, extra_obj):
        self.sent = sent
        self.i0, self.i1 = i0, i1
        self.label = label
        self.priority = priority
        self.extra_obj = extra_obj  # probably another object

    @property
    def token0(self): return self.sent.tokens[self.i0]

    @property
    def token1(self): return self.sent.tokens[self.i1]

    def __repr__(self):
        return f"{self.token0}--({self.label})->{self.token1}"

class SentTpGraph:
    def __init__(self, sent: Sent, add_syn=True, add_sem=True, **kwargs):
        self.sent = sent
        self.edges = [[] for _ in range(len(sent))]  # links from each token
        self.caches = {}  # (i0->i1): List[Edge]
        # --
        # add syn (priority=0.)
        if add_syn:
            dep_heads, dep_labels = sent.tree_dep.seq_head.vals, [z.split(":")[0] for z in sent.tree_dep.seq_label.vals]
            for m, (h, lab) in enumerate(zip(dep_heads, dep_labels)):
                if h <= 0: continue  # skip root
                self.add_link(h-1, m, f"syn:{lab}", 0., None)  # head to mod
                self.add_link(m, h-1, f"syn:^{lab}", 0., None)  # mod reversed to head
        # --
        # add sem (priority=1.)
        if add_sem:
            set_ee_heads([sent])
            for evt in sent.events:
                evt_hwidx = evt.mention.shead_widx
                for arg in evt.args:
                    arg_hwidx = arg.arg.mention.shead_widx
                    self.add_link(evt_hwidx, arg_hwidx, f"sem:{arg.label}", 1., arg)  # pred to arg
                    self.add_link(arg_hwidx, evt_hwidx, f"sem:^{arg.label}", 1., arg)  # arg reversed to pred
        # --

    def add_link(self, i0: int, i1: int, label: str, priority: float, extra_obj):
        edge = SentTpEdge(self.sent, i0, i1, label, priority, extra_obj)
        self.edges[i0].append(edge)
        self.caches.clear()  # clean cache!

    def shortest_path(self, i0: int, i1: int, pfilter=None):
        _key = (i0, i1)
        res = self.caches.get(_key)
        if res is None:
            # simply do BFS search
            import queue
            visited = {i0}
            q = queue.Queue()
            q.put((i0, []))  # (node, edges)
            while not q.empty():
                _n0, _es = q.get()
                if (i0, _n0) not in self.caches:  # by the way, add cache for intermediate ones
                    self.caches[(i0, _n0)] = _es
                if _n0 == i1:  # todo(+1): could have return when putting this!
                    res = _es
                    break  # ok for this run!
                for extra_edge in sorted(self.edges[_n0], key=lambda x: -x.priority):
                    if pfilter is not None and not pfilter(extra_edge):
                        continue
                    _n1 = extra_edge.i1
                    if _n1 not in visited:
                        q.put((_n1, _es+[extra_edge]))  # put into queue
                        visited.add(_n1)
            # --
            assert res is not None, "With dep-tree, we should have everything connected!"
        return res
