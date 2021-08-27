#

# other helpers (like inst-yielder, pprinter, ...)

__all__ = [
    "yield_sents", "yield_sent_pairs", "yield_frames",
    "MyPrettyPrinterConf", "MyPrettyPrinter", "set_ee_heads",
]

from typing import List, Union, Tuple
from msp2.utils import wrap_color, Conf
from .doc import Sent, Doc, Token
from .frame import Mention, Frame, ArgLink
from .tree import HeadFinder

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
    assert len(gold_insts) == len(pred_insts), "Err: Input size mismatch!"
    for one_g, one_p in zip(gold_insts, pred_insts):
        if isinstance(one_g, Sent) and isinstance(one_p, Sent):
            yield (one_g, one_p)
        elif isinstance(one_g, Doc) and isinstance(one_p, Doc):
            assert one_g.id == one_p.id, "Err: DocID mismatch!"
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

class MyPrettyPrinter:
    @staticmethod
    def str_frame(f: Frame, conf: MyPrettyPrinterConf = None, **kwargs):
        if f is None:
            return "[None]"
        # --
        conf = MyPrettyPrinterConf.direct_conf(conf, **kwargs)
        # headline = f"## Frame `{f.mention.text}'[{wrap_color(str(f.type), bcolor=conf.color_frame)}]\n--[S{f.sent.sid}] "
        headline = f"## Frame `{f.mention.text}'[{str(f.type)}]\n"
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
