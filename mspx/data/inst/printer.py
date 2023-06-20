# --

__all__ = [
    "MyPrettyPrinterConf", "MyPrettyPrinter",
]

from typing import List, Tuple, Union
from mspx.utils import zwarn, Conf, Configurable, wrap_color, Registrable
from .doc import Sent, Token
from .frame import Frame, Mention, ArgLink
from .tree import DepTree

# --
# my inst printer

class MyPrettyPrinterConf(Conf):
    def __init__(self):
        # frame printing
        self.color_frame = "blue"  # frame
        self.color_arg = "red"  # arglink
        self.color_hl = 'magenta'  # highlight
        self.color_set = ['blue', 'red', 'green', 'yellow', 'magenta', 'cyan']
        # sent printing
        self.sent_evt = False  # show 'evt' frames
        self.sent_ef = False  # show 'ef' frames
        self.sent_frame_cates = []  # show other cates
        self.sent_win = 0  # check surrounding sents
        # tree
        self.tree_lab_level = 2

@MyPrettyPrinterConf.conf_rd()
class MyPrettyPrinter(Configurable):
    def __init__(self, conf: MyPrettyPrinterConf = None, **kwargs):
        super().__init__(conf, **kwargs)

    # anns: List[(mention, name, color)]
    def add_anns(self, toks: List[str], anns: List[Tuple[Mention, str, str]]):
        for one_mention, one_name, one_color in sorted(anns, key=lambda x: x[0].get_span()):
            # --
            shidx = one_mention.shead_widx
            if shidx is not None:
                toks[shidx] = toks[shidx] + "~H"
            # --
            widx, wridx = one_mention.widx, one_mention.wridx
            toks[widx] = wrap_color("[", bcolor=one_color) + toks[widx]
            toks[wridx - 1] = toks[wridx - 1] + wrap_color(f"]{one_name}", bcolor=one_color)

    def str_auto(self, inst):
        ff = getattr(self, "str_" + str.lower(inst.__class__.__name__))
        return ff(inst)

    def str_frame(self, f: Frame):
        conf: MyPrettyPrinterConf = self.conf
        # --
        # headline = f"## Frame `{f.mention.text}'[{wrap_color(str(f.type), bcolor=conf.color_frame)}]\n--[S{f.sent.sid}] "
        # headline = f"## Frame `{f.mention.text}'[{str(f.type)}] ({f.info.get('items')})\n"
        if f.mention is None:
            _text = None
        else:
            _text = f.mention.text
        headline = f"## Frame `{_text}'[{str(f.type)}] ({f.info}) ({f.args})"
        # --
        lines = [headline]
        if f.sent is not None:
            all_sents = f.sent.get_sent_win(conf.sent_win)
            for ss in all_sents:
                toks = list(ss.seq_word.vals)
                all_anns = ([(f.mention, f.type, conf.color_frame)] if f.sent is ss else []) + \
                           [(a.arg.mention, a.role, conf.color_arg) for a in f.args if a.arg.sent is ss]
                self.add_anns(toks, all_anns)
                lines.append(" ".join(toks))
        # --
        return "\n".join(lines)

    def str_arglink(self, a: ArgLink):
        conf: MyPrettyPrinterConf = self.conf
        headline = f"## ArgLink {a.role} {a.main.type}->{a.arg.type}"
        str_main = self.str_frame(a.main)
        str_arg = self.str_frame(a.arg)
        return "\n".join([headline, str_main, str_arg])

    def str_token(self, t: Token):
        conf: MyPrettyPrinterConf = self.conf
        headline = f"## Token {t}"
        str_sent = self.str_sent(t.sent, hlspans=[t.widx, t.widx+1])
        return "\n".join([headline, str_sent])

    def str_mention(self, m: Mention):
        conf: MyPrettyPrinterConf = self.conf
        headline = f"## Mention {m}"
        str_sent = self.str_sent(m.sent, hlspans=[m.widx, m.wridx])
        return "\n".join([headline, str_sent])

    def str_sent(self, s: Sent, hlspans=()):
        conf: MyPrettyPrinterConf = self.conf
        # --
        def _str_sent(_s: Sent, _is_center: bool):
            _toks = list(_s.seq_word.vals)
            # highlight span
            for a, b in hlspans:
                for _tidx in range(a, b):
                    _toks[_tidx] = wrap_color(_toks[_tidx], bcolor=conf.color_hl)
            # add frames
            if _is_center:
                cates = list(conf.sent_frame_cates)
                for one_cate, one_flag in zip(["ef", "evt"], [conf.sent_ef, conf.sent_evt]):
                    if one_flag and one_cate not in cates:
                        cates = [one_cate] + cates  # add to front
                for ii, one_cate in enumerate(cates):
                    _items = _s.yield_frames(cates=one_cate)
                    _color = conf.color_set[ii % len(conf.color_set)]
                    self.add_anns(_toks, [(_e.mention, _e.type, _color) for _e in _items])
            # --
            return " ".join(_toks)
        # --
        all_sents = s.get_sent_win(conf.sent_win)
        cur_sid = - all_sents.index(s)
        all_ss = []
        for s2 in all_sents:
            _head = f"[S{cur_sid}]({s2.sid})"
            if s2 is s:
                _head = wrap_color(_head, bcolor='green')
            one_ss = f"{_head} {_str_sent(s2, (s2 is s))}"
            all_ss.append(one_ss)
            cur_sid += 1
        return "\n".join(all_ss)

    def str_deptree(self, trees: Union[DepTree, List[DepTree]]):
        conf: MyPrettyPrinterConf = self.conf
        if not isinstance(trees, (list, tuple)):
            trees = [trees]
        # --
        def _get_upos_tags(_s):
            return _s.seq_upos.vals if _s.seq_upos is not None else (["_"]*len(_s))
        def _get_labs(_t):
            return [":".join(_z.split(":")[:conf.tree_lab_level]) for _z in _t.seq_label.vals]
        # --
        import pandas as pd
        t0 = trees[0]  # treat it as gold
        slen = len(t0.sent)
        gold_upos_tags = _get_upos_tags(t0.sent)
        gold_udep_heads = t0.seq_head.vals
        gold_udep_labels = _get_labs(t0)
        all_cols = [gold_upos_tags, gold_udep_heads, gold_udep_labels]
        for t1 in trees[1:]:
            assert len(t1) == slen
            pred_upos_tags = [wrap_color(t1, bcolor=('red' if t1 != t2 else 'black')) for t1, t2 in
                              zip(_get_upos_tags(t1.sent), gold_upos_tags)]
            pred_udep_heads = [wrap_color(str(t1), bcolor=('red' if t1 != t2 else 'black')) for t1, t2 in
                               zip(t1.seq_head.vals, gold_udep_heads)]
            pred_udep_labels = [wrap_color(t1, bcolor=('red' if t1 != t2 else 'black')) for t1, t2 in
                                zip(_get_labs(t1), gold_udep_labels)]
            all_cols.append(["||"] * slen)
            all_cols.extend([pred_upos_tags, pred_udep_heads, pred_udep_labels])
        # --
        all_cols.append(t0.sent.seq_word.vals)
        data = [[all_cols[j][i] for j in range(len(all_cols))] for i in range(slen)]  # .T
        d = pd.DataFrame(data, index=list(range(1, 1 + slen)))
        ret = d.to_string()
        return ret
