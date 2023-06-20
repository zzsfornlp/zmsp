#

# conll styled files (mostly from msp2)

__all__ = [
    "ConllFormator", "ConllFormatorConf", "ConllHelper",
]

from typing import List
from .base import DataFormator
from mspx.utils import Registrable, Conf, zwarn
from mspx.data.inst import DataInst, Doc, Sent, Frame, ArgLink

# --
class ConllFormatorConf(Conf):
    def __init__(self):
        # separator
        self.sep_in = None  # by default any space
        self.sep_out = "\t"
        # fields (first according to conll12)
        self.f_doc = None  # Document ID
        self.f_part = None  # Part number
        self.f_widx = None  # Word number
        self.f_word = None  # Word itself
        self.f_xpos = None  # Part-of-Speech
        self.f_parse = None  # Parse bit
        self.combine_lemma_id = False  # combine lemma&ID in one field (f_pred)
        self.f_pred = None  # Predicate lemma
        self.f_pred_id = None  # Predicate Frameset ID
        self.f_sense = None  # Word sense
        self.f_speaker = None  # Speaker/Author
        self.f_ne = None  # Named Entities
        self.arg_is_dep = False  # is dep-arg (wlen==1)
        self.f_arg_start = None  # ->N: Predicate Arguments
        self.f_coref = None  # usually neg: Coreference
        # extra for dep
        self.f_upos = None  # UD pos
        self.f_dep_head = None  # Dep Head
        self.f_dep_label = None  # Dep Label
        # extra for other info: only for reading purpose!!
        self.f_others = []  # other infos
        # --
        # special ones
        self.widx_start = 0  # 0 or 1?
        self.pred_nil_vals = "-_"
        self.pred_cate = "evt"
        self.arg_cate = "ef"
        self.ignore_line = '#'  # ignore lines starting with this!
        self.out_nil_val = '-'
        # --

    @property
    def num_extra_field(self):
        # currently only coref can have NEG!
        return 0 if self.f_coref is None or self.f_coref>=0 else 1

# --
# @DataFormator.rd("??")  # note: no default one!!
class ConllFormator(DataFormator):
    def __init__(self, conf: ConllFormatorConf, **kwargs):
        conf = ConllFormatorConf.direct_conf(conf, **kwargs)
        # --
        self.conf = conf

    def to_obj(self, inst: DataInst) -> str:
        conf: ConllFormatorConf = self.conf
        # --
        if isinstance(inst, Doc):
            ss = [self.to_obj(s) for s in inst.sents]
            return "\n".join(ss)
        # --
        all_cols = {}  # int -> List[str]
        sent: Sent = inst
        # -> write the conll fields
        slen = len(sent)
        # doc id
        if conf.f_doc is not None:
            f_doc = int(conf.f_doc)
            doc_id = sent.info.get("doc_id", "[unk]")
            all_cols[f_doc] = [doc_id] * slen
        # part id
        if conf.f_part is not None:
            f_part = int(conf.f_part)
            part_id = sent.info.get("part_id", 0)
            all_cols[f_part] = [str(part_id)] * slen
        # word idx
        if conf.f_widx is not None:
            f_widx = int(conf.f_widx)
            widxes = [str(x) for x in range(conf.widx_start, conf.widx_start+slen)]
            all_cols[f_widx] = widxes
        # words
        if conf.f_word is not None:
            f_word = int(conf.f_word)
            all_cols[f_word] = sent.seq_word.vals  # directly borrowing since read only!
        # pred + predid + args
        if conf.f_pred is not None:  # frames
            f_pred = int(conf.f_pred)
            f_pred_id = int(conf.f_pred_id)
            ret_preds, ret_pred_ids, ret_frames = ConllHelper.put_preds(sent.yield_frames(cates=conf.pred_cate), slen)
            if conf.combine_lemma_id:  # combine them into one field!!
                all_cols[f_pred], all_cols[f_pred_id] = [f"{a}.{b}" for a,b in zip(ret_preds, ret_pred_ids)], ["-"]*slen
            else:
                all_cols[f_pred], all_cols[f_pred_id] = ret_preds, ret_pred_ids
            # args
            if conf.f_arg_start is not None:
                f_arg_start = int(conf.f_arg_start)
                for one_frame in ret_frames:
                    if one_frame is None: continue
                    _put_f = ConllHelper.put_args_dep if conf.arg_is_dep else ConllHelper.put_args
                    all_cols[f_arg_start] = _put_f([a for a in one_frame.args if a.arg.sent is sent], slen)
                    # add one field further
                    f_arg_start += 1
        # todo(+W): currently getting others from info
        for info_name, f_field in zip(
                ["xpos", "parse", "sense", "speaker", "ne", "coref"],
                [conf.f_xpos, conf.f_parse, conf.f_sense, conf.f_speaker, conf.f_ne, conf.f_coref]):
            if f_field is not None:
                _tmp_idx = int(f_field)
                all_cols[_tmp_idx] = sent.info.get(info_name, ["[unk]"]*slen)  # note: can be absent
        # --
        # finally UD related fields
        # upos
        if conf.f_upos is not None:
            f_upos = int(conf.f_upos)
            all_cols[f_upos] = sent.seq_upos.vals
        # dep
        if conf.f_dep_head is not None:
            f_dep_head = int(conf.f_dep_head)
            all_cols[f_dep_head] = [str(z) for z in sent.tree_dep.seq_head.vals]
            if conf.f_dep_label is not None:
                f_dep_label = int(conf.f_dep_label)
                all_cols[f_dep_label] = sent.tree_dep.seq_label.vals
        # ==
        # ==
        # finally combine together
        all_keys = sorted(all_cols.keys())
        num_field = all_keys[-1] - min(0, all_keys[0]) + 1
        all_fields = [[conf.out_nil_val]*num_field for _ in range(slen)]
        for k in all_keys:
            vs = all_cols[k]
            for i in range(slen):
                all_fields[i][k] = vs[i]
        return "\n".join([conf.sep_out.join(z) for z in all_fields]) + "\n"

    def from_obj(self, s: str) -> DataInst:
        conf: ConllFormatorConf = self.conf
        # --
        lines = s.rstrip().split("\n")
        if conf.ignore_line:
            lines = [z for z in lines if not z.startswith(conf.ignore_line)]
        all_fields = [line.split(conf.sep_in) for line in lines]
        num_col = 0
        if len(all_fields) > 0:
            num_col = len(all_fields[0])
            # assert all(len(z)<=num_col for z in all_fields)
            for z in all_fields:
                if len(z) != num_col:
                    zwarn(f"Line length not match ({len(z)} vs {num_col})")
        # --
        sent = Sent()  # make an empty one!!
        doc = sent.make_singleton_doc()
        # -> read in conll fields
        # doc id
        if conf.f_doc is not None:
            f_doc = int(conf.f_doc)
            doc_id = ConllHelper.get_f_doc([z[f_doc] for z in all_fields])
            sent.info["doc_id"] = doc_id  # temporaly put it here!
        # part id
        if conf.f_part is not None:
            f_part = int(conf.f_part)
            part_id = ConllHelper.get_f_doc([z[f_part] for z in all_fields])
            sent.info["part_id"] = part_id
        # word idx
        if conf.f_widx is not None:
            f_widx = int(conf.f_widx)
            valids = ConllHelper.get_f_widx([z[f_widx] for z in all_fields], conf.widx_start)
            # note: filtering lines!!
            all_fields = [z for z,v in zip(all_fields, valids) if v]
        # words
        if conf.f_word is not None:
            f_word = int(conf.f_word)
            words = [z[f_word] for z in all_fields]
            sent.build_words(words)
        # pred + predid + args
        if conf.f_pred is not None:  # frames
            f_pred = int(conf.f_pred)
            f_pred_id = int(conf.f_pred_id)
            preds = ConllHelper.get_preds(
                [z[f_pred] for z in all_fields], [z[f_pred_id] for z in all_fields], conf.combine_lemma_id,
                nil_vals=conf.pred_nil_vals)
            new_frames = [sent.make_frame(
                p_widx, 1, label=p_lab, cate=conf.pred_cate) for p_widx, p_lab in preds]  # note: wlen==1
            # args?
            if conf.f_arg_start is not None:
                f_arg_start = int(conf.f_arg_start)
                # --
                if num_col - conf.num_extra_field - f_arg_start != len(new_frames):
                    zwarn(f"Unequal num of args: {num_col - conf.num_extra_field - f_arg_start} vs {len(new_frames)}")
                # --
                for one_new_frame in new_frames:
                    # read args
                    _get_f = ConllHelper.get_f_args_dep if conf.arg_is_dep else ConllHelper.get_f_args
                    args = _get_f(one_new_frame.mention.widx, [z[f_arg_start] for z in all_fields])
                    for a_widx, a_wlen, a_lab in args:
                        new_ef = sent.make_frame(a_widx, a_wlen, label="UNK", cate=conf.arg_cate)
                        one_new_frame.add_arg(new_ef, a_lab)
                    # add one field further
                    f_arg_start += 1
        # todo(+W): currently putting others at info
        for info_name, f_field in zip(
                ["xpos", "parse", "sense", "speaker", "ne", "coref"],
                [conf.f_xpos, conf.f_parse, conf.f_sense, conf.f_speaker, conf.f_ne, conf.f_coref]):
            if f_field is not None:
                _tmp_idx = int(f_field)
                _tmp_items = [z[_tmp_idx] for z in all_fields]
                sent.info[info_name] = _tmp_items
        # --
        # finally UD related fields
        # upos
        if conf.f_upos is not None:
            f_upos = int(conf.f_upos)
            upos = [z[f_upos] for z in all_fields]
            sent.build_uposes(upos)
        # dep
        if conf.f_dep_head is not None:
            f_dep_head = int(conf.f_dep_head)
            dep_head = [int(z[f_dep_head]) for z in all_fields]
            if conf.f_dep_label is not None:
                f_dep_label = int(conf.f_dep_label)
                dep_label = [z[f_dep_label] for z in all_fields]
            else:
                dep_label = None
            sent.build_dep_tree(dep_head, dep_label)
        # --
        # other info
        for f_idx in conf.f_others:
            f_idx = int(f_idx)
            sent.info[f_idx] = [z[f_idx] for z in all_fields]  # simply put it at info!
        # --
        return doc

# helper
class ConllHelper:
    @staticmethod
    def get_f_doc(ss: List[str]):
        doc_id = ss[0]
        assert all(z==doc_id for z in ss)
        return doc_id

    @staticmethod
    def get_f_part(ss: List[str]):
        part_id = int(ss[0])
        assert all(int(z)==part_id for z in ss)
        return part_id

    @staticmethod
    def get_f_widx(ss: List[str], start: int):
        valids = []
        cur_widx = start
        for s in ss:
            try:  # note: only keep valid numbers!
                one_widx = int(s)
                valids.append(True)
                assert one_widx == cur_widx
                cur_widx += 1
            except ValueError:
                valids.append(False)
        return valids

    @staticmethod
    def get_preds(preds: List[str], pred_ids: List[str], combine_lemma_id: bool, nil_vals="-_"):
        slen = len(preds)
        assert slen == len(pred_ids)
        rets = []
        for widx in range(slen):
            one_pred, one_pred_id = preds[widx], pred_ids[widx]
            # --
            if one_pred in nil_vals or ((not combine_lemma_id) and one_pred_id in nil_vals):
                continue  # if not combine_lemma_id, require both to be valid!!
            if combine_lemma_id:
                ff = one_pred  # directly use this
            elif "." in one_pred_id:
                ff = one_pred_id  # directly use that
            else:
                ff = f"{one_pred}.{one_pred_id}"
            rets.append((widx, ff))
        return rets

    @staticmethod
    def get_f_args(pred_widx: int, args: List[str]):
        rets = []
        # --
        stack = []  # cur_start, cur_lab
        for one_widx, one_s in enumerate(args):
            while one_s[0] == "(":  # open
                _tmp_idx = 1
                while one_s[_tmp_idx] not in "(*":
                    _tmp_idx += 1
                cur_lab = one_s[1:_tmp_idx]
                one_s = one_s[_tmp_idx:]
                stack.append((one_widx, cur_lab))
            while one_s[-1] == ")":  # close
                cur_start, cur_lab = stack.pop()
                rets.append((cur_start, one_widx-cur_start+1, cur_lab))  # widx, wlen, label
                one_s = one_s[:-1]
        assert len(stack) == 0
        return rets

    # each arg is single-span (dep-srl)
    @staticmethod
    def get_f_args_dep(pred_widx: int, args: List[str]):
        rets = []
        for one_widx, one_lab in enumerate(args):
            if one_lab not in "-_":
                for _lab in one_lab.split("|"):  # in conll09-cs, there can be multiple ones
                    rets.append((one_widx, 1, _lab))  # widx, wlen, label
        return rets

    # =====

    @staticmethod
    def put_preds(frames: List[Frame], slen: int, f_pred=None):
        # --
        def _default_f_pred(_f):
            _res = _f.label.rsplit(".", 1)
            if len(_res) == 2:
                return _res[0], _res[1]
            else:
                return _res[0], "00"
        # --
        if f_pred is None:
            f_pred = _default_f_pred
        # --
        ret_preds, ret_pred_ids, ret_frames = ["-"] * slen, ["-"] * slen, [None] * slen
        for f in frames:
            # todo(+N): here only output len==1
            if f.mention.wlen != 1: continue
            one_pred, one_pred_id = f_pred(f)
            # directly overwrite!!
            one_widx = f.mention.widx
            ret_preds[one_widx] = one_pred
            ret_pred_ids[one_widx] = one_pred_id
            ret_frames[one_widx] = f
        return ret_preds, ret_pred_ids, ret_frames

    @staticmethod
    def put_args(args: List[ArgLink], slen: int):
        ret_labs = ["*"] * slen
        for arg in sorted(args, key=lambda x: x.mention.get_span()):  # sort by (widx, wlen)
            widx, wlen = arg.mention.get_span()
            role = arg.role
            ret_labs[widx] = f"({role}" + ret_labs[widx]
            ret_labs[widx+wlen-1] = ret_labs[widx+wlen-1] + ")"
        return ret_labs

    @staticmethod
    def put_args_dep(args: List[ArgLink], slen: int):
        ret_labs = ["_"] * slen
        for arg in args:
            widx, wlen = arg.mention.get_span()
            assert wlen == 1
            if ret_labs[widx] == "_":
                ret_labs[widx] = arg.role
            else:
                ret_labs[widx] = ret_labs[widx] + "|" + arg.role
        return ret_labs

# --
# shortcuts
DataFormator.reg(key="conll05o", T=lambda: ConllFormator(  # old version
    None, f_word=0, f_xpos=1, f_parse=2, f_pred=5, f_pred_id=4, f_arg_start=6))
DataFormator.reg(key="conll05", T=lambda: ConllFormator(  # note: add doc info!!
    None, f_doc=0, f_word=1, f_xpos=2, f_parse=3, f_pred=6, f_pred_id=5, f_arg_start=7, f_ne=4))
DataFormator.reg(key="conll12", T=lambda: ConllFormator(
    None, f_doc=0, f_part=1, f_widx=2, f_word=3, f_xpos=4, f_parse=5, f_pred=6, f_pred_id=7, f_sense=8, f_speaker=9,
    f_ne=10, f_arg_start=11, f_coref=-1))
DataFormator.reg(key="conllpb", T=lambda: ConllFormator(
    None, f_doc=0, f_part=1, f_widx=2, f_word=3, f_xpos=4, f_parse=5, f_pred=6, f_pred_id=7, f_arg_start=8))
DataFormator.reg(key="conllu", T=lambda: ConllFormator(
    None, f_widx=0, f_word=1, f_upos=3, f_xpos=4, f_dep_head=6, f_dep_label=7, sep_in='\t', widx_start=1, out_nil_val='_'))
DataFormator.reg(key="conllup", T=lambda: ConllFormator(  # up
    None, f_widx=0, f_word=1, f_upos=3, f_dep_head=6, f_dep_label=7,
    f_pred=9, f_pred_id=8, f_arg_start=10, sep_in='\t', widx_start=1, combine_lemma_id=True, arg_is_dep=True))
DataFormator.reg(key="conllufipb", T=lambda: ConllFormator(
    None, f_widx=0, f_word=1, f_upos=3, f_xpos=4, f_dep_head=6, f_dep_label=7, sep_in='\t', widx_start=1, f_others=[8,9]))
DataFormator.reg(key="conll09", T=lambda: ConllFormator(
    None, f_widx=0, f_word=1, f_xpos=4, f_dep_head=8, f_dep_label=10, sep_in='\t', widx_start=1,
    f_pred=13, f_pred_id=12, f_arg_start=14, combine_lemma_id=True, arg_is_dep=True,
    # note: notice that cs can have pred of "-"
    pred_nil_vals="_"))
DataFormator.reg(key="conlltrpb", T=lambda: ConllFormator(
    None, f_widx=0, f_word=1, f_upos=3, f_dep_head=6, f_dep_label=7,
    f_pred=11, f_pred_id=10, f_arg_start=12, sep_in='\t', widx_start=1, combine_lemma_id=True, arg_is_dep=True))

# --
# b mspx/data/rw/formats/conll:141
