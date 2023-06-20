#

# how to specify input/output for the model
# -- instruction + List[demo] + {query-instance}

import re
from typing import List, Dict, Sequence
from mspx.utils import Conf, Configurable, zwarn

class TemplateConf(Conf):
    def __init__(self):
        # overall specs
        self.t_instruction = ''  # optional beginning instruction
        self.instruction_end = "\n\n"
        self.demo_end = "\n\n"  # separators(ending) for demo instances
        # spec for one instance
        self.t_instance = ''  # template for one instance
        self.demo_switch = {}  # switch for demo: for example: {MASK_TOKEN}->{label}
        # extra ones to overwrite info
        self.extra_info = {}

class Template(Configurable):
    def __init__(self, conf: TemplateConf, toker, **kwargs):
        super().__init__(conf, **kwargs)
        conf: TemplateConf = self.conf
        # --
        self.toker = toker
        self.default_ones = {'MASK_TOKEN': toker.mask_token}  # some default ones
        # --
        # compile patterns
        _t_instruction = conf.t_instruction
        if _t_instruction:
            _t_instruction = _t_instruction + conf.instruction_end
        self.p_instruction = PartialSeq.parse(_t_instruction, toker)  # instruction
        # --
        # allow dynamic template based on the input!
        _t_instance = conf.t_instance
        if not isinstance(_t_instance, (list, tuple)):
            _t_instance = [(None, _t_instance)]
        assert _t_instance[-1][0] is None  # need a default one!
        # --
        self.p_instance = []
        self.p_demo_instance = []
        for ff, tt in _t_instance:
            self.p_instance.append((ff, PartialSeq.parse(tt, toker)))
            tt2 = tt
            for k, v in conf.demo_switch.items():
                tt2 = tt2.replace(k, v)
            tt2 = tt2 + conf.demo_end
            self.p_demo_instance.append((ff, PartialSeq.parse(tt2, toker)))
        # --

    def fill_template(self, ps, ms, orig_map, cur_offset):
        # first find the suitable template
        sel_pp = None
        if isinstance(ps, (list, tuple)):
            for ff, pp in ps:
                if ff is None or ff(orig_map):
                    sel_pp = pp
                    break
        else:  # no selection!
            sel_pp = ps
        _ids, _k2posi = sel_pp.fill(ms, offset=cur_offset)
        return _ids, _k2posi

    def prepare(self, inst, demos: Sequence = (), inst_extras=None, offset=0):
        conf: TemplateConf = self.conf
        # --
        cur_offset = offset
        ret_ids = []
        ret_k2posi= {'instruction': None, 'demo': [], 'instance': None}
        for one_p, one_ms, one_k in zip([self.p_instruction, self.p_demo_instance, self.p_instance], [inst, demos, inst], ['instruction', 'demo', 'instance']):
            if not isinstance(one_ms, (list, tuple)):
                one_ms = [one_ms]
            one_k2posi = []
            for one_m in one_ms:
                _maps = [self.default_ones, one_m]
                if inst_extras and one_k == 'instance':  # note: only for instance!!
                    _maps.append(inst_extras)
                _ids, _k2posi = self.fill_template(one_p, _maps, one_m, cur_offset)
                # update
                cur_offset += len(_ids)
                ret_ids += _ids
                one_k2posi.append(_k2posi)
            if one_k == 'demo':
                ret_k2posi['demo'] = one_k2posi
            else:
                assert len(one_k2posi) == 1
                ret_k2posi[one_k] = one_k2posi[0]
        # --
        return ret_ids, ret_k2posi

# --
# partially filled seq
class PartialSeq:
    _PATTERN = re.compile(r"\{[_a-zA-Z][-_a-zA-Z0-9]*\}")

    def __init__(self, orig_str: str, sub_ids: List[int], var_i2k: List[str], var_i2bb: List[bool], var_k2i: Dict[str, int], tokenizer=None):
        self.orig_str = orig_str  # original str
        self.sub_ids = sub_ids  # subtoken-id or None as PLH
        self.var_i2k = var_i2k  # var-idx to name
        self.var_i2bb = var_i2bb  # var-idx to before-blank
        self.var_k2i = var_k2i  # var name to var-idx
        self.tokenizer = tokenizer

    def has_var(self, s):
        return s in self.var_k2i

    def __repr__(self):
        return self.to_string()

    def to_string(self, ):
        if self.tokenizer is None:
            return str(self.sub_ids)
        else:
            ii = 0
            rets = []
            for _id in self.sub_ids:
                if _id is None:
                    rets.append(f"{{{self.var_i2k[ii]}}}")
                    ii += 1
                else:
                    rets.append(self.tokenizer.convert_ids_to_tokens([_id])[0])
            return str(rets)
        # --

    def find_key(self, key, ms):
        for m in ms:
            if key in m:  # note: find the first match!
                return True, m[key]
        return False, None

    # key -> str or List[int]
    def fill(self, maps, _add_cls=False, _add_sep=False, offset=0):
        if isinstance(maps, dict):
            maps = [maps]
        ms = list(reversed(maps))  # note: can overwrite previous!
        # --
        # fill it!
        ret_ids = [self.tokenizer.cls_token_id] if _add_cls else []
        ret_k2posi = {}  # key -> [widx, wlen]
        cur_ii = len(ret_ids) + offset  # cur full subtoken idx
        cur_vi = 0  # cur var idx
        for one_id in self.sub_ids:
            if one_id is not None:  # valid subtoken-id
                ret_ids.append(one_id)
                cur_ii += 1
            else:  # fill in var!
                _k = self.var_i2k[cur_vi]
                _, fill_ids = self.find_key(_k, ms)
                if isinstance(fill_ids, str):
                    if len(fill_ids) > 0 and fill_ids[0] in "<[":
                        _ss = fill_ids  # no before space for special tokens!
                    else:
                        _ss = (" " if self.var_i2bb[cur_vi] else "") + fill_ids  # space before?
                    _ss = _ss.rstrip(' ')  # note: remove extra blanks at the right!
                    fill_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(_ss))
                if fill_ids is None:
                    if not _k.startswith("_"):  # not special one!
                        zwarn(f"Unfilled item: {_k}")
                    fill_ids = []
                # update
                cur_vi += 1
                ret_k2posi[_k] = [cur_ii, len(fill_ids)]
                cur_ii += len(fill_ids)
                ret_ids.extend(fill_ids)
        if _add_sep:
            ret_ids.append(self.tokenizer.sep_token_id)
        # --
        # note: one can use "tokenizer.decode" to see the full seq
        return ret_ids, ret_k2posi

    # fill with orig str: can be used as sanity check!
    def fill_orig(self, maps, _add_cls=False, _add_sep=False):
        mm = {}
        if isinstance(maps, dict):
            maps = [maps]
        for m2 in maps:  # note: can overwrite previous!
            mm.update(m2)
        # --
        final_str = self.orig_str.format_map(mm)
        ret_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(final_str))
        if _add_cls:
            ret_ids = [self.tokenizer.cls_token_id] + ret_ids
        if _add_sep:
            ret_ids.append(self.tokenizer.sep_token_id)
        return ret_ids  # note: no mapping positions

    # accept special patterns: ... {key1} ...
    @staticmethod
    def parse(pat: str, tokenizer):
        _pp = PartialSeq._PATTERN
        # --
        # parse pat & tok
        var_i2k, var_i2bb = [], []  # idx to key, idx to blank_before??
        _prev_char = 0
        sub_ids = []
        pat = pat + "{_end}"  # note: add special END as sentinel!
        for m in _pp.finditer(pat):
            a, b = m.start(), m.end()
            var_i2k.append(pat[a:b][1:-1])  # exclude the special surroundings!
            var_i2bb.append(a>0 and str.isspace(pat[a-1]) and (pat[a-1] != '\n'))  # before is blank?
            # --
            if a > _prev_char:
                _ss = pat[_prev_char:a].rstrip(' ')
                if _ss:
                    sub_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(_ss)))
            sub_ids.append(None)
            _prev_char = b
            # --
        # final one
        var_k2i = {k:i for i,k in enumerate(var_i2k)}  # key to idx
        assert len(var_k2i) == len(var_i2k), f"Currently no support for repeated keys: {var_i2k}"
        # --
        ret = PartialSeq(pat, sub_ids, var_i2k, var_i2bb, var_k2i, tokenizer=tokenizer)
        return ret

# --
# b mspx/znew/prompt/model/template:??
