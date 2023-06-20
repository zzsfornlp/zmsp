#

import re
import numpy as np
from collections import Counter
from mspx.utils import Conf, zglob1, default_json_serializer, zlog, Random, ZHelper

# --
class MyPattern:
    VAR_PAT_INDICATOR = '##'
    VAR_PAT = re.compile(r"(##[a-zA-Z_]+?##)")

    def __init__(self, pattern):
        # parse the pattern
        self.pieces = re.split(MyPattern.VAR_PAT, pattern)
        self.maps = {}
        for ii, pp in enumerate(self.pieces):
            if MyPattern.VAR_PAT.fullmatch(pp):
                nn = pp[2:-2]
                assert len(nn) > 0 and nn not in self.maps
                self.maps[nn] = ii

    def fill_one(self, d, cc=None):
        if isinstance(d, dict):
            d = [d]
        # --
        ret = self.pieces.copy()
        for nn, ii in self.maps.items():
            vv = ''
            hit_flag = False
            for d0 in d:
                if nn in d0:
                    vv = d0[nn]
                    hit_flag = True
                    break
            if cc is not None and not hit_flag:
                cc[f'miss_{nn}'] += 1
            ret[ii] = vv
        ret_s = ''.join(ret)
        return ret_s

class MyPatternDict:
    KEY_DIRECT = "ZZDIRECTZZ",  # special key

    def __init__(self, pattern_dict):
        _dict = {}
        for k, v in pattern_dict.items():
            if isinstance(v, str) and MyPattern.VAR_PAT_INDICATOR in v:
                _dict[k] = MyPattern(v)
            else:
                _dict[k] = v
        self.pattern_dict = _dict

    def fill_one(self, d, cc=None):
        if isinstance(d, dict):
            d = [d]
        # --
        if cc is not None:
            cc['all'] += 1
        ret = {}
        for k, v in self.pattern_dict.items():
            if isinstance(v, MyPattern):
                v2 = v.fill_one(d, cc)
            elif isinstance(v, tuple) and v[0] == self.KEY_DIRECT:
                hit_flag = False
                for d0 in d:
                    if k in d0:
                        v2 = d0[k]
                        hit_flag = True
                        break
                if not hit_flag:
                    v2 = v[1]  # use a default one!
            else:
                v2 = v
            ret[k] = v2
        return ret

# --
class IclFormatConf(Conf):
    def __init__(self):
        self.template = ''  # main template
        self.templateC = ''  # template for calibration
        self.inst_sep = ".\n"  # separators between insts
        self.label_f = "x"  # function for label transformation
        self.print_first = 1  # print first several ones for debugging
        self.f_repr_sent = ['sent']  # sent formatting

def format_queries(data_pairs, conf: IclFormatConf, label_options, task_helper=None, **kwargs):
    conf: IclFormatConf = IclFormatConf.direct_conf(conf, **kwargs)
    from .select import InstFormator
    _repr_former = InstFormator(conf.f_repr_sent)
    # --
    zlog("Start to format data ...")
    _pat, _patC = MyPatternDict(TEMPLATES[conf.template]), \
        (MyPatternDict(TEMPLATES[conf.templateC]) if conf.templateC else None)
    _label_f = ZHelper.eval_ff(conf.label_f, 'x')
    ret = []
    for demos, dp in data_pairs:
        # first form demo prefix
        demo_prefixes = []
        for one_demo in demos:
            one_d = _pat.fill_one([one_demo, {'_label': _label_f(one_demo['label']),
                                              '_sent': _repr_former.form_sent_repr(one_demo)}])
            one_s = one_d['input'] + one_d['output'] + conf.inst_sep  # simply directly concat
            demo_prefixes.append(one_s)
        demo_prefix = ''.join(demo_prefixes)
        # then add inst
        _label_options = label_options
        if isinstance(_label_options, str):
            _label_options = dp[_label_options]  # obtain from the data point
        if _label_options is None:
            _label_options = [None]
        # --
        one_ret = {'data': [], 'map': []}
        for cur_pat, record_map in zip([_pat, _patC], [1,0]):
            if cur_pat is None: continue
            for _lab in _label_options:
                # if None option, simply feed the original one!
                _fill_lab = _lab if _lab is not None else dp['label']
                one_d = cur_pat.fill_one([dp, {'_label': _label_f(_fill_lab),
                                               '_sent': _repr_former.form_sent_repr(dp)}])
                one_ret['data'].append((demo_prefix+one_d['input'], one_d['output']))  # (in,out)
                if record_map:
                    one_ret['map'].append(_lab)
        ret.append(one_ret)
        # --
        if len(ret) <= conf.print_first:
            zlog(f"Try printing the first several query: map={one_ret['map']}")
            for ii, dd in enumerate(one_ret['data']):
                zlog(f"# -- Input[{ii}]:\n{dd[0]}\n# -- Output[{ii}]\n{dd[1]}")
    # --
    return ret

# --
TEMPLATES = {
# --
"eae": {  # EAE
    'input': """##_sent## In the event "##evt##", the entity "##ent##" plays the role of""",
    'output': " ##_label##",
},
"eaeC": {  # no specific entity/event (for calibration)
    'input': """##_sent## In the event, the entity plays the role of""",
    'output': " ##_label##",
},
"eaeR": {  # reverse (channel for EAE)
    'input': "##_label##",
    'output': """ is the role that the entity "##ent##" plays in the event "##evt##" for the following sentence: "##_sent##\"""",
},
# --
"sst": {
    'input': """In the sentence "##_sent##", the sentiment is""",
    'output': " ##_label##",
},
"sstC": {
    'input': """In the sentence, the sentiment is""",
    'output': " ##_label##",
},
"sstR": {
    'input': "##_label##",
    'output': """ is the sentiment of the sentence "##_sent##\"""",
},
# --
"sstv2": {
    'input': """Review: ##_sent##\nSentiment:""",
    'output': " ##_label##",
},
}
