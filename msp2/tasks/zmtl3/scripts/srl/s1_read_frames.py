#

# step 1: read all the frames
# (adopted from "read_frames.py")

import json
import os
import re
from collections import Counter, OrderedDict
from shlex import split as sh_split
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer
import xml.etree.ElementTree as ET
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

class MainConf(Conf):
    def __init__(self):
        # reading options
        self.dir = ""
        self.onto = "UNK"  # pb/fn/...
        self.output = ""
        # specific ones
        self.ignore_novp = True  # unless it is nb

    def _do_validate(self):
        # --
        if self.onto == 'nb':
            self.ignore_novp = False  # no ignoring for NB
        # --

# --
# note: quickly select one!
FN_VP = {
'Abounding_with': 'teem',
'Absorb_heat': 'cook',
'Activity_ongoing': 'keep',
'Agriculture': 'farm',
'Apply_heat': 'cook',
'Awareness': 'know',
'Bail_decision': None,
'Be_subset_of': None,
'Becoming_a_member': 'join',
'Becoming_aware': 'find',
'Being_employed': 'work',
'Being_incarcerated': None,
'Biological_mechanisms': None,
'Capacity': None,
'Catastrophe': 'suffer',
'Cause_change_of_phase': None,
'Cause_fluidic_motion': 'spray',
'Cause_to_move_in_place': 'turn',
'Ceasing_to_be': 'disappear',
'Certainty': 'believe',
'Change_event_time': 'delay',
'Change_of_quantity_of_possession': 'gain',
'Change_resistance': None,
'Chemical-sense_description': None,
'Circumscribed_existence': 'appear',
'Cognitive_connection': 'identify with',
'Cognitive_impact': 'strike',
'Commutative_process': 'add',
'Contingency': 'depend',
'Contrition': 'repent',
'Convey_importance': 'emphasize',
'Desirability': None,
'Desirable_event': 'should',
'Dimension': 'measure',
'Duration_relation': 'last',
'Emotion_heat': 'burn',
'Event': 'happen',
'Eventive_affecting': None,
'Execute_plan': 'implement',
'Expensiveness': 'cost',
'Finish_game': 'win',
'Forming_relationships': 'form relationships',
'Friction': 'scrape',
'Frugality': 'waste',
'Get_a_job': 'sign on',
'Getting_vehicle_underway': 'launch',
'Give_impression': 'look',
'Go_into_shape': 'fold',
'Have_associated': 'have',
'Hunting_success_or_failure': 'catch',
'Immobilization': 'handcuff',
'Intentional_traversing': None,
'Intentionally_affect': 'do',
'Judgment': 'judge',
'Light_movement': 'shine',
'Likelihood': 'tend',
'Lively_place': None,
'Locative_relation': None,
'Making_arrangements': 'plan',
'Mass_motion': 'move',
'Memory': 'remember',
'Misdeed': 'transgress',
'Motion_noise': 'noise',
'Moving_in_place': 'move',
'Non-commutative_process': None,
'Nuclear_process': None,
'Obviousness': 'show',
'Omen': 'portend',
'Ontogeny': 'develop',
'Operate_vehicle': 'drive',
'Opinion': 'think',
'Possibility': 'can',
'Personal_relationship': None,
'Posture': 'posture',
'Precipitation': 'rain',
'Punctual_perception': 'glimpse',
'Relative_time': None,
'Reliance_on_expectation': 'count on',
'Render_nonfunctional': 'break',
'Required_event': 'must',
'Rite': None,
'Rope_manipulation': 'tie',
'Sidereal_appearance': 'come up',
'Similarity': 'resemble',
'Simple_naming': 'call',
'Simultaneity': 'coincide',
'Sound_movement': None,
'Stinginess': None,
'Transition_to_state': 'grow',
'Undressing': 'take off',
'Vehicle_departure_initial_stage': 'take off',
'Verdict': 'verdict',
}
# --
PBNB_ARGM = [
    # pb
    ('ARGM-TMP', 'time'),
    ('ARGM-DIS', 'discourse'),
    ('ARGM-ADV', 'adverb'),
    ('ARGM-MOD', 'modal'),
    ('ARGM-LOC', 'place'),
    ('ARGM-MNR', 'manner'),
    ('ARGM-NEG', 'not'),
    ('ARGM-DIR', 'direction'),
    ('ARGM-ADJ', 'adjective'),
    ('ARGM-PRP', 'purpose'),
    ('ARGM-CAU', 'cause'),
    ('ARGM-PRD', 'secondary predication'),
    ('ARGM-EXT', 'extent'),
    ('ARGM-LVB', 'light verb'),
    ('ARGM-GOL', 'goal'),
    ('ARGM-COM', 'companion'),
    ('ARGM-REC', 'oneself'),
    ('ARGM-CXN', 'construction'),
    ('ARGM-DSP', 'direct speech'),
    ('ARGM-PRR', 'predicating relation'),  # for light verb
    ('ARGM-PNC', 'purpose'),  # old one
]
# --

class FrameReader:
    def __init__(self, conf: MainConf):
        from nltk.corpus import stopwords
        self.stopword_set = set(stopwords.words('english'))
        self.conf = conf

    # get lemma
    def get_lemma(self, s: str):
        from pattern.en import lemma
        rets = None
        for ii in range(5):
            try:  # note: sometimes this raises error
                rets0 = [lemma(z) for z in s.lower().split("_")]
                rets = rets0
                break
            except:
                continue
        return rets

    # score lemma
    def score_lemma(self, trg: str, cand: str):
        _tlen = min(len(trg), len(cand))
        ii = 0
        while ii<_tlen and trg[ii]==cand[ii]:
            ii += 1
        return ii/len(trg)

    # normalize role name
    def role2np_fn(self, role: str):
        ret = role.lower()
        ret = " ".join(ret.split("_"))
        return ret

    def role2np_pb(self, role_descr: str, **extra_info):
        ret = role_descr.lower()
        # replace '/' with or
        ret = ret.replace('/', ' or ')
        ret = ret.replace('=', '-')
        ret = ret.replace('__', 'what')
        ret = ret.replace('_', '-')
        for z in ["...", "fixed:", "m-loc", "etc.", "bloom-v:", ":)", "dir:", "haha"]:
            ret = ret.replace(z, '')
        # remove (...)
        ret = ret.strip()
        if ret[0] == "(" and ret[-1] == ")":
            ret = ret[1:-1]  # sometimes there are fully ones: (...)
        ret = re.sub(r'\(.*?\)', '', ret)  # note: use non-greedy!
        # simply take first piece sep by sepcial marks
        for sep in ["--", ";", ",", "?", "!"]:
            ret = ret.split(sep)[0]
        # norm & also remove '' ""
        ret = " ".join([z[1:-1] if ((z[0]+z[-1]) in ["''", "\"\"", "[]"]) else z for z in ret.split()])
        ret = " ".join(z for z in ret.split())
        # remove '' ""
        if len(ret) > 2 and (ret[0] + ret[-1]) in ["''", "\"\"", "[]"]:
            ret = ret[1:-1]
        ret = ret.replace(".", "")  # also remove '.'
        ret = re.sub(r'\(.*', '', ret)  # single open (some in nb)
        ret = ret.replace(")", "-")  # single open (some in nb)
        # --
        if not all(c=='-' or str.isspace(c) or str.isalnum(c) for c in ret.replace("'s", "").replace("'t", "")):
            zwarn(f"Role: {ret} ||| ({extra_info}) {role_descr}")
        # --
        assert len(ret) > 0
        return ret

    # read one frame from fn.xml
    def read_one_fn(self, file: str):
        tree = ET.parse(file)
        ns = "{http://framenet.icsi.berkeley.edu}"
        node_frame = tree.getroot()
        # --
        # read name
        frame_name = node_frame.attrib['name']
        # read FE
        core_roles, noncore_roles = [], []
        for node_fe in node_frame.findall(ns+'FE'):
            _cur_role, _cur_category = node_fe.attrib['name'], node_fe.attrib['coreType']
            assert _cur_category in ['Core', 'Peripheral', 'Extra-Thematic', 'Core-Unexpressed']
            if _cur_category.startswith("Core"):
                core_roles.append(_cur_role)  # simply put these
            else:
                noncore_roles.append(_cur_role)
        # read LU (find a suitable LU)
        _trgs = [z for z in self.get_lemma(frame_name) if z not in self.stopword_set]
        if len(_trgs) == 0:  # restore
            _trgs = [z for z in self.get_lemma(frame_name)]
        _vp_cands = []
        for node_lu in node_frame.findall(ns+'lexUnit'):
            _cur_lemma, _cur_pos = node_lu.attrib['name'].split('.')
            _cur_lemma = _cur_lemma.split("[")[0].strip()  # ignore [...]
            if _cur_pos == 'v':  # looking for verbs
                _score = max(self.score_lemma(z, _cur_lemma) for z in _trgs)
                _vp_cands.append((_cur_lemma, _score))
        _vp_cands.sort(key=(lambda x: -x[-1]))
        # --
        # return
        ret = {'name': frame_name, 'core_roles': core_roles, 'noncore_roles': noncore_roles, 'vp_cands': _vp_cands}
        return ret

    # --
    def _merge_fn_roles(self, frame, key):
        _roles = frame[key]
        # note: special merging of roles: for eg, Friendly_or_hostile: Side_1/Side_2/Sides -> Sides
        r1s = sorted([z for z in _roles if z.endswith("_1")])
        r2s = sorted([z for z in _roles if z.endswith("_2")])
        if len(r1s) > 0 or len(r2s) > 0:
            assert len(r1s) == len(r2s)
            rss = []
            for a, b in zip(r1s, r2s):
                assert a[:-2] == b[:-2]
                c = (a[:-3] + 'ies') if a[-3] == 'y' else (a[:-2] + 's')
                rss.append(f"{c}(in={c in _roles})")
                if c not in _roles:
                    _roles.append(c)
                _roles.remove(a)
                _roles.remove(b)
            zwarn(f"merge {frame['name']}/{key}:{rss} {_roles}")
        # --

    # read all for fn
    def read_fn(self, frame_dir: str):
        # --
        all_frames = []
        all_role_map = OrderedDict()
        cc = Counter()
        for fname in sorted(os.listdir(frame_dir)):
            if fname.endswith('.xml'):
                one_frame = self.read_one_fn(os.path.join(frame_dir, fname))
                cc['frame_all'] += 1
                # --
                # first check frame's vp
                if len(one_frame['vp_cands']) == 0:
                    cc['frame_noverb'] += 1
                    # zwarn(f"Ignore noverb frame: {one_frame}")
                    vp = None
                elif one_frame['name'] in FN_VP:
                    vp = FN_VP[one_frame['name']]
                    if vp is None:
                        cc['frame_rm'] += 1
                else:
                    if len(one_frame['vp_cands'])>1 and one_frame['vp_cands'][0][1] == 0.:  # no match vp
                        cc['frame_vp0'] += 1
                        # zwarn(f"This frame might have strange vp: {one_frame}")
                        # zwarn(f"Select vp: {one_frame['name']}: {[z[0] for z in one_frame['vp_cands']]}")
                        zlog(f"'{one_frame['name']}': {[z[0] for z in one_frame['vp_cands']]},")
                    vp = one_frame['vp_cands'][0][0]
                # --
                # put roles
                if vp is not None:
                    cc['frame_inc'] += 1
                elif self.conf.ignore_novp:
                    continue
                # --
                res = {}
                for key in ['core_roles', 'noncore_roles']:
                    res[key] = []
                    self._merge_fn_roles(one_frame, key)
                    for role_name in one_frame[key]:
                        np = self.role2np_fn(role_name)
                        rr = all_role_map.get(role_name)  # note: simply use role_name as key!
                        if rr is None:
                            rr = zonto.Role(role_name, np=np)
                            all_role_map[role_name] = rr
                            cc['role'] += 1
                        res[key].append(rr)
                # --
                # put frame
                ff = zonto.Frame(
                    one_frame['name'], vp=vp, core_roles=res['core_roles'], noncore_roles=res['noncore_roles'])
                all_frames.append(ff)
                # --
        # --
        onto = zonto.Onto(all_frames, list(all_role_map.values()))
        zlog(f"Read {onto} from {frame_dir}: {cc}")
        return onto

    # read one frame for pb/nb
    def read_one_pbnb(self, file: str):
        tree = ET.parse(file)
        node_frameset = tree.getroot()
        all_frames = []
        for node_predicate in node_frameset.findall('predicate'):
            lemma = node_predicate.attrib['lemma']
            vp = " ".join(lemma.lower().split("_"))  # simply use it!
            for node_roleset in node_predicate.findall('roleset'):
                # read name
                frame_name = node_roleset.attrib['id']  # XX.0?
                frame_descr = node_roleset.attrib['name']  # a short description
                if frame_name.split('.')[0] != lemma:
                    zwarn(f"Frame_name({frame_name}) != lemma({lemma})")
                # source (for nb)
                _src_fname = None
                if 'source' in node_roleset.attrib:
                    _prefix = 'verb-'
                    _source = node_roleset.attrib['source']
                    if not _source.startswith(_prefix):
                        zwarn(f"Strange source ignored: {node_roleset.attrib}")
                    else:
                        _src_fname = _source[len(_prefix):]
                # aliases
                aliases = OrderedDict()
                frame_pos = []
                for node_aliases in node_roleset.findall('aliases'):
                    for node_alias in node_aliases.findall('alias'):
                        _cur_lemma = node_alias.text.strip()
                        _cur_pos = node_alias.attrib['pos']
                        if _cur_pos:
                            frame_pos.append(_cur_pos)
                        aliases[_cur_lemma] = 1
                frame_pos = sorted(set(frame_pos))
                aliases = list(aliases.keys())
                # roles
                core_roles = []
                for node_roles in node_roleset.findall('roles'):
                    for node_role in node_roles.findall('role'):
                        role_name, role_descr = "ARG"+str.upper(node_role.attrib['n']), node_role.attrib['descr']
                        if role_name == 'ARGM':
                            _role_f = node_role.attrib.get('f','').upper()
                            if not _role_f:
                                # guess some?
                                if 'time' in role_descr:
                                    _role_f = 'TMP'
                                elif 'location' in role_descr:
                                    _role_f = 'LOC'
                            if not _role_f:
                                zwarn(f"Ignore role of ARGM-UNK?: {frame_name} {node_role.attrib}")
                                continue
                            role_name = role_name + "-" + _role_f
                            zwarn(f"Get ARGM as core: {role_name}")
                        np = self.role2np_pb(role_descr, frame_name=frame_name)
                        core_roles.append({'name': role_name, 'np': np})
                        # --
                        node_vnrole = node_role.find('vnrole')
                        if node_vnrole is not None:
                            core_roles[-1]['np_vn'] = str.lower(node_vnrole.attrib['vntheta'])
                # one
                one_frame = {'name': frame_name, 'vp': vp, 'core_roles': core_roles,
                             'frame_source': _src_fname, 'frame_aliases': aliases, 'frame_pos': frame_pos}
                all_frames.append(one_frame)
        # --
        return all_frames

    # read all for pb
    def read_pb(self, frame_dir: str):
        # --
        # first construct ARGM*
        argm_roles = [zonto.Role(name, np=np) for name,np in PBNB_ARGM]
        argm_role_map = {z.name: z for z in argm_roles}
        all_frames = []
        all_roles = [] + argm_roles  # first put argm roles
        cc = Counter()
        for fname in sorted(os.listdir(frame_dir)):
            if fname.endswith('.xml'):
                one_frames = self.read_one_pbnb(os.path.join(frame_dir, fname))
                for one_frame in one_frames:
                    cc['frame_all'] += 1
                    # --
                    has_vp = any(z in one_frame['frame_pos'] for z in 'lv')  # 'l' as light-verb
                    cc[f'frame_all_v{int(has_vp)}'] += 1
                    if self.conf.ignore_novp and not has_vp:
                        cc[f'frame_rm'] += 1
                        has_nj = False
                        for _p in 'nj':
                            if _p in one_frame['frame_pos']:
                                cc[f'frame_rm_{_p}'] += 1
                                has_nj = True
                        assert has_nj, f"Strange pos set: {one_frame}"
                        continue
                    # --
                    core_roles = []
                    for core_role in one_frame['core_roles']:
                        if core_role['name'] in argm_role_map:
                            rr = argm_role_map[core_role['name']]
                        else:  # note: each core role gets an individual one!
                            rr = zonto.Role(core_role['name'], np=core_role['np'])
                            cc['role_core'] += 1
                            cc['role_core_vn'] += int('np_vn' in core_role)
                            if 'np_vn' in core_role:
                                rr.info['np_vn'] = core_role['np_vn']
                            all_roles.append(rr)
                        core_roles.append(rr)
                    if len(core_roles) != len(set([z.name for z in core_roles])):
                        zwarn(f"Repeated roles: {one_frame['name']} {Counter([z.name for z in core_roles])}")
                        cc['frame_skip'] += 1  # simply skip this frame!
                        continue
                    # --
                    _corerole_names = set([z.name for z in core_roles])
                    noncore_roles = [z for z in argm_roles if z.name not in _corerole_names]  # if not inside core-roles
                    ff = zonto.Frame(
                        one_frame['name'], vp=one_frame['vp'], core_roles=core_roles, noncore_roles=noncore_roles,
                        frame_source=one_frame['frame_source'], frame_aliases=one_frame['frame_aliases'],
                    )
                    all_frames.append(ff)
        # --
        onto = zonto.Onto(all_frames, all_roles)
        zlog(f"Read {onto} from {frame_dir}: {cc}")
        return onto

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    reader = FrameReader(conf)
    if conf.onto == 'fn':
        onto = reader.read_fn(conf.dir)
    elif conf.onto in ['pb', 'nb']:
        onto = reader.read_pb(conf.dir)
    else:
        raise NotImplementedError(f"UNK onto {conf.onto}")
    # --
    if conf.output:
        default_json_serializer.to_file(
            onto.to_json(), conf.output, indent=2,
        )
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.s1_read_frames onto:?? dir:??
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
