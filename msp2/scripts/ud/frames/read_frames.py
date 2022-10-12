#

# read frame files from pb/fn

import os
from shlex import split as sh_split
from msp2.data.vocab.frames import *
from msp2.utils import Conf, zlog, zopen, default_pickle_serializer
import xml.etree.ElementTree as ET

class MainConf(Conf):
    def __init__(self):
        # reading options
        self.dir = ""
        self.onto = "UNK"  # pb/fn/...
        # load/save
        self.load_pkl = ''
        self.save_txt = ''
        self.save_pkl = ''
        # debug and others
        self.debug = False
        self.query = False
        # --

class FrameReader:
    def __init__(self):
        pass

    # read one frame from fn.xml
    def _read_one_fn(self, file: str):
        # --
        def _clean_def(_def, _name):
            # note: discard certain parts of descriptions
            # todo(note): currently not handling frame ones
            _flag = False
            for _prefix0 in ["This FE", "This frame element", "This Frame Element",
                             f"The {_name}", f"{_name}", f"The FE {_name}"]:
                for _prefix1 in ["describes", "indicates", "is", "denotes", "marks", "signifies", "presents", "identifies"]:
                    _p2 = f"{_prefix0} {_prefix1} "
                    if _def.startswith(_p2):
                        _def = _def[len(_p2)].upper() + _def[len(_p2)+1:]
                        _flag = True
                        break
                if _flag: break
            return _def
        # --
        def _extract_def(_annotated_def, _name):
            _t = ET.fromstring(_annotated_def)
            _alltext = ''.join(_t.itertext())
            _ret = _alltext.split("\n\n")[0]  # only the first line
            if _name is not None:
                _ret = _clean_def(_ret, _name)
            return _ret.strip()
        # --
        tree = ET.parse(file)
        ns = "{http://framenet.icsi.berkeley.edu}"
        node_frame = tree.getroot()
        # --
        # read name
        frame_name = node_frame.attrib['name']
        frame_descr = _extract_def(node_frame.find(ns+"definition").text, None)
        one_frame = ZFrame(frame_name, descr=frame_descr)
        # read FE
        for node_fe in node_frame.findall(ns+'FE'):
            _cur_role, _cur_category = node_fe.attrib['name'], node_fe.attrib['coreType']
            role_descr = _extract_def(node_fe.find(ns+"definition").text, _cur_role)
            _role = ZRole(_cur_role, _cur_category, descr=role_descr)
            one_frame.add_role(_role)
            # extra info
            for _key in ["semType", "requiresFE", "excludesFE"]:
                _values = [z.get('name') for z in node_fe.findall(ns+_key)]
                if len(_values) > 0:
                    _role.info[_key] = _values
        # read lexicons (LU)
        for node_lu in node_frame.findall(ns+'lexUnit'):
            _cur_lemma, _cur_pos = node_lu.attrib['name'].split('.')
            _attr_pos = node_lu.attrib['POS'].lower()
            if not (_cur_pos == _attr_pos or "idio" in [_cur_pos, _attr_pos]):
                zwarn(f"Strange atrrib: {node_lu.attrib}")
            _lex = ZLexicon(_cur_lemma, _cur_pos)
            one_frame.add_lexicon(_lex)  # add lexicon
        # --
        return [one_frame]

    # read one frame from pb.xml
    def _read_one_pb(self, file: str):
        tree = ET.parse(file)
        node_frameset = tree.getroot()
        all_frames = []
        file_name = os.path.basename(file)  # file name
        for node_predicate in node_frameset.findall('predicate'):
            lemma = node_predicate.attrib['lemma']
            # assert file_lemma in lemma.split("_")  # note: nope
            for node_roleset in node_predicate.findall('roleset'):
                # read name
                frame_name = node_roleset.attrib['id']  # XX.0?
                frame_descr = node_roleset.attrib['name']  # a short description
                # assert frame_name.split('.')[0] == lemma  # we can read lemma by frame name!
                if frame_name.split('.')[0] != lemma:
                    zwarn(f"Frame_name({frame_name}) != lemma({lemma})")
                one_frame = ZFrame(frame_name, descr=frame_descr)
                # source
                if 'source' in node_roleset.attrib:
                    _source = node_roleset.attrib['source']
                    if not _source.startswith('verb-'):
                        zwarn(f"Strange source ignored: {node_roleset.attrib}")
                    else:
                        one_frame.info['source'] = _source[len('verb-'):]
                # read aliases/lexicons
                aliases = []
                for node_aliases in node_roleset.findall('aliases'):
                    for node_alias in node_aliases.findall('alias'):
                        _cur_lemma, _cur_pos = node_alias.text.strip(), node_alias.attrib['pos']
                        _lex = ZLexicon(_cur_lemma, _cur_pos)
                        one_frame.add_lexicon(_lex)  # add lexicon
                        # add aliases
                        for prefix, keyname in zip(['fn:', 'vn:'], ['framenet', 'verbnet']):
                            _extra_names = node_alias.get(keyname)
                            if _extra_names:
                                aliases.extend([prefix+n for n in _extra_names.split()])
                one_frame.info['filename'] = file_name
                if len(aliases)>0:
                    one_frame.info['aliases'] = aliases
                    assert lemma in [z.lemma for z in one_frame.lexicons]
                else:
                    _lex = ZLexicon(lemma, 'UNK')
                    one_frame.add_lexicon(_lex)  # add lexicon
                # read roles
                for node_roles in node_roleset.findall('roles'):
                    for node_role in node_roles.findall('role'):
                        role_name, role_descr = "ARG"+node_role.attrib['n'], node_role.attrib['descr']
                        _role = ZRole(role_name, category='core', descr=role_descr)
                        one_frame.add_role(_role)
                        # extra info
                        _role.info['f'] = node_role.attrib.get('f', '').upper()
                        node_vnrole = node_role.find('vnrole')
                        if node_vnrole is not None:
                            _role.info['aliases'] = ["vn:"+node_vnrole.attrib['vntheta']]
                # --
                all_frames.append(one_frame)
        return all_frames

    # read pb's examples
    def _read_pb_examples(self, file: str):
        raise NotImplementedError()

    # read all frames from one directory
    def read_all(self, directory: str, onto: str):
        _read_f = getattr(self, "_read_one_"+onto)
        # --
        all_frames = []
        nfiles = 0
        for fname in sorted(os.listdir(directory)):
            if fname.endswith('.xml'):
                nfiles += 1
                one_frames = _read_f(os.path.join(directory, fname))
                all_frames.extend(one_frames)
        zlog(f"Read from {directory}: {nfiles} files and {len(all_frames)} frames!")
        # --
        collection = ZFrameCollection(all_frames)
        return collection

# --
# extra helpers
# for example: (lambda r: r.role+r.info['f']) or (lambda r: r.descr) or
# (lambda r: r.info['aliases'][0] if 'aliases' in r else r.descr)
def group_frames(frames, role_key):
    groups = {}
    for f in frames:
        key = "|".join([role_key(z) for z in f.roles])
        if key not in groups:
            groups[key] = []
        groups[key].append(f)
    return groups

# --
def main(*args):
    conf = MainConf()
    conf.update_from_args(args)
    # --
    if conf.load_pkl:
        collection = default_pickle_serializer.from_file(conf.load_pkl)
    else:
        reader = FrameReader()
        collection = reader.read_all(conf.dir, conf.onto)
    if conf.save_pkl:
        default_pickle_serializer.to_file(collection, conf.save_pkl)
    if conf.save_txt:
        with zopen(conf.save_txt, 'w') as fd:
            for f in collection.frames:
                fd.write("#--\n" + f.to_string() + "\n")
    # --
    if conf.debug:
        breakpoint()
    if conf.query:
        map_frame = {f.name: f for f in collection.frames}
        map_lu = ZFrameCollectionHelper.build_lu_map(collection, split_lu={"pb":"_", "fn":None}[conf.onto])
        map_role = ZFrameCollectionHelper.build_role_map(collection)
        while True:
            line = input(">> ")
            fields = sh_split(line.strip())
            if len(fields) == 0:
                continue
            try:
                query0, query1 = fields
                _map = {'frame': map_frame, 'lu': map_lu, 'role': map_role}[query0]
                answer = _map.get(query1, None)
                if isinstance(answer, ZFrame):
                    zlog(answer.to_string())
                else:
                    zlog(answer)
            except:
                zlog(f"Wrong cmd: {fields}")
                pass
    # --

# PYTHONPATH=../src/ python3 read_frames.py
# PYTHONPATH=../src/ python3 -m msp2.scripts.ud.frames.read_frames
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
