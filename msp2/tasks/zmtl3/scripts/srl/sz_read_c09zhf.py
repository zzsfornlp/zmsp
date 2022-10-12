#

# read frames from c09zh
import os
import sys
from collections import Counter, OrderedDict
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer
import xml.etree.ElementTree as ET
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

# --
def main(frame_dir: str, output_file: str):
    import stanza
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', use_gpu=False)
    # --
    cc = Counter()
    all_frames, all_roles = [], []
    for file in sorted(os.listdir(frame_dir)):
        if not file.endswith(".xml"):
            continue
        cc['file'] += 1
        f = os.path.join(frame_dir, file)
        try:
            tree = ET.parse(f)
        except:
            cc['fileB'] += 1
            zwarn(f"Failed parsing {file}")
            continue
        node_root = tree.getroot()
        assert node_root.tag == 'verb'
        lemma = node_root.find('id').text.strip()
        cur_id = 1
        for node_frameset in node_root.findall('frameset'):
            _rid = int(node_frameset.attrib['id'][1:])
            assert _rid >= cur_id
            if _rid > cur_id:
                zwarn(f"Skip from {cur_id} to {_rid}")
                cur_id = _rid
            frame_name = f"{lemma}.{cur_id:02d}"
            roles = []
            hit_argnums = set()
            for node_role in node_frameset.findall('role'):
                _argnum = int(node_role.attrib['argnum'])
                if _argnum in hit_argnums:
                    while _argnum in hit_argnums:
                        _argnum += 1
                    zwarn(f"Repeated _argnum: {frame_name}/{node_role.attrib} -> {_argnum}")
                hit_argnums.add(_argnum)
                # --
                role_name = f"ARG{_argnum}"
                role_descr = node_role.attrib['argrole'].strip().replace(',', ' or ').replace('/', ' or ')
                if role_descr == "":  # empty one?
                    cc['role0'] += 1
                    _np = "entity"  # dummy one
                elif len(role_descr.split()) == 1:  # only one word
                    cc['role1'] += 1
                    _np = role_descr
                else:  # pos
                    _doc = nlp(role_descr)  # simply check pos
                    _words = [word for sent in _doc.sentences for word in sent.words]
                    nouns = [word.text for word in _words if word.upos in ['NOUN', 'PROPN']]
                    if len(nouns) > 0:
                        cc['roleN'] += 1
                        _np = nouns[0]
                    else:
                        cc['role?'] += 1
                        zwarn(f"Cannot find noun for {frame_name}/{role_name}: {role_descr}")
                        _np = "entity"  # put a dummy one!
                roles.append(zonto.Role(role_name, np=_np))
            ff = zonto.Frame(frame_name, vp=lemma, core_roles=roles)
            all_frames.append(ff)
            all_roles.extend(roles)
            # --
            cc['frame'] += 1
            cc['role'] += len(roles)
            cur_id += 1
    # --
    zlog(f"Read frames from {frame_dir}: {cc}")
    onto = zonto.Onto(all_frames, all_roles)
    if output_file:
        default_json_serializer.to_file(onto.to_json(), output_file, indent=2)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.sz_read_c09zhf IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
