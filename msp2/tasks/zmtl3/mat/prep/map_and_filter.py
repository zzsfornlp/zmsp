#

# map types between schemes

import os
import sys
from collections import OrderedDict, Counter
from msp2.utils import zopen, zlog, mkdir_p
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, CharIndexer
from msp2.data.rw import ReaderGetterConf, WriterGetterConf


MAP_ENT = {
    'Material': 'Material',
    'Number': 'Number',
    'Operation': 'Operation',
    'Amount-Unit': 'Amount_Unit',
    'Condition-Unit': 'Amount_Unit',
    'Material-Descriptor': 'Descriptor',
    'Condition-Misc': 'Environment',
    'Synthesis-Apparatus': None,
    'Nonrecipe-Material': None,
    'Brand': None,
    'Property-Misc': 'Property',
    'Apparatus-Descriptor': 'Descriptor',
    'Amount-Misc': None,
    'Property-Type': 'Property',  # probably
    'Apparatus-Unit': 'Amount_Unit',
    'Reference': None,
    'Property-Unit': 'Amount_Unit',
    'Meta': 'Synthesis',
    'Condition-Type': 'Environment',  # probably
    'Characterization-Apparatus': 'Characterization',
    'Apparatus-Property-Type': None,
}
MAP_REL = {
    'Next_Operation': 'Next_Opr',
    'Number_Of': 'Number_of',
    'Condition_Of': 'Condition_Of',
    'Participant_Material': None,
    'Amount_Of': 'Amount_of',
    'Descriptor_Of': 'Form_Of',
    'Recipe_Precursor': 'Input',
    'Property_Of': 'Property_Of',
    'Solvent_Material': None,
    'Apparatus_Of': None,
    'Brand_Of': None,
    'Recipe_Target': 'Output',
    'Coref_Of': 'Coref',
    'Atmospheric_Material': None,
    'Type_Of': None,
    'Apparatus_Attr_Of': None,
}
REV_REL = {'Input'}  # reverse direction!

def main(input_path, output_path, is_src):
    is_src = bool(int(is_src))  # src->trg or trg->src
    cc = Counter()
    cc_delE, cc_delER, cc_delR = Counter(), Counter(), Counter()
    reader = ReaderGetterConf().get_reader(input_path=input_path)
    TRG_ENT = set([z for z in MAP_ENT.values() if z is not None])
    TRG_REL = set([z for z in MAP_REL.values() if z is not None])
    all_insts = []
    rev_rels = []
    deleted_frames = set()
    for inst in reader:
        all_insts.append(inst)
        cc['inst'] += 1
        for sent in yield_sents(inst):
            cc['sent'] += 1
            for evt in list(sent.events):
                cc['evt'] += 1
                cc['rel'] += len(evt.args)
                # mapping
                trg_label = MAP_ENT[evt.label] if is_src else evt.label
                if trg_label not in TRG_ENT:
                    cc['evt_del'] += 1
                    cc_delE.update([evt.label])
                    cc['rel_delE'] += len(evt.args)  # relation also deleted
                    cc_delER.update([z.label for z in evt.args])
                    sent.delete_frame(evt, 'evt')
                    deleted_frames.add(id(evt))
                else:
                    evt.set_label(trg_label)
                    cc['evt_keep'] += 1
                    for arg in list(evt.args):
                        trg_role = MAP_REL[arg.label] if is_src else arg.label
                        if trg_role not in TRG_REL:
                            cc['rel_del'] += 1
                            cc_delR.update([arg.label])
                            arg.delete_self()
                        else:
                            arg.set_label(trg_role)
                            if is_src and trg_role in REV_REL:
                                rev_rels.append(arg)
                            else:
                                cc['rel_keep'] += 1
    for alink in rev_rels:  # reverse direction
        if id(alink.main) not in deleted_frames and id(alink.arg) not in deleted_frames:
            cc['rel_rev'] += 1
            alink.arg.add_arg(alink.main, alink.role)
            alink.delete_self()
    # --
    zlog(f"Map {input_path}(is_src={is_src}) -> {output_path}: {cc}")
    zlog(f"cc_delE: {cc_delE}")
    zlog(f"cc_delER: {cc_delER}")
    zlog(f"cc_delR: {cc_delR}")
    if output_path:
        with WriterGetterConf().get_writer(output_path=output_path) as writer:
            writer.write_insts(all_insts)
    # --

def reformat_scores(s: str):
    import re
    pat = r"([0-9.]+)\(([0-9.]+)\)|([0-9.]+)"
    fs = s.strip().split('/')
    ret = ""
    for f in fs:
        mm = re.fullmatch(pat, f)
        a, b, c = mm.groups()
        if c is None:
            ret += f" & {float(a)*100:.2f}$_{{{float(b)*100:.2f}}}$"
        else:
            ret += f" & {float(c)*100:.2f}"
    return ret

# python3 -m msp2.tasks.zmtl3.mat.prep.map_and_filter ...
if __name__ == '__main__':
    main(*sys.argv[1:])
