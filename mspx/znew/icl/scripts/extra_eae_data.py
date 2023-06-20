#

# extract evt-arg and rel examples from data

import sys
import re
from collections import Counter
from mspx.data.inst import yield_sents
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.utils import zlog, zopen_withwrapper, Conf, init_everything, default_json_serializer, Random, ZHelper

class MainConf(Conf):
    def __init__(self):
        self.input_path = ""
        self.output_path = ""
        # --
        # collect
        self.evt_filters = ["Conflict:Attack"]  # evt types?
        self.role_filters = ["*"]  # "*" means match all!
        self.nil_name = "None"  # "" means no nil!
        self.spath_lab_level = 2

def collect_examples(conf):
    evt_filters, role_filters = conf.evt_filters, conf.role_filters
    _match_all_evt, _match_all_role = "*" in conf.evt_filters, "*" in conf.role_filters
    # --
    all_items = []
    for inst in ReaderGetterConf().get_reader(input_path=conf.input_path):
        for sent in yield_sents(inst):
            for evt in sent.get_frames(cates='evt'):
                if not _match_all_evt and not any(z in evt.label for z in evt_filters):
                    continue
                a_map = {id(aa.arg): aa.label for aa in evt.get_args()}
                for ef in sent.get_frames(cates='ef'):
                    a_lab = a_map.get(id(ef), conf.nil_name)
                    if not a_lab:
                        continue  # ignore NIL if not specified
                    if not _match_all_role and not any(z in a_lab for z in role_filters):
                        continue
                    # --
                    # get more details of the syntax path
                    spine_evt, spine_ef = sent.tree_dep.get_path_between_mentions(evt.mention, ef.mention, inc_common=1, return_joint_lab=False)
                    assert spine_evt[-1] == spine_ef[-1]
                    _spines = [spine_evt[:-1], [spine_evt[-1]], spine_ef[:-1]]
                    _syntax = [sent.tree_dep.seq_head.vals, sent.tree_dep.seq_label.vals]
                    # --
                    dd = {'sent': sent.get_text(), 'tokens': sent.seq_word.vals, 'ent': ef.mention.get_words(concat=True), 'evt': evt.mention.get_words(concat=True), 'ent_label': ef.label, 'evt_label': evt.label, 'role': a_lab, 'spath': sent.tree_dep.get_path_between_mentions(evt.mention, ef.mention, level=conf.spath_lab_level), 'spines': _spines, 'syntax': _syntax}
                    all_items.append(dd)
    # --
    cc_role = Counter([z['role'] for z in all_items])
    zlog(f"Collecting all roles: {cc_role}")
    return all_items

def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    all_items = collect_examples(conf)
    with zopen_withwrapper(conf.output_path, empty_std=True, mode='w') as fd:
        default_json_serializer.save_iter(all_items, fd)
    # --

# python3 -m mspx.znew.icl.scripts.extra_eae_data IN OUT
if __name__ == '__main__':
    main(sys.argv[1:])

"""
python3 -m mspx.znew.icl.scripts.extra_eae_data input_path:../../data/evt/dataRS/genres/en.nw.test.ud2.json
"""
