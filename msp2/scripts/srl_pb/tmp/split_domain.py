#

# simply split json files by "doc_id"

import json
import os
from typing import List
from collections import Counter
from msp2.utils import zopen, zlog, Conf, OtherHelper, default_json_serializer
from msp2.data.rw import ConllHelper

# --
class MainConf(Conf):
    def __init__(self):
        self.input = "/dev/stdin"
        self.output_dir = "./"
        self.output_insert_place = 1  # after the first '.'
        # --
        self.domain_f = "pb12"

# --
DOMAIN_FS = {
    "pb12": lambda inst: inst["info"]["doc_id"].split("/")[0],
}
# --

def main(args):
    conf = MainConf()
    conf.update_from_args(args)
    stat = Counter()
    if conf.domain_f in DOMAIN_FS:
        domain_f = DOMAIN_FS[conf.domain_f]
    else:
        domain_f = eval(conf.domain_f)
    # --
    all_insts = {}  # ID->List[inst]
    with zopen(conf.input) as fin:
        for line in fin:
            inst = json.loads(line)
            domain = domain_f(inst)
            if domain not in all_insts:
                all_insts[domain] = []
            all_insts[domain].append(inst)
            stat["inst"] += 1
            stat[f"inst_{domain}"] += 1
    # --
    # write
    input_name = os.path.basename(conf.input)
    for domain, insts in all_insts.items():
        output_name_fields = input_name.split(".")
        output_name_fields.insert(conf.output_insert_place, domain)
        output_name = os.path.join(conf.output_dir, ".".join(output_name_fields))
        zlog(f"Write to {output_name} {len(insts)}")
        with zopen(output_name, 'w') as fout:
            default_json_serializer.save_iter(insts, fout)
    # --
    zlog(f"Read from {fin}, stat=\n{OtherHelper.printd_str(stat)}")

# PYTHONPATH=../../../zsp2021/src/ python3 split_domain.py ...
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
