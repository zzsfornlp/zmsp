#

# simply split by doc-id

import json
import sys
from collections import Counter

SPLITS = {
    "train": {f'{z:02d}' for z in range(2,22)},
    "dev": {'22'},
    "test": {'23'},
}

# --
def main(input_file: str, output_prefix: str):
    hits = Counter()
    split_docs = {'train': [], 'dev': [], 'test': []}
    with open(input_file) as fd:
        for line in fd:
            doc = json.loads(line)
            doc_id = doc['_id']
            assert doc_id.startswith("wsj_")
            wset = doc_id[4:6]
            for k, v in SPLITS.items():
                if wset in v:
                    hits[k] += 1
                    split_docs[k].append(line)
    # --
    for k, _ds in split_docs.items():
        with open(output_prefix+f".{k}.json", 'w') as fd:
            for _d in _ds:
                fd.write(_d)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.nombank.split_nb ?? nb
if __name__ == '__main__':
    main(*sys.argv[1:])
