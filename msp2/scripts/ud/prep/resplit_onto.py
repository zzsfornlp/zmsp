#

# resplit ontonotes data (according to those in conll12)

from glob import glob
import json
from collections import Counter

# --
def _load_line(line):
    doc = json.loads(line)
    doc_id = doc["info"]["doc_id"]
    if doc_id.startswith("ontonotes/"):
        doc_id = doc_id[len("ontonotes/"):]
    return doc, doc_id

# --
def main(input_pattern: str, ref_pattern: str, output_pattern: str):
    # --
    # first read all ref ones
    ref_files = sum([glob(z) for z in ref_pattern.split(',')], [])
    groups = {z: set() for z in ['train', 'dev', 'test']}  # note: here only three groups
    for f in ref_files:
        with open(f) as fd:
            key = [k for k in f.split(".") if k in groups]
            assert len(key) == 1 and len(groups[key[0]])==0
            key = key[0]
            # --
            cc = 0
            for line in fd:
                doc, doc_id = _load_line(line)
                # assert doc_id not in groups[key]
                groups[key].add(doc_id)
                cc += 1
            print(f"Read ref {f}: {key} -> {len(groups[key])}/{cc}")
    # first read all input ones and put them!
    input_files = sum([glob(z) for z in input_pattern.split(',')], [])
    docs = {z: [] for z in ['train', 'dev', 'test', 'unk']}
    for f in input_files:
        with open(f) as fd:
            distr = Counter()
            cc = 0
            for line in fd:
                doc, doc_id = _load_line(line)
                key = [z for z in groups if doc_id in groups[z]]
                if len(key) == 0:
                    key = ['unk']
                assert len(key) == 1
                key = key[0]
                docs[key].append(doc)
                distr[key] += 1
                cc += 1
            print(f"Read input {f} ({cc}) -> {distr}")
    # finally write
    for key, ds in docs.items():
        out_f = output_pattern.replace('*', key)
        assert out_f not in input_files and out_f not in ref_files
        with open(out_f, 'w') as fd:
            for one in ds:
                fd.write(json.dumps(one)+"\n")
        print(f"Write {out_f}: {len(ds)}")
    # --

# python3 resplit_onto.py en.onto.train.ud.json,en.onto.dev.ud.json,en.onto.test.ud.json "../pb12/en.*.conll.json" "en.ontoC.*.ud.json" |& tee _log.ontoC
# stat
# for f in en.onto*; do python3 stat_udsrl.py zjson $f; done |& tee _log.stat_onto
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
