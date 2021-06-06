#

# format changer (only for line/file rw)

from msp2.cli import annotate
import sys

# --
# extra: simple one to make doc into sent
def doc2sent(df, sf):
    import json
    docs = [json.loads(line) for line in open(df)]
    with open(sf, 'w') as fd:
        for doc in docs:
            for sent in doc["sents"]:
                fd.write(json.dumps(sent)+"\n")
# --

# PYTHONPATH=../src/ python3 -m msp2.cli.change_format ...
if __name__ == '__main__':
    # simply annotate with no annotators
    annotate.main("", *sys.argv[1:])
