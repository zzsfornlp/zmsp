#

# get stat of the data and shuffle them all

import sys
import json
from collections import Counter
from msp2.utils import Conf, init_everything, Random, zglob, zlog, zopen

class MainConf(Conf):
    def __init__(self):
        self.input_prefix = ""  # {input_prefix}*
        self.output_prefix = ""  # {output_prefix}_[N].json
        self.output_lpf = 50000  # output lines per file
        self.shuffle_times = 1
        # --
        self.sent_thresh = 1  # should >= this
        self.tok_thresh = 128  # should >= this
        # --
        # extra processing
        self.do_unescape = False  # xml.sax.saxutils.unescape
        self.do_shorten_http = False  # shorten tokens starting with https: or http:
        # --

# --
# some processors (inplaced!!)
def do_unescape(doc):
    from xml.sax.saxutils import unescape
    for sent in doc["sents"]:
        sent['seq_word']['vals'] = [unescape(z) for z in sent['seq_word']['vals']]

def do_shorten_http(doc):
    for sent in doc["sents"]:
        sent['seq_word']['vals'] = [
            ('http' if (z.startswith('http://') or z.startswith('https://')) else z) for z in sent['seq_word']['vals']]
# --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    processors = []
    if conf.do_unescape: processors.append(do_unescape)
    if conf.do_shorten_http: processors.append(do_shorten_http)
    # --
    cc = Counter()
    vocab = Counter()
    all_files = sorted(zglob(conf.input_prefix+"*"))
    all_lines = []
    for f in all_files:
        cc0 = Counter()
        zlog(f"Read from {f} ...")
        cc0["file"] += 1
        with zopen(f) as fd:
            for line in fd:
                line = line.strip()
                cc0["doc"] += 1
                doc = json.loads(line)
                # --
                for pp in processors:
                    pp(doc)
                # --
                num_sent = len(doc["sents"])
                num_tok = sum(len(z["seq_word"]["vals"]) for z in doc["sents"])
                cc0["sent"] += num_sent
                cc0["tok"] += num_tok
                if num_sent>=conf.sent_thresh and num_tok>=conf.tok_thresh:
                    cc0["doc_ok"] += 1
                    cc0["sent_ok"] += num_sent
                    cc0["tok_ok"] += num_tok
                    # all_lines.append(line)
                    all_lines.append(json.dumps(doc))
                    for ss in doc["sents"]:
                        for tt in ss["seq_word"]["vals"]:
                            vocab[tt] += 1
        zlog(f"Read finish: {cc0}")
        cc += cc0
    zlog(f"Read all finish: {cc}")
    # --
    # output
    zlog("Shuffle them ...")
    _gen = Random.get_generator("shuffle")
    for _ in range(conf.shuffle_times):
        _gen.shuffle(all_lines)
    if conf.output_prefix != "":
        num_line = len(all_lines)
        num_file = (num_line + conf.output_lpf - 1) // conf.output_lpf
        _padn = 1 if num_file<=0 else len(str(num_file-1))
        _pads = f"%0{_padn}d"
        zlog(f"Write them (N={num_file}) ...")
        for ii in range(num_file):
            _ss = _pads % ii
            _ff = f"{conf.output_prefix}_{_ss}.json"
            zlog(f"Write {_ff} ...")
            with zopen(_ff, 'w') as fd:
                for line in all_lines[ii*conf.output_lpf : (ii+1)*conf.output_lpf]:
                    fd.write(line+"\n")
    # --
    # write vocab
    _sum = sum(vocab.values())
    zlog(f"Vocab is {len(vocab)}/{_sum}")
    ii = 0
    accu = 0
    for kk, vv in vocab.most_common():
        accu += vv
        zlog(f"#{ii}\t{kk}\t{vv}({vv/_sum:.4f})\t{accu}({accu/_sum:.4f})")
        ii += 1
    # --

# --
# python3 stat_shuffle.py input_prefix: output_prefix:
if __name__ == '__main__':
    main(*sys.argv[1:])
