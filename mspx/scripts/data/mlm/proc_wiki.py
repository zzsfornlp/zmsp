#

# process wiki extracted data
import sys
import time
import re
from collections import Counter
from mspx.tools.annotate import *
from mspx.data.inst import Doc, Sent
from mspx.data.rw import WriterGetterConf
from mspx.utils import Conf, init_everything, zopen, zlog, zglob1, Timer

class MainConf(Conf):
    def __init__(self):
        super().__init__()
        self.do_ann = True
        self.ann = AnnotatorStanzaConf.direct_conf(ann_input_mode='raw')
        self.W = WriterGetterConf()
        # --
        self.cl = "en"
        self.max_count = -1  # max number of doc to process
        self.doc_min_char = 50  # at least
        self.input_path = ""
        self.report_interval = 1000
        self.convert_zh_t2s = True
        # --

def yield_docs(fd):
    re_pat = re.compile(r"""<doc id="(\d+)" url="(.+)" title="(.+)">""")
    cur_doc = None
    cur_lines = []
    for line in fd:
        if line.startswith("<doc"):
            assert cur_doc is None
            groups = re_pat.fullmatch(line.strip()).groups()
            info = {k:v for k,v in zip(["id","url","title"], groups)}
            cur_doc = Doc()
            cur_doc.info.update(info)
        elif line.startswith("</doc"):
            assert cur_doc is not None
            yield cur_doc, cur_lines
            cur_doc = None
            cur_lines = []
        else:
            if cur_doc is not None:
                line = line.strip()
                if line == "":
                    pass
                elif len(cur_lines) == 0 and line == cur_doc.info.get("title"):
                    pass  # ignore title in the main context
                else:
                    cur_lines.append(line)
    # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    ann = conf.ann.make_node(stanza_lang=conf.cl) if conf.do_ann else None
    # --
    cc = Counter()
    _input_path = zglob1(conf.input_path)
    time0 = time.time()
    convert_zh_t2s = (conf.cl == 'zh') and conf.convert_zh_t2s
    if convert_zh_t2s:
        # pip install opencc-python-reimplemented
        from opencc import OpenCC
        conv = OpenCC('t2s')
    with zopen(_input_path) as fd:
        with conf.W.get_writer() as writer:
            for doc, lines in yield_docs(fd):
                if cc['doc_valid'] == conf.max_count:
                    break
                cc['doc_all'] += 1
                if sum(len(z) for z in lines) <= conf.doc_min_char:
                    cc['doc_skip'] += 1
                    continue
                if convert_zh_t2s:
                    lines = [conv.convert(z) for z in lines]
                if ann is not None:
                    for _line in lines:  # simply put one line into one sent
                        if _line:
                            tmp_doc = Doc(text=_line)
                            ann.annotate([tmp_doc])  # one at one time
                            doc.add_sents(tmp_doc.sents)
                else:
                    for _line in lines:  # simply put one line into one sent
                        doc.add_sent(Sent(text=_line))
                writer.write_inst(doc)
                cc['doc_valid'] += 1
                cc['sent'] += len(doc.sents)
                cc['word'] += sum(len(z) for z in doc.sents)
                if cc['doc_valid'] % conf.report_interval == 0:
                    tt = time.time() - time0
                    zlog(f"Processed: {cc} [{tt:.2f} sec][{tt/cc['doc_valid']:.2f} sec/d]")
    # --
    tt = time.time() - time0
    zlog(f"Finished: {cc} [{tt:.2f} sec][{tt/cc['doc_valid']:.2f} sec/d]")
    # --

# python3 -m mspx.scripts.data.mlm.proc_wiki ...
if __name__ == '__main__':
    main(*sys.argv[1:])
