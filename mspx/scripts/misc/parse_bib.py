#

# parse bib file

import sys
import bibtexparser
import re
from itertools import chain
from collections import Counter
from mspx.utils import zopen_withwrapper, default_json_serializer, Conf, init_everything, zlog

class MainConf(Conf):
    def __init__(self):
        self.input_path = ''
        self.output_path = ''
        self.report_interval = 1000
        self.stored_keys = ['ID', 'title', 'year', 'url']  # contents to store
        self.max_count = -1
        self.print_hit = False
        self.filter_titles = []  # re patterns
        self.filter_abstracts = []  # re patterns
        self.filter_year = [0, 9999]  # [year-start, year-end]
        self.stop_by_year = True  # assuming the entries are sorted reversed by year?

def yield_entries(fin):
    buffer = []
    for line in chain(fin, [None]):
        if line is None or line.strip().startswith('@'):  # indicating end
            s = '\n'.join(buffer)
            if s.strip():
                yield from bibtexparser.loads(s).entries
            buffer = []
        if line is not None:
            buffer.append(line)

def filter_str(s: str, filters):
    s = ''.join([c for c in s if str.isalnum(c) or str.isspace(c)]).lower()  # keep only alnum
    for f in filters:
        if re.search(f, s):
            return True
    return False

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    input_path, output_path = conf.input_path, conf.output_path
    filter_titles = [re.compile(z) for z in conf.filter_titles]
    filter_abstracts = [re.compile(z) for z in conf.filter_abstracts]
    # --
    with zopen_withwrapper(input_path if input_path else sys.stdin) as fin:
        with zopen_withwrapper(output_path if output_path else sys.stdout, mode='w') as fout:
            ret = []
            for entry in yield_entries(fin):
                cc['entry_all'] += 1
                # --
                # filter
                keep = True
                if keep and filter_titles and not filter_str(entry.get('title',''), filter_titles):
                    keep = False
                if keep and filter_abstracts and not filter_str(entry.get('abstract',''), filter_abstracts):
                    keep = False
                year = int(entry.get('year', None))
                if year:
                    if year < conf.filter_year[0] or year > conf.filter_year[1]:
                        keep = False
                # check year
                if keep:
                    cc['entry_in'] += 1
                    item = {}
                    for k in conf.stored_keys:
                        ks = k.split(":")
                        if len(ks) <= 1:
                            ks.append(ks[0])
                        k0, k1 = ks[:2]
                        item[k1] = entry.get(k0)
                    ret.append(item)
                    if conf.print_hit:
                        zlog(f"Find: {item}")
                # --
                if cc['entry_all'] % conf.report_interval == 0:
                    zlog(f"Current stat: {cc}", timed=True)
                if cc['entry_in'] == conf.max_count:
                    break
                if conf.stop_by_year and year is not None and year < conf.filter_year[0]:
                    break
            zlog(f"Final stat: {cc}", timed=True)
            default_json_serializer.save_iter(ret, fout)

# python3 -m mspx.scripts.misc.parse_bib IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])

# examples
"""
python3 -m mspx.scripts.misc.parse_bib input_path:anthology.bib.gz print_hit:1 "filter_titles: active,^active" |& tee _active.log
python3 -m mspx.scripts.misc.parse_bib input_path:anthology.bib.gz print_hit:1 "filter_titles:paraphras" |& tee _para.log
"""
