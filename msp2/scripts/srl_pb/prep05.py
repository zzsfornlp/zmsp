#

# prepare data for conll05
# reference: https://github.com/strubell/preprocess-conll05

import sys
import os
from typing import List
from msp2.utils import system, zlog, zglob

# --
my_system = lambda *args, **kwargs: system(*args, **kwargs, pp=True)
# --

# first collect the data
"""
# -----
# step0: data & paths
wget https://www.cs.upc.edu/~srlconll/conll05st-release.tar.gz
tar -xzvf conll05st-release.tar.gz
wget https://www.cs.upc.edu/~srlconll/conll05st-tests.tar.gz
tar -xzvf conll05st-tests.tar.gz
wget https://www.cs.upc.edu/~srlconll/srlconll-1.1.tgz
tar -xzvf srlconll-1.1.tgz
# --
SRLCONLL="`pwd`/srlconll-1.1"
CONLL05="`pwd`/conll05st-release"
PTB3="`pwd`/TREEBANK_3"
export PERL5LIB=$SRLCONLL/lib:$PERL5LIB
"""

# setup paths
SRL05_HOME = os.environ.get("SRL05_HOME", os.curdir)  # note: use ENV
SRLCONLL = f"{SRL05_HOME}/srlconll-1.1"
CONLL05 = f"{SRL05_HOME}/conll05st-release"
PTB3 = f"{SRL05_HOME}/TREEBANK_3"
GO_PERL = f"PERL5LIB={SRLCONLL}/lib:$PERL5LIB perl"

# get words and gold trees
def prepare_from_treebanks(wset: str, section: str):
    system(f"mkdir -p {CONLL05}/{wset}/words")
    system(f"mkdir -p {CONLL05}/{wset}/synt")
    # --
    if "brown" in wset:
        assert section is None
        trees = [f"{PTB3}/PARSED/MRG/BROWN/CK/CK0{z}.MRG" for z in "123"]
        _tmp_in_cmd = " | grep -v '^\\*x\\*' "
        sec_infix = ""
    else:
        trees = sorted(zglob(f"{PTB3}/PARSED/MRG/WSJ/{section}/*.MRG"))
        _tmp_in_cmd = ""
        sec_infix = f".{section}" if "test" not in wset else ""
    # --
    # remove files if existing!
    system(f"mkdir -p {CONLL05}/{wset}/docids/")
    f_word, f_docid, f_synt = f"{CONLL05}/{wset}/words/{wset}{sec_infix}.words.gz", \
                              f"{CONLL05}/{wset}/docids/{wset}{sec_infix}.docids.gz", \
                              f"{CONLL05}/{wset}/synt/{wset}{sec_infix}.synt.gz"
    for f in [f_word, f_docid, f_synt]:
        if os.path.exists(f):
            os.remove(f)
    # add all
    for one_tree in trees:
        one_doc_id = one_tree.split("/")[-1].split(".")[0]
        my_system(f"cat {one_tree} {_tmp_in_cmd} | {GO_PERL} {SRLCONLL}/bin/wsj-removetraces.pl | {GO_PERL} {SRLCONLL}/bin/wsj-to-se.pl -w 1 | awk '{{print $1}}' | gzip >>{f_word}")
        my_system(f"cat {one_tree} {_tmp_in_cmd} | {GO_PERL} {SRLCONLL}/bin/wsj-removetraces.pl | {GO_PERL} {SRLCONLL}/bin/wsj-to-se.pl -w 1 | awk '{{print $1}}' | sed 's/^.\\+$/{one_doc_id}/' | gzip >>{f_docid}")
        my_system(f"cat {one_tree} {_tmp_in_cmd} | {GO_PERL} {SRLCONLL}/bin/wsj-removetraces.pl | {GO_PERL} {SRLCONLL}/bin/wsj-to-se.pl -w 0 -p 1 | gzip >>{f_synt}")
    # --

# concat things together
def concat_wset(wset: str, sections: List[str]):
    all_output_files = []
    for section in sections:
        # --
        if "brown" in wset:
            assert section is None
            sec_infix = ""
        else:
            sec_infix = f".{section}" if "test" not in wset else ""
        # --
        all_comps = ["docids", "words", "synt", "ne", "senses", "props"]
        all_files = []
        for comp in all_comps:
            _tmp_prefix = f"{CONLL05}/{wset}/{comp}/{wset}{sec_infix}.{comp}"
            _tmp_incmd = ""
            if not os.path.isfile(_tmp_prefix+".gz"):
                assert comp=="senses" and wset in ["test.wsj", "test.brown"]
                comp = "null"  # use null instead
                _tmp_incmd = " | sed 's/-/00/' "  # change to a default one "00"
            _tmp_prefix = f"{CONLL05}/{wset}/{comp}/{wset}{sec_infix}.{comp}"
            my_system(f"zcat {_tmp_prefix}.gz {_tmp_incmd} >{_tmp_prefix}.txt")
            all_files.append(f"{_tmp_prefix}.txt")
        # final paste
        one_output_file = f"{CONLL05}/{wset}/paste.{section}.gz"
        all_output_files.append(one_output_file)
        my_system(f"paste -d ' ' {' '.join(all_files)} | gzip >{one_output_file}")  # note: here use single ' ' to sep
    # final final concat
    my_system(f"zcat {' '.join(all_output_files)} >{CONLL05}/{wset}.conll")
    # --

# --
# small helper
def check_equal_with_strip(fa: str, fb: str):
    with open(fa) as fd1, open(fb) as fd2:
        lines1, lines2 = list(fd1), list(fd2)
        assert len(lines1) == len(lines2)
        assert all(x.strip()==y.strip() for x,y in zip(lines1, lines2))
# --

# main
def main():
    for wset, sections in zip(
        ["train", "devel", "test.wsj", "test.brown"],
        [[f"{z:02d}" for z in range(2,22)], ["24"], ["23"], [None]],
        # ["test.wsj", "test.brown"], [["23"], [None]],
    ):
        # prepare from treebanks
        for sec in sections:
            prepare_from_treebanks(wset, sec)
        # then concatenate things
        concat_wset(wset, sections)
    # --

# PYTHONPATH=../../../zsp2021/src/ python3 prep05.py |& tee _log
# for wset in train devel test.wsj test.brown; do ln -s conll05st-release/$wset.conll .; done
# mv devel.conll dev.conll
if __name__ == '__main__':
    main()
