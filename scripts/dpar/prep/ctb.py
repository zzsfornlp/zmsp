#!/usr/bin/env python3

# prepare dependencies for CTB with Penn2Malt
# -> https://stp.lingfil.uu.se/~nivre/research/Penn2Malt.html
# the splitting follows http://aclweb.org/anthology/D08-1059

import os

# in the directory of */ctb/
BR_HOME = "../CTB/data/utf-8/bracketed/"
OUT_DIR = "./"
dir_name = os.path.dirname(os.path.abspath(__file__))
CONVERT_PYFILE = os.path.join(dir_name, "conll.py")

PENN2MALT_JAR = "../../tools/Penn2Malt.jar"
CHN_P2M_HEADRULES = "../../tools/chn_headrules.txt"

def system(cmd, pp=False, ass=True):
    if pp:
        print("Executing cmd: %s" % cmd)
    n = os.system(cmd)
    if ass:
        assert n==0

def get_name(s, N=4):
    n_zero = N - len(s)
    assert n_zero >= 0
    for i in range(n_zero):
        s = "0" + s
    return s

def main():
    print("Step1: merge ctb files")
    infos = [
        ["train.ctb", [(1, 815), (1001, 1136)]],
        ["dev.ctb", [(886, 931), (1148, 1151)]],
        ["test.ctb", [(816, 885), (1137, 1147)]],
    ]
    for info in infos:
        print("Merging for %s" % info)
        FNAME = OUT_DIR + info[0]
        system(">%s" % FNAME)
        for rr in [range(p[0], p[1]+1) for p in info[1]]:
            for i in rr:
                nn = get_name(str(i))
                system("cat %s/chtb_%s.* | grep -v -E '<.*>' >>%s" % (BR_HOME, nn, FNAME), ass=False)
    print("Step2: convert to dependencies")
    for info in infos:
        FNAME = OUT_DIR + info[0]
        _BGK_FNAME = FNAME+".gbk"
        TAB_NAME = FNAME+".tab"
        FINAL_NAME = FNAME+".conll"
        # penn2malt
        system("iconv -f UTF8 -t GBK < %s > %s" % (FNAME, _BGK_FNAME))
        system("java -jar %s %s %s 3 2 chtb" % (PENN2MALT_JAR, _BGK_FNAME, CHN_P2M_HEADRULES), True)
        system("iconv -f GBK -t UTF8 < %s.3.pa.gs.tab > %s" % (_BGK_FNAME, TAB_NAME))
        # tab2conll
        system("python3 %s %s tab %s conll06" % (CONVERT_PYFILE, TAB_NAME, FINAL_NAME), True)

if __name__ == '__main__':
    main()
