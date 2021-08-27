#

# map pos from ctb to ud

import sys
from msp2.data.resources.ud_zh import UPOS2CTB, CTB2UPOS
from msp2.utils import zopen_withwrapper

def main(args):
    file_in, file_out = args
    if file_in in ["", "-"]:
        file_in = sys.stdin
    if file_out in ["", "-"]:
        file_out = sys.stdout
    # --
    # read
    with zopen_withwrapper(file_in) as fd:
        in_lines = list(fd)
    # convert
    out_lines = []
    for line in in_lines:
        fields = line.strip().split("\t")
        if len(line.strip())==0 or len(fields) == 0 or line.startswith("#"):
            out_lines.append(line)
        else:
            fields[3] = CTB2UPOS[fields[3]]
            out_lines.append("\t".join(fields)+"\n")
    # write
    with zopen_withwrapper(file_out, mode='w') as fd:
        for line in out_lines:
            fd.write(line)
    # --
# --
# python3 upos_ctb2ud.py '' '' <? >?
if __name__ == '__main__':
    main(sys.argv[1:])
