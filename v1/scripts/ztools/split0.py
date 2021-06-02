#

# split one file into multiple ones

import sys, gzip, math
import argparse

def zopen(filename, mode='r', encoding="utf-8"):
    if filename.endswith('.gz'):
        # "t" for text mode of gzip
        return gzip.open(filename, mode+"t", encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("nums", type=str)
    parser.add_argument("--outs", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)       # fname.
    parser.add_argument("--suffix", type=str, default="")
    #
    a = parser.parse_args()
    args = vars(a)
    #
    args["nums"] = [int(x) for x in args["nums"].split(",")]
    pieces = len(args["nums"])
    if args["prefix"] is None:
        if args["outs"] is None:
            args["prefix"] = args["file"] + "."
        else:
            args["prefix"] = ""
    if args["outs"] is None:
        digits = int(math.log10(pieces)) + 1
        pattern = "%%0%d.d" % (digits,)
        args["outs"] = [pattern%i for i in range(pieces)]
    else:
        args["outs"] = args["outs"].split(",")
    assert len(args["outs"]) == pieces
    return args

# split0.py <file> <n0,n1,n2,...> [outs=<numbering>] [prefix=<file.>] [suffix=""]
def main():
    # parse cmd
    args = parse_cmd()
    # write
    with zopen(args["file"]) as fd:
        for num_lines, out_name0 in zip(args["nums"], args["outs"]):
            out_name = args["prefix"] + out_name0 + args["suffix"]
            with zopen(out_name, "w") as fdw:
                cur_num = 0
                while cur_num != num_lines:
                    line = fd.readline()
                    if len(line) == 0:
                        break
                    fdw.write(line)
                    cur_num += 1
                print("Write %s lines to %s." % (cur_num, out_name))
    print("over")

if __name__ == '__main__':
    main()
