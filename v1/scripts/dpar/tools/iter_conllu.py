#

# simple script for itering conllu file
import sys

# todo(warn): a small stand-alone procedure!
# a generator that returns the stream of (orig_tokens, normed_words, pos, types)
def iter_file(fin):
    ret = {"len": 0, "word": [], "pos": [], "head": [], "type": []}
    for line in fin:
        line = line.strip()
        # yield and reset
        if len(line) == 0 or line[0] == "#":
            if ret["len"] > 0:
                yield ret
            ret = {"len": 0, "word": [], "pos": [], "head": [], "type": []}
        else:
            fields = line.split('\t')
            # skip special lines
            try:
                idx = int(fields[0])
            except:
                continue
            #
            ret["len"] += 1
            ret["word"].append(fields[1])
            ret["pos"].append(fields[3])
            ret["head"].append(int(fields[6]))
            ret["type"].append(fields[7])
    if ret["len"] > 0:
        yield ret

# python3 iter_conllu.py <*.conllu >*.txt
def main():
    for one_parse in iter_file(sys.stdin):
        sys.stdout.write(" ".join(one_parse["word"])+"\n")

if __name__ == '__main__':
    main()
