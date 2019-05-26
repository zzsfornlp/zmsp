#

# paste the fields of various files (in conll format)
# eg: python ~ -i f1 f2 -p 0:0 0:1 0:2 1:3 1:4 0:5 0:6 0:7 0:8 0:9

import argparse, sys, re

def parse_cmd(args):
    parser = argparse.ArgumentParser("Paste the fields.")
    parser.add_argument("-i", "--inputs", type=str, required=True, nargs='+', help="The input files")
    parser.add_argument("-o", "--output", type=str, help="Default to std-out.")
    parser.add_argument("-p", "--pieces", type=str, required=True, nargs='+', help="What to write on each field, 'file-dix:field-idx'")
    parser.add_argument("-s", "--sep", type=str)
    a = parser.parse_args(args)
    return a

def main(args):
    a = parse_cmd(args)
    #
    eof_f = lambda s: len(s)==0
    EMPTY_PATTERN = re.compile(r"(\s+)|(^\s*#.*)")     # whitespace or comment
    empty_f = lambda s: re.fullmatch(EMPTY_PATTERN, s)
    #
    input_fds = [open(f) for f in a.inputs]
    output_fd = open(a.output, "w") if a.output else sys.stdout
    input_sep = a.sep
    output_sep = a.sep if a.sep else "\t"
    pieces = [[int(z) for z in s.split(":")] for s in a.pieces]
    #
    while True:
        lines = [fd.readline() for fd in input_fds]
        # check end
        eofs = [eof_f(s) for s in lines]
        if all(eofs):       # EOS
            break
        assert not any(eofs), "Unequal lines!"
        # empty lines
        emptys = [empty_f(s) for s in lines]
        if all(emptys):
            output_fd.write("\n")
            continue
        assert not any(emptys), "Unmatched empty line!"
        # split
        fields = [s.strip().split(input_sep) for s in lines]
        entries = [fields[a][b] for a,b in pieces]
        output_fd.write(output_sep.join(entries)+"\n")
    #
    [fd.close() for fd in input_fds]
    output_fd.close()

if __name__ == '__main__':
    main(sys.argv[1:])
