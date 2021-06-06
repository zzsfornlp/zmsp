#

# cut and join

# --
def main(fs: str, fs_plus=-1, out_sep="\t", sep=None):
    fs = [int(z) for z in fs.split(",")] if len(fs)>0 else []
    fs_plus = int(fs_plus)
    import sys
    for line in sys.stdin:
        line = line.rstrip()
        if len(line) == 0:
            sys.stdout.write("\n")
        else:
            fields = line.rstrip().split(sep)
            output_fields = [fields[z] for z in fs if z<len(fields)] + ([] if fs_plus<0 else fields[fs_plus:])
            output = out_sep.join(output_fields)
            sys.stdout.write(output+"\n")
# --

# python cut_join.py '' 1 <IN >OUT
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
