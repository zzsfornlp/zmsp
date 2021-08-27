#

# check c09's "Y" predicates

from collections import Counter

# --
def main(file):
    cc = Counter()
    prev_fields = []
    with open(file) as fd:
        for line in fd:
            fields = line.rstrip().split('\t')
            if len(fields) > 1:
                if int(fields[0]) == 1:
                    cc["sent"] += 1
                cc["tok"] += 1
                if fields[12] == "Y":
                    cc["pY"] += 1
                if fields[13] not in "-_":
                    cc["pS"] += 1
                # --
                if fields[12] == "Y" and fields[13] in "-_":
                    breakpoint()
                # --
                prev_fields.append(fields)
            else:
                prev_fields.clear()
    # --
    print(cc)

# --
# python3 check_c09.py FILE
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
