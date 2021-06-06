#

# stat chars for the raw input
# -- used when looking at ar chars

import sys

# --
def main(file=None):
    from collections import Counter
    cc = Counter()
    if file is None:
        fd = sys.stdin
    else:
        fd = open(file)
    for line in fd:
        for c in line:
            if not str.isspace(c):
                cc[c] += 1
    # print
    count_sum = sum(cc.values())
    accu = 0.
    ii = 0
    for char, count in cc.most_common():
        accu += count/count_sum
        print(f"#{ii}: {hex(ord(char))}: {count}({count/count_sum:.4f},{accu:.4f})")
        ii += 1
    # --

# --
# python3 stat_chars.py <??
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
