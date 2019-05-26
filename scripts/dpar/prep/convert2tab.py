#!/bin/python3

# convert '\t'-separated files to tab

import sys

def main(selects):
    sep = '\t'
    removing = ['_']
    for line in sys.stdin:
        fields = line.strip().split(sep)
        if selects:
            fields = [fields[idx] if idx<len(fields) else removing[0] for idx in selects]
        remains = list(filter(lambda x: x not in removing, fields))
        sys.stdout.write(sep.join(remains)+"\n")

# python3 *.py [1,2,4,...] <in >out
if __name__ == '__main__':
    try:
        sels = sys.argv[1].split(",")
        sels = [int(x) for x in sels]
    except:
        sels = None
    main(sels)
