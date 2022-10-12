#
import sys

RR = """
"""

def main(file_in, file_out):
    with open(file_in) as fd1, open(file_out, 'w') as fd2:
        text = fd1.read()
        for line in RR.split():
            r1, r2 = line.strip().split(',')
            text = text.replace(r1, r2)
        fd2.write(text)
    # --

if __name__ == '__main__':
    main(*sys.argv[1:])
