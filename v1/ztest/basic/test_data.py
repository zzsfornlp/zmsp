#

from msp.data import VocabBuilder
from msp.data import TextReader
from msp.utils import Helper

def main():
    s = TextReader("./test_utils.py")
    vb = VocabBuilder("w")
    for one in s:
        vb.feed_stream(one.tokens)
    v = vb.finish()
    pass

if __name__ == '__main__':
    main()
