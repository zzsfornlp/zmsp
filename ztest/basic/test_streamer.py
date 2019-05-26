#

from msp.data import FAdapterStreamer, FileOrFdStreamer, IterStreamer, MultiCatStreamer, InstCacher, BatchArranger
from msp.utils import Helper

def main():
    s0 = IterStreamer(range(200))
    s1 = InstCacher(range(200), shuffle=True)
    s2 = InstCacher(MultiCatStreamer([IterStreamer(range(100, 200)), IterStreamer(range(100))]))
    s3 = BatchArranger(InstCacher(IterStreamer(range(200))), 8, 10, None, lambda x: x==48, None, lambda x: (x-24)**2, True)
    #
    nums = set(list(s0))
    for R in range(10):
        assert nums == set(list(s1))
        assert nums == set(list(s2))
        zz = list(s3)
        assert nums == set(Helper.join_list(zz) + [48])

if __name__ == '__main__':
    main()
