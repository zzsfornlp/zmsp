#

# sample/shuffle lines

from msp2.utils import zlog, zopen, Conf, Random

# --
class MainConf(Conf):
    def __init__(self):
        self.input = ""
        self.output = ""
        self.skip_blank = False  # skip blank line
        self.shuffle = False
        self.shuffle_times = 0  # shuffle times?
        self.rate = 1.  # <=1. as rate, >1 as number

# --
def main(args):
    conf = MainConf()
    conf.update_from_args(args)
    # input
    with zopen(conf.input) as fd:
        lines = list(fd)
        if conf.skip_blank:
            lines = [z for z in lines if str.isspace(z)]
    # shuffle?
    origin_len = len(lines)
    if conf.shuffle_times>0 or conf.shuffle:
        _t = max(1, conf.shuffle_times)  # at least once!
        _gen = Random.get_generator('')
        for _ in range(_t):
            _gen.shuffle(lines)
    # sample?
    final_size = int(0.999 + (conf.rate * origin_len if conf.rate<=1. else conf.rate))
    out_lines = lines[:final_size]
    # output
    if conf.output:
        with zopen(conf.output, 'w') as fd2:
            for line in out_lines:
                fd2.write(line)
    # --
    zlog(f"Sample({conf.rate}) {conf.input}=>{conf.output}: {origin_len}=>{len(out_lines)}")

# --
# PYTHONPATH=../src/ python3 sample_shuffle.py input:?? output:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
