#

# build vocabs for tasks
from msp2.utils import zlog, zwarn, init_everything, Timer
from ..drive import *

# --
def main(args):
    # conf
    conf: ZOverallConf = init_everything(ZOverallConf(), args)
    # task
    t_center = TaskCenter(conf.tconf)
    # data
    d_center = DataCenter(conf.dconf)
    # build vocab: no try loading here!!
    t_center.build_vocabs(d_center)
    # save vocab
    t_center.save_vocabs(t_center.conf.vocab_save_dir)
    # --
    zlog("The end of Building.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Building", print_date=True) as et:
        main(sys.argv[1:])
