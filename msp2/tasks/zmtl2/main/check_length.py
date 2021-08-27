#

# check length

from collections import Counter
from msp2.data.vocab import SimpleVocab
from msp2.utils import zlog, zwarn, init_everything, Timer
from ..drive import *

# --
def main(args):
    # conf
    conf: ZOverallConf = init_everything(ZOverallConf(), args)
    # task
    t_center = TaskCenter(conf.tconf)
    enc = t_center.tasks['enc']
    # data
    d_center = DataCenter(conf.dconf)
    for dataset in d_center.get_datasets():
        enc.prepare_dataset(dataset)
        vv = SimpleVocab.build_by_static([])
        vv2 = SimpleVocab.build_by_static([])
        for item in dataset.items:
            vv.feed_one(item._batch_len)
            vv2.feed_one(sum(len(z) for z in item.sents)+1)
        vv.build_sort(lambda w, i, c: w)
        vv2.build_sort(lambda w, i, c: w)
        zlog(f"#== For {dataset} (subword):\n{vv.get_info_table().to_string()}")
        zlog(f"#== For {dataset} (word):\n{vv2.get_info_table().to_string()}")
    # --
    zlog("The end of Building.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"CheckLength", print_date=True) as et:
        main(sys.argv[1:])

# --
# python3 -m msp2.tasks.zmtl2.main.check_length train0.input_dir:ud train0.input_format:conllu train0.group_files:_ud14/en2 train0.approx_prev_next:1 train0.left_extend_nsent:1 train0.right_extend_nsent:1
