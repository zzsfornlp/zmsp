#

# main procedures: building, training, testing

from mspx.utils import Conf, init_everything, zlog, Timer, Random, Logger
from mspx.nn import BK
from mspx.proc.run import ZTaskCenterConf, ZTaskCenter, ZDataCenterConf, ZDataCenter, ZRunCenterConf, ZRunCenter

class ZOverallConf(Conf):
    def __init__(self):
        super().__init__()
        # --
        self.tconf = ZTaskCenterConf()  # task conf
        self.dconf = ZDataCenterConf()  # data conf
        self.rconf = ZRunCenterConf()  # run conf
        # --
        self.do_build = self.do_train = self.do_test = False
        self.fs = []  # build/train/test
        # --

# --
def main(args, sbase_getter=None):
    # conf
    conf: ZOverallConf = init_everything(ZOverallConf(), args, sbase_getter=sbase_getter)
    do_build, do_train, do_test = \
        [((z in conf.fs) or f) for z, f in zip(['build', 'train', 'test'], [conf.do_build, conf.do_train, conf.do_test])]
    # task
    t_center = ZTaskCenter(conf.tconf)
    # data
    dc_kwargs = {}
    if (not do_build) and (not do_train):
        dc_kwargs = {'eager_prefixes': ('test', )}
    d_center = ZDataCenter(conf.dconf, **dc_kwargs)
    # build vocabs
    _tcf = t_center.conf
    if do_build:
        with Timer(info=f"Main.Build", print_date=True):
            t_center.build_vocabs(
                d_center, try_load_vdir=(None if _tcf.vocab_force_rebuild else _tcf.vocab_load_dir),
                save_vdir=_tcf.vocab_save_dir)
    # main ones
    if do_train or do_test:
        _rcf = conf.rconf
        # load vocab
        if not do_build:
            t_center.load_vocabs(_tcf.vocab_load_dir)
        # train?
        if do_train:
            with Timer(info=f"Main.Train", print_date=True):
                t_center.make_model(preload_name=_rcf.train_preload_model, quite=False)
                BK.init_seed(Random.get_curr_seed())  # note: reset seed!
                r_center = ZRunCenter(conf.rconf, t_center, d_center)
                r_center.do_train()
        # test?
        if do_test:
            # --
            # note: add an extra log file!
            logger = Logger.get_singleton_logger()
            logs = [z for z in logger.log_files if isinstance(z, str)]
            if len(logs) > 0:
                adding = logs[0] + "_test"
                logger.add_log_files([adding])  # add one!
                zlog(f"Add log file for testing of {adding}")
            # --
            with Timer(info=f"Main.Test", print_date=True):
                if do_train:  # re-loading is fine
                    t_center.model.load(_rcf.model_load_name, quite=False)
                else:  # make a new one
                    t_center.make_model(load_name=_rcf.model_load_name, quite=False)
                BK.init_seed(Random.get_curr_seed())  # note: reset seed!
                r_center = ZRunCenter(conf.rconf, t_center, d_center)
                if _rcf.test_do_iter:
                    res = r_center.do_iter_test()
                else:
                    res = r_center.do_test()
                zlog(f"zzzzztestfinal: {res.to_dict()}")
                zlog(f"zzzzz---------: {res}")
    # --

# python3 -m mspx.cli.main ...
if __name__ == '__main__':
    import sys
    with Timer(info=f"Main", print_date=True) as et:
        main(sys.argv[1:])
    # --
