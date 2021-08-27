#

# testing
from msp2.utils import zlog, zwarn, init_everything, Timer
from ..drive import *
from ..zmod import ZModel

# --
def main(args):
    # conf
    conf: ZOverallConf = init_everything(ZOverallConf(), args)
    # task
    t_center = TaskCenter(conf.tconf)
    # data
    d_center = DataCenter(conf.dconf, specified_wset=["test"])
    # load vocab
    t_center.load_vocabs(t_center.conf.vocab_load_dir)
    # prepare datasets
    t_center.prepare_datasets(d_center.get_datasets())
    # build model
    model = ZModel(conf.mconf)
    t_center.build_mods(model)
    model.finish_sr()  # note: build sr before possible loading in testing!!
    # run
    r_center = RunCenter(conf.rconf, model, t_center, d_center)
    if conf.rconf.model_load_name != "":
        r_center.load(conf.rconf.model_load_name)
    else:
        zwarn("No model to load, Debugging mode??")
    res = r_center.do_test()
    zlog(f"zzzztestfinal: {res}")
    # --
    zlog("The end of Testing.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Testing", print_date=True) as et:
        main(sys.argv[1:])
