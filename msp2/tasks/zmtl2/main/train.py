#

# training
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
    d_center = DataCenter(conf.dconf)
    # build/load vocab: try loading here, and save new built ones!
    _tcf = t_center.conf
    t_center.build_vocabs(d_center, try_load_vdir=(None if _tcf.vocab_force_rebuild else _tcf.vocab_load_dir),
                          save_vdir=_tcf.vocab_save_dir)
    # prepare datasets
    t_center.prepare_datasets(d_center.get_datasets())
    # build model
    model = ZModel(conf.mconf)
    t_center.build_mods(model)
    # run
    r_center = RunCenter(conf.rconf, model, t_center, d_center)
    if conf.rconf.train_preload_model:
        r_center.load(conf.rconf.train_preload_model)
    model.finish_sr()  # note: build sr after possible loading in training!!
    r_center.do_train()
    # --
    zlog("The end of Training.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Training", print_date=True) as et:
        main(sys.argv[1:])
