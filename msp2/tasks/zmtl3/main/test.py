#

# testing
from typing import List
import shlex
import pandas as pd
from msp2.utils import zlog, zwarn, init_everything, Timer, ZObject
from msp2.nn.l3 import Zmodel
from ..core import ZOverallConf, TaskCenter, DataCenter, RunCenter
from .conf import conf_getter_test

# --
def go_test(args, conf0: ZOverallConf):
    # conf
    if conf0 is not None:
        conf = conf0
    else:
        conf: ZOverallConf = init_everything(ZOverallConf(), args, sbase_getter=conf_getter_test)
    # task
    t_center = TaskCenter(conf.tconf)
    # data
    d_center = DataCenter(conf.dconf, greedy_wsets=["test"])
    # load vocab
    _tcf = t_center.conf
    if _tcf.vocab_force_rebuild:  # we may want to rebuild vocab in testing at some special cases
        t_center.build_vocabs(d_center, try_load_vdir=None, save_vdir=_tcf.vocab_save_dir)
    else:
        t_center.load_vocabs(_tcf.vocab_load_dir)
    # # prepare datasets: note: now we make it all lazy preparation!
    # t_center.prepare_datasets(d_center.get_datasets())
    # build model
    model = Zmodel(conf.mconf)
    t_center.build_mods(model)
    model.finish_build()  # note: build sr before possible loading in testing!!
    # run
    r_center = RunCenter(conf.rconf, model, t_center, d_center)
    if conf.rconf.model_load_name != "":
        r_center.load(conf.rconf.model_load_name)
    else:
        zwarn("No model to load, Debugging mode??")
    # --
    if conf.rconf.test_do_iter:
        res = r_center.do_iter_test()
    else:
        res = r_center.do_test()
    zlog(f"zzzztestfinal: {res}")
    # --
    zlog("The end of Testing.")
    return res
    # --

def main(args):
    conf0: ZOverallConf = init_everything(ZOverallConf(), args, sbase_getter=conf_getter_test)
    st_args = conf0.st_conf.st_args
    # --
    if not st_args:
        go_test(None, conf0)
        return
    # --
    # loop testing with different args
    all_res = []
    for sidx, st_arg in enumerate(st_args):
        zlog(f"ZRUN {st_arg}")
        cur_args = args + shlex.split(st_arg)
        cur_res = go_test(cur_args, None)
        cur_res.results['st'] = st_arg.split()[0]  # make it consice
        all_res.append(cur_res)
    # --
    res = {'st': [], }
    for sidx, cur_res in enumerate(all_res):
        # read the results
        res['st'].append(cur_res.results['st'])
        for k, v in cur_res.results.items():
            try:
                rr = round(float(v['zres']), 4)  # gather the one zres
                if k not in res:
                    res[k] = []
                if len(res[k]) < sidx:
                    res[k].extend([-1.] * (sidx - len(res[k])))
                res[k].append(rr)
            except:
                continue
    df = pd.DataFrame(res)
    zlog(f"# After all decodings: --\n{df.to_string()}")
    df2 = df.copy()
    del df2['st']
    max_ones = [(str(k), int(v)) for k,v in df2.idxmax().items()]
    df3 = df.iloc[[z[1] for z in max_ones]].copy()
    df3.insert(0, 'WSET', [z[0] for z in max_ones])
    zlog(f"# Maximum ones: --\n{df3.to_string()}")
    # breakpoint()
    # --

if __name__ == '__main__':
    import sys
    with Timer(info=f"Testing", print_date=True) as et:
        main(sys.argv[1:])
    # --
