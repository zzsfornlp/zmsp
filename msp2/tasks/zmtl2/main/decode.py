#

# test things in a stream mode! (from stdin to stdout)
import os
import sys
import json
from collections import Counter
from msp2.utils import zlog, zwarn, init_everything, Timer
from msp2.data.inst import Sent
from ..drive import *
from ..zmod import ZModel
from ..core import ZDatasetConf, ZDataset, ZDataPreprocessor

# --
def yield_lines(fd, batch: int):
    ret = []
    for line in fd:
        ret.append(line)
        if len(ret) == batch:
            yield ret
            ret = []
    if len(ret) > 0:
        yield ret
# --

# --
def main(args):
    # conf
    conf: ZOverallConf = init_everything(ZOverallConf(), args)
    # task
    t_center = TaskCenter(conf.tconf)
    # data
    d_center = DataCenter(conf.dconf, specified_wset=[])  # nothing to load here!
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
    # =====
    cc = Counter()
    BATCH_LINE = os.environ.get('ZMSP_BATCH_LINE', 1000)  # 1000 sents once time
    test_dataset = ZDataset(d_center.conf.testM, 'testM', 'decode', _no_load=True)  # use testM for other options!
    for lines in yield_lines(sys.stdin, BATCH_LINE):
        insts = [Sent.create(one.split()) for one in lines]  # note: simply split as sentence!!
        test_dataset.set_insts(insts)  # directly set it!
        cc["sent"] += len(insts)
        if cc["sent"] % 50000 == 0:
            zlog(f"Decode for {cc}")
        # --
        t_center.prepare_datasets([test_dataset])  # re-prepare!!
        for ibatch in test_dataset.yield_batches(loop=False):
            one_res = model.predict_on_batch(ibatch)
        # --
        for inst in insts:
            sys.stdout.write(json.dumps(inst.to_json(), ensure_ascii=False) + "\n")
    # =====
    zlog(f"The end of Decoding: {cc}")

# --
# MDIR=??
# PYTHONPATH=../src/ CUDA_VISIBLE_DEVICES=0 python3 -m msp2.tasks.zmtl2.main.decode ${MDIR}/_conf model_load_name:${MDIR}/zmodel.best.m vocab_load_dir:${MDIR}/ log_stderr:1 testM.group_tasks:??
if __name__ == '__main__':
    with Timer(info=f"Decoding", print_date=True) as et:
        main(sys.argv[1:])
