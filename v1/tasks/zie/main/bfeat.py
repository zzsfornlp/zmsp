#

# extract bert features and store

from msp import utils
from msp.utils import zlog, zopen, PickleRW

from ..common.confs import OverallConf, init_everything, build_model, get_berter
from ..common.data import get_data_reader, BerterDataAuger

#
def main(args):
    conf: OverallConf = init_everything(args)
    dconf = conf.dconf
    #
    bmodel = get_berter(dconf.bconf)
    for one_input, one_output in zip([dconf.train, dconf.dev, dconf.test],
                                     [dconf.aux_repr_train, dconf.aux_repr_dev, dconf.aux_repr_test]):
        zlog(f"Read from {one_input} and write to {one_output}")
        num_doc, num_sent = 0, 0
        if one_input and one_output:
            one_streamer = get_data_reader(one_input, dconf.input_format, dconf.use_label0, dconf.noef_link0, None)
            bertaug_streamer = BerterDataAuger(one_streamer, bmodel, "aux_repr")
            with zopen(one_output, 'wb') as fd:
                for one_doc in bertaug_streamer:
                    PickleRW.save_list([s.extra_features["aux_repr"] for s in one_doc.sents], fd)
                    num_doc += 1
                    num_sent += len(one_doc.sents)
            zlog(f"Finish with doc={num_doc}, sent={num_sent}")
        else:
            zlog("Skip empty files")
    zlog("Finish all.")

"""
SRC_DIR="../src/"
DATA_DIR="../data5/outputs_split/"
DATA_SET=en.ere
for ws in train dev test; do
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zie.main.bfeat device:0 train:${DATA_DIR}/${DATA_SET}.${ws}.json aux_repr_train:_tmp.pkl
done
"""
