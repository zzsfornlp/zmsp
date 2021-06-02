#

from tasks.zdpar.main import train
from msp.utils import FileHelper
import os

def main():
    DATA_DIR = FileHelper.path_join(FileHelper.path_dirname(__file__), "..", "..", "testing", "zdpar", "data")
    os.chdir(DATA_DIR)
    args = ["partype:graph", "log_file:log"] + ["%s:en_ewt-ud-%s.conllu" % (s, s) for s in ["train", "dev", "test"]]
    args.append("no_build_dict:1")
    args.append("margin.init_val:1.")
    args.append("tconf.batch_size:4")
    # args.extend(["output_normalizing:single", "loss_single_sample:0.5", "loss_function:hinge"])
    # args.extend(["output_normalizing:global", "loss_function:prob", "dec_algorithm:unproj"])
    # args.extend(["output_normalizing:global", "loss_function:prob", "dec_algorithm:proj"])
    # args.extend(["output_normalizing:hlocal", "loss_function:prob", "dec_algorithm:proj"])
    args.extend(["output_normalizing:local", "loss_function:prob", "dec_algorithm:proj"])
    # args.extend(["output_normalizing:local", "loss_function:mr", "dec_algorithm:proj"])
    #
    # args.append("code_train:en")
    # debug
    args.append("train:en_train_debug.conllu")
    args.append("dev:en_train_debug.conllu")
    args.append("test:en_train_debug.conllu")
    # testing other architectures
    args.extend("enc_rnn_layer:0 enc_att_layer:6 enc_att_rel_clip:10 enc_att_rel_neg:0 enc_hidden:350 dim_char:0 dim_posi:0 enc_hidden:350 lrate.init_val:0.0001 lrate_warmup:-10 enc_att_add_wrapper:addtanh enc_att_type:mh enc_att_use_ranges:1".split())
    #
    train.main(args)
    pass

if __name__ == '__main__':
    main()

# various strategies
"""
for alg in unproj greedy proj; do
for out_norm in "local" single global; do
for loss in prob hinge; do
for margin in 0. 1. 2.; do
echo RUNNING ${alg} ${out_norm} ${loss} ${margin};
PYTHONPATH=../src python3 ../src/tasks/cmd.py zdpar.main.train _conf device:0 margin.init_val:${margin} dec_algorithm:${alg} output_normalizing:${out_norm} loss_function:${loss}
done
done
done
done
"""
