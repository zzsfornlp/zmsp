#

# pre-train an o1 score and pre-compute the scores
# for training, using leave-one-out cv styles

# only a script, the real trainings are called with sub programs

import sys, os, math
import pickle
from typing import List
from msp.utils import Conf, zlog, FileHelper, system, Timer, Random, Helper, zopen
from msp.zext.dpar.conllu_reader import write_conllu
from ..common.data import get_data_reader, ParseInstance

#
class PsConf(Conf):
    def __init__(self):
        self.train = ""
        self.dev = ""
        self.test = ""
        self.pretrain_file = ""
        self.pieces = 10  # split training into 10 pieces
        self.use_pos = True
        self.max_epoch = 300
        self.reg_scores_lambda = 0.
        self.cur_run = 1
        # self.src_dir = "../../src/"
        # self.data_dir = "../../data/UD_RUN/ud23/"

# basic options
def get_base_opt(conf_name, model_name, use_pos, build_dict, max_epoch, reg_scores_lambda, cur_run):
    # TODO(+N): have we determined the hps here?
    base_opt = f"conf_output:{conf_name} model_name:{model_name} log_file:{conf_name}.log"
    base_opt += " init_from_pretrain:1 drop_embed:0.33 dropmd_embed:0.33 singleton_thr:2 fix_drop:1"
    base_opt += f" lrate.val:0.001 tconf.batch_size:32 lrate.m:0.75 min_save_epochs:150 max_epochs:{max_epoch} patience:8 anneal_times:10"
    base_opt += f" margin.val:0.5 margin.max_val:0.5 margin.mode:linear margin.b:0. margin.k:0.01 reg_scores_lambda:{reg_scores_lambda}"
    base_opt += " arc_space:512 lab_space:128 transform_act:elu biaffine_div:1. biaffine_init_ortho:1"
    base_opt += " enc_lrf.val:1. enc_lrf.mode:none enc_lrf.start_bias:0 enc_lrf.b:0. enc_lrf.k:1. enc_optim.weight_decay:0."
    base_opt += " dec_lrf.val:0. dec_lrf.mode:none dec_lrf.start_bias:0 dec_lrf.b:0. dec_lrf.k:1. dec_optim.weight_decay:0."
    base_opt += f" niconf.random_seed:9347{cur_run} niconf.random_cuda_seed:9349{cur_run}"
    # use the original graph
    # base_opt += " loss_function:hinge dec_algorithm:unproj output_normalizing:global"
    # base_opt += " partype:graph"
    # or use the g1p
    base_opt += " partype:g1"
    if not use_pos:
        base_opt += " dim_extras:[] extra_names:[]"
    if not build_dict:
        base_opt += " no_build_dict:1"
    return base_opt

# settings
# todo(WARN): fixed here!
SRC_DIR="../../src/"
DATA_DIR="../../data/UD_RUN/ud23/"

def get_data(path):
    if FileHelper.exists(path):
        return path
    else:
        return FileHelper.path_join(DATA_DIR, path)

def get_train_cmd(rgpu, base_opt, path_train, path_dev, path_test, pretrain_file):
    return f'CUDA_VISIBLE_DEVICES={rgpu} PYTHONPATH={SRC_DIR} python3 {SRC_DIR}/tasks/cmd.py zdpar.main.train train:{path_train} dev:{path_dev} test:{path_test} pretrain_file:{pretrain_file} device:0 {base_opt} niconf.random_seed:93471 niconf.random_cuda_seed:93491'

def get_score_cmd(rgpu, path_conf, path_model, path_test, path_output):
    return f'CUDA_VISIBLE_DEVICES={rgpu} PYTHONPATH={SRC_DIR} python3 {SRC_DIR}/tasks/cmd.py zdpar.main.score {path_conf} test:{path_test} device:0 model_load_name:{path_model} output_file:{path_output} log_file:{path_conf+".slog"}'

def write_insts(filename, insts: List[ParseInstance]):
    with zopen(filename, "w") as fd:
        for inst in insts:
            write_conllu(fd, *(inst.get_real_values_select(["words", "poses", "heads", "labels"])))

# zopen=open, import pickle, x=read_results("../scores/en/dev.scores.pkl")
def read_results(fname):
    results = []
    with zopen(fname, "rb") as fd:
        while True:
            try:
                one = pickle.load(fd)
                results.append(one)
            except EOFError:
                break
    return results

def write_results(fname, results):
    with zopen(fname, "wb") as fd:
        for one in results:
            pickle.dump(one, fd)

#
def main(args):
    conf = PsConf()
    conf.update_from_args(args)
    # read the data
    path_train, path_dev, path_test = [get_data(z) for z in [conf.train, conf.dev, conf.test]]
    pretrain_file = get_data(conf.pretrain_file)
    train_insts = list(get_data_reader(path_train, "conllu", "", False, ""))
    dev_insts = list(get_data_reader(path_dev, "conllu", "", False, ""))
    test_insts = list(get_data_reader(path_test, "conllu", "", False, ""))
    use_pos = conf.use_pos
    num_pieces = conf.pieces
    max_epoch = conf.max_epoch
    reg_scores_lambda = conf.reg_scores_lambda
    cur_run = conf.cur_run
    zlog(f"Read from train/dev/test: {len(train_insts)}/{len(dev_insts)}/{len(test_insts)}, split train into {num_pieces}")
    # others
    RGPU = os.getenv("RGPU", "")
    # first train on all: 1. get dict (only build once), 2: score dev/test
    with Timer("train", "Train-ALL"):
        cur_conf, cur_model = "_conf.all", "_model.all"
        cur_load_model = cur_model + ".best"
        cur_base_opt = get_base_opt(cur_conf, cur_model, use_pos, True, max_epoch, reg_scores_lambda, cur_run)
        system(get_train_cmd(RGPU, cur_base_opt, path_train, path_dev, path_test, pretrain_file), pp=True)
        system(get_score_cmd(RGPU, cur_conf, cur_load_model, path_dev, "dev.scores.pkl"), pp=True)
        system(get_score_cmd(RGPU, cur_conf, cur_load_model, path_test, "test.scores.pkl"), pp=True)
    # then training on the pieces (leaving one out)
    # first split into pieces
    Random.shuffle(train_insts)
    piece_length = math.ceil(len(train_insts) / num_pieces)
    train_pieces = []
    cur_idx = 0
    while cur_idx < len(train_insts):
        next_idx = min(len(train_insts), cur_idx+piece_length)
        train_pieces.append(train_insts[cur_idx:next_idx])
        cur_idx = next_idx
    zlog(f"Split training into {num_pieces}: {[len(x) for x in train_pieces]}")
    assert len(train_pieces) == num_pieces
    # next train each of the pieces
    for piece_id in range(num_pieces):
        with Timer("train", f"Train-{piece_id}"):
            # get current training pieces
            cur_training_insts = Helper.join_list([train_pieces[x] for x in range(num_pieces) if x!=piece_id])
            cur_testing_insts = train_pieces[piece_id]
            # write files
            cur_path_train, cur_path_test = f"tmp.train.{piece_id}.conllu", f"tmp.test.{piece_id}.conllu"
            write_insts(cur_path_train, cur_training_insts)
            write_insts(cur_path_test, cur_testing_insts)
            cur_conf, cur_model = f"_conf.{piece_id}", f"_model.{piece_id}"
            cur_load_model = cur_model + ".best"
            # no build dict, reuse previous
            cur_base_opt = get_base_opt(cur_conf, cur_model, use_pos, False, max_epoch, reg_scores_lambda, cur_run)
            system(get_train_cmd(RGPU, cur_base_opt, cur_path_train, path_dev, cur_path_test, pretrain_file), pp=True)
            system(get_score_cmd(RGPU, cur_conf, cur_load_model, cur_path_test, f"tmp.test.{piece_id}.scores.pkl"), pp=True)
    # finally put them in order
    all_results = []
    for piece_id in range(num_pieces):
        all_results.extend(read_results(f"tmp.test.{piece_id}.scores.pkl"))
    # reorder to the original order
    orig_indexes = [z.inst_idx for z in train_insts]
    orig_results = [None] * len(orig_indexes)
    for new_idx, orig_idx in enumerate(orig_indexes):
        assert orig_results[orig_idx] is None
        orig_results[orig_idx] = all_results[new_idx]
    # saving
    write_results("train.scores.pkl", orig_results)
    zlog("The end.")

# run
"""
SRC_DIR="../../src/"
CUR_LANG=en
CUR_RUN=1
RGPU=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.score_go train:${CUR_LANG}_train.conllu dev:${CUR_LANG}_dev.conllu test:${CUR_LANG}_test.conllu pretrain_file:wiki.${CUR_LANG}.filtered.vec cur_run:${CUR_RUN}
"""
