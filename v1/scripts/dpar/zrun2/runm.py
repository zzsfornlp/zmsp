#

# a single script handling the runnings of multiple modes

import argparse
import sys
import os

def printing(x, end='\n', file=sys.stderr, flush=True):
    print(x, end=end, file=file, flush=flush)

def system(cmd, pp=True, ass=True, popen=False):
    if pp:
        printing("Executing cmd: %s" % cmd)
    n = os.system(cmd)
    if ass:
        assert n==0, f"Executing previous cmd returns error {n}."
    return n

def parse():
    parser = argparse.ArgumentParser(description='Process running confs.')
    # basic settings
    # todo(note): be sure to make the dirs relative to run_dir
    parser.add_argument("--src_dir", type=str, default="../../src/")
    parser.add_argument("--data_dir", type=str, default="../../data/UD_RUN/ud24/")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--log_suffix", type=str, default=".log")
    # env
    parser.add_argument("--cur_run", type=int, default=1)
    parser.add_argument("--rgpu", type=int, required=True)  # -1 for CPU, >=0 for GPU
    # languages
    parser.add_argument("--cross", type=int, default=1)  # whether cross-lingual (fix embeddings, etc)
    parser.add_argument("--langs", type=str, nargs="+", required=True)
    parser.add_argument("--train_cuts", type=str, nargs="+", required=True)
    parser.add_argument("--dev_cuts", type=str, nargs="+", required=True)
    # specific for cr-embeds and mbert features
    parser.add_argument("--use_word", type=int, default=1)  # whether use cr word embeds
    parser.add_argument("--use_mb", type=int, default=0)
    parser.add_argument("--mb_suffix", type=str, default="mb48.pkl")
    parser.add_argument("--mb_fold", type=int, default=5)
    # pos mode
    parser.add_argument("--pos_mode", type=str, default="pred", choices=["pred", "nope", "gold"])
    # model
    parser.add_argument("--model", type=str, choices=["g1", "ef"], required=True)
    # enc type
    parser.add_argument("--enc_type", type=str, default="att", choices=["att", "rnn"])
    # extras
    parser.add_argument("--extras", type=str, default="")
    #
    args = parser.parse_args()
    return args

def main():
    printing(f"Run with {sys.argv}")
    args = parse()
    for k in sorted(dir(args)):
        if not k.startswith("_"):
            printing(f"--{k} = {getattr(args, k)}")
    #
    RGPU = args.rgpu
    CUR_RUN = args.cur_run
    SRC_DIR = args.src_dir
    DATA_DIR = args.data_dir
    LOG_SUFFIX = args.log_suffix
    # ===== step 1
    # mkdir and cd into it
    run_dir = args.run_dir
    if run_dir is None:
        # todo(note): by default, simply changing the script name
        fields = __file__.split(".")
        assert fields[-1] == "py"
        run_dir = ".".join(fields[:-1]) + "_run" + str(CUR_RUN)
    system(f"mkdir -p {run_dir}")
    os.chdir(run_dir)
    # ===== step 2
    printing(f"Run in {run_dir} with RGPU={RGPU} CUR_LANG={args.langs} CUR_RUN={CUR_RUN}")
    #
    # =====
    # basis
    base_opt = "conf_output:_conf init_from_pretrain:1 drop_embed:0.2 dropmd_embed:0.2 singleton_thr:2 fix_drop:1"
    if args.cross:
        # == for cross-lingual settings
        # todo(note): cannot let 0. as freezed embeddings since this will make layer-norm NAN; assign a smaller range instead
        base_opt += " emb_conf.dim_char:0 emb_conf.word_freeze:1 use_label0:1 pretrain_init_nohit:0.1"
    # ==
    base_opt += " lrate.val:0.001 tconf.batch_size:32 lrate.m:0.75 min_save_epochs:100 max_epochs:300 patience:8 anneal_times:10"
    base_opt += " loss_function:hinge margin.val:0.5 margin.max_val:0.5 margin.mode:linear margin.b:0. margin.k:0.01 reg_scores_lambda:0."
    base_opt += " arc_space:512 lab_space:128 transform_act:elu biaffine_div:1. biaffine_init_ortho:1"
    base_opt += " enc_lrf.val:1. enc_lrf.mode:none enc_lrf.start_bias:0 enc_lrf.b:0. enc_lrf.k:1. enc_optim.weight_decay:0."
    # use the previous setting and fix all
    base_opt += " dec_lrf.val:0. dec_lrf.mode:none dec_lrf.start_bias:0 dec_lrf.b:0. dec_lrf.k:1. dec_optim.weight_decay:0."
    base_opt += " dec2_lrf.val:0. dec2_lrf.mode:none dec2_lrf.start_bias:0 dec2_lrf.b:0. dec2_lrf.k:1. dec2_optim.weight_decay:0."
    # =====
    # which system
    if args.model == "g1":
        # todo(note): simply chaning this to "graph" is also ok
        base_opt += " partype:g1"
    elif args.model == "ef":
        # =====
        # ef system
        base_opt += " aabs.val:5. aabs.max_val:5. aabs.mode:linear aabs.start_bias:100 aabs.k:0.01"
        arc_k = 1
        # system
        base_opt += " partype:ef zero_extra_output_params:0 iconf.ef_mode:free tconf.ef_mode:free system_labeled:1"
        base_opt += " chs_num:10 chs_att.out_act:linear chs_att.att_dropout:0.1 chs_att.head_count:2"
        # loss
        base_opt += " tconf.ending_mode:maxv cost0_weight.val:1. loss_div_weights:0 loss_div_fullbatch:1"
        # features for ef
        base_opt += " fdrop_chs:0. fdrop_par:0. use_label_feat:1 use_chs:1 use_par:1 ef_ignore_chs:func"
        # search
        # arc_k=1  # set previously
        lab_k = 1
        base_opt += f" iconf.plain_k_arc:{arc_k} iconf.plain_k_label:{lab_k} iconf.plain_beam_size:{arc_k}"
        base_opt += f" tconf.plain_k_arc:{arc_k} tconf.plain_k_label:{lab_k} tconf.plain_beam_size:{arc_k}"
    else:
        raise NotImplementedError(f"UNK model {args.model}")
    # =====
    # which encoder
    base_opt += " emb_proj_dim:512 enc_conf.enc_hidden:512"
    if args.enc_type == "att":
        # specific for self-att encoder
        base_opt += " enc_conf.enc_rnn_layer:0 enc_conf.enc_att_layer:6 enc_conf.enc_att_add_wrapper:addnorm enc_conf.enc_att_conf.clip_dist:10 enc_conf.enc_att_conf.use_neg_dist:0 enc_conf.enc_att_conf.att_dropout:0.1 enc_conf.enc_att_conf.use_ranges:1 enc_conf.enc_att_fixed_ranges:5,10,20,30,50,100 enc_conf.enc_att_final_act:linear"
        base_opt += " max_epochs:300 train_skip_length:80 tconf.batch_size:80 split_batch:2 biaffine_div:0. margin.b:0. reg_scores_lambda:0.01"
        base_opt += " lrate.val:0.0001 lrate_warmup:-2 drop_embed:0.1 dropmd_embed:0. drop_hidden:0.1"
    elif args.enc_type == "rnn":
        pass
    else:
        raise NotImplementedError(f"UNK enc {args.enc_type}")
    # =====
    # inputs
    langs = args.langs
    train_cuts = args.train_cuts
    dev_cuts = args.dev_cuts
    assert len(langs) == len(train_cuts) and len(langs) == len(dev_cuts)
    printing(f"Langs: {langs}, train-cut: {train_cuts}, dev-cut: {dev_cuts}")
    # pos mode
    FILE_INFIX = ""
    if args.pos_mode == "nope":
        base_opt += " dim_extras:[] extra_names:[]"
    elif args.pos_mode == "pred":
        FILE_INFIX = ".ppos"
    elif args.pos_mode == "gold":
        pass
    else:
        raise NotImplementedError(f"UNK pos_mode {args.pos_mode}")
    # inputs; todo(note): currently using gold for train; test take full lang0 test file
    base_opt += f" train:" + ",".join([f"{DATA_DIR}/{cl}_train.conllu" for cl in langs])
    base_opt += f" dev:" + ",".join([f"{DATA_DIR}/{cl}_dev{FILE_INFIX}.conllu" for cl in langs])
    base_opt += f" test:{DATA_DIR}/{langs[0]}_test{FILE_INFIX}.conllu"
    base_opt += f" pretrain_file:" + ",".join([f"{DATA_DIR}/wiki.multi.{cl}.filtered.vec" for cl in langs])
    base_opt += f" multi_source:1 cut_train:{','.join([str(z) for z in train_cuts])} cut_dev:{','.join([str(z) for z in dev_cuts])} word_rthres:200000"
    # embeds and mbert features
    if args.use_word:
        pass
    else:
        base_opt += " dim_word:0"
    if args.use_mb:
        MB_SUFFIX = args.mb_suffix
        MB_FOLD = args.mb_fold
        base_opt += f" dim_auxes:[768] fold_auxes:[{MB_FOLD}]"
        base_opt += f" aux_repr_train:" + ",".join([f"{DATA_DIR}/{cl}_train.{MB_SUFFIX}" for cl in langs])
        base_opt += f" aux_repr_dev:" + ",".join([f"{DATA_DIR}/{cl}_dev.{MB_SUFFIX}" for cl in langs])
        base_opt += f" aux_repr_test:{DATA_DIR}/{langs[0]}_test.{MB_SUFFIX}"
        # todo(note): with bfeats, only smaller lrate works?
        base_opt += " lrate.val:0.0001"
    # different seed for different runnings
    base_opt += f" niconf.random_seed:9347{CUR_RUN} niconf.random_cuda_seed:9349{CUR_RUN}"
    # =====
    # get it run
    # training
    DEVICE = 0 if RGPU>=0 else -1
    system(f"CUDA_VISIBLE_DEVICES={RGPU} PYTHONPATH={SRC_DIR} python3 {SRC_DIR}/tasks/cmd.py zdpar.main.train device:{DEVICE} {base_opt} {args.extras} >_train{LOG_SUFFIX} 2>&1")
    # testing
    def _run_decode(decode_langs, which_set):
        dec_opt = ""
        dec_opt += f" test:{DATA_DIR}/${{cl}}_{which_set}{FILE_INFIX}.conllu"
        dec_opt += f" test_extra_pretrain_files:{DATA_DIR}/wiki.multi.${{cl}}.filtered.vec"
        dec_opt += f" output_file:crout_${{cl}}.{which_set}.out"
        if args.model == "ef":
            dec_opt += " aabs.mode:none"  # ensure use max beam size
        if args.use_mb:
            dec_opt += f" aux_repr_test:{DATA_DIR}/${{cl}}_{which_set}.{MB_SUFFIX}"
        system(f"for cl in {decode_langs}; do CUDA_VISIBLE_DEVICES={RGPU} PYTHONPATH={SRC_DIR} python3 {SRC_DIR}/tasks/cmd.py zdpar.main.test _conf conf_output: log_file: device:{DEVICE} {dec_opt} {args.extras}; done >_decode.{which_set}{LOG_SUFFIX} 2>&1")
    #
    DECODE_LANGS = "en ar eu zh 'fi' he hi it ja ko ru sv tr"
    _run_decode(DECODE_LANGS, "dev")
    _run_decode(DECODE_LANGS, "test")

if __name__ == '__main__':
    main()

# example
# python3 runm.py --rgpu 2 --model g1 --langs ru --train_cuts 12000 --dev_cuts 1000 --run_dir ?

# running batch 1 (pos, word, mbert?)
"""
# general run of the basic ones
RGPU=2
# =====
# run single source
for cur_model in ef; do
for cur_run in 1 2 3; do
for use_mb in 0 1; do
DIR_NAME="z${cur_model}_ru_b${use_mb}_${cur_run}"
python3 runm.py --rgpu $RGPU --model ${cur_model} --langs ru --train_cuts 12000 --dev_cuts 1000 --use_word 1 --use_mb ${use_mb} --pos_mode pred --run_dir $DIR_NAME --cur_run ${cur_run}
done; done; done
# =====
# run multi source
RGPU=2
for cur_model in g1 ef; do
for cur_run in 1 2 3; do
for use_mb in 0 1; do
DIR_NAME="z${cur_model}_er_b${use_mb}_${cur_run}"
python3 runm.py --rgpu $RGPU --model ${cur_model} --langs en ru --train_cuts 6000 6000 --dev_cuts 500 500 --use_word 1 --use_mb ${use_mb} --pos_mode pred --run_dir $DIR_NAME --cur_run ${cur_run}
done; done; done
# =====
# test other ef mode
python3 runm.py --rgpu 3 --model ef --langs en --train_cuts 12000 --dev_cuts 1000 --use_word 1 --use_mb 0 --pos_mode pred --run_dir test_t2d --cur_run 1 --extras "iconf.ef_mode:t2d tconf.ef_mode:t2d"
"""

# final running batch
"""
# =====
# -----
RGPU=2
# run single source
for cur_model in g1 ef; do
for cur_run in 1 2 3 4 5; do
for use_mb in 1; do
for cl in en ru; do
DIR_NAME="z${cur_model}_${cl}_${cur_run}"
python3 runm.py --rgpu $RGPU --model ${cur_model} --langs ${cl} --train_cuts 12000 --dev_cuts 1000 --use_word 1 --use_mb ${use_mb} --pos_mode pred --run_dir $DIR_NAME --cur_run ${cur_run}
done; done; done; done
# -----
RGPU=3
# run multi source
for cur_model in g1 ef; do
for cur_run in 1 2 3 4 5; do
for use_mb in 1; do
DIR_NAME="z${cur_model}_er_${cur_run}"
python3 runm.py --rgpu $RGPU --model ${cur_model} --langs en ru --train_cuts 6000 6000 --dev_cuts 500 500 --use_word 1 --use_mb ${use_mb} --pos_mode pred --run_dir $DIR_NAME --cur_run ${cur_run}
done; done; done
# -----
RGPU=3
# run other ef mode
for ef_mode in t2d n2f l2r r2l; do
for cur_run in 1 2 3; do
for cur_arg in "--run_dir zana_en_${ef_mode}_${cur_run} --langs en --train_cuts 12000 --dev_cuts 1000" "--run_dir zana_ru_${ef_mode}_${cur_run} --langs ru --train_cuts 12000 --dev_cuts 1000" "--run_dir zana_er_${ef_mode}_${cur_run} --langs en ru --train_cuts 6000 6000 --dev_cuts 500 500"; do
python3 runm.py --rgpu $RGPU --model ef --use_word 1 --use_mb 1 --pos_mode pred --cur_run ${cur_run} --extras "iconf.ef_mode:${ef_mode} tconf.ef_mode:${ef_mode}" ${cur_arg}
done; done; done
# -----
RGPU=2
# run with various features for ef
for cur_feat in "00" "01" "10"; do
for cur_run in 1 2 3; do
for cur_arg in "--run_dir zfeat_en_${cur_feat}_${cur_run} --langs en --train_cuts 12000 --dev_cuts 1000" "--run_dir zfeat_ru_${cur_feat}_${cur_run} --langs ru --train_cuts 12000 --dev_cuts 1000" "--run_dir zfeat_er_${cur_feat}_${cur_run} --langs en ru --train_cuts 6000 6000 --dev_cuts 500 500"; do
if [[ ${cur_feat} == "00" ]]; then
cur_feat_arg="sl_conf.use_chs:0 sl_conf.use_par:0";
elif [[ ${cur_feat} == "01" ]]; then
cur_feat_arg="sl_conf.use_chs:0 sl_conf.use_par:1";
elif [[ ${cur_feat} == "10" ]]; then
cur_feat_arg="sl_conf.use_chs:1 sl_conf.use_par:0";
fi
python3 runm.py --rgpu $RGPU --model ef --use_word 1 --use_mb 1 --pos_mode pred --cur_run ${cur_run} --extras "${cur_feat_arg}" ${cur_arg}
done; done; done
"""
