#

# (updated one) a single script handling the runnings of multiple modes

import argparse
import sys
import os

SCRIPT_ABSPATH = os.path.abspath(__file__)

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
    # =====
    # -- dirs
    # todo(note): be sure to make base_dir is relative to run_dir and others are relative to base_dir
    parser.add_argument("--run_dir", type=str, required=True)
    # --
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--src_dir", type=str, default="src")
    parser.add_argument("--dict_dir", type=str, default="data/UD_RUN/vocabs")
    parser.add_argument("--embed_dir", type=str, default="data/UD_RUN/embeds")
    parser.add_argument("--ud_dir", type=str, default="data/UD_RUN/ud24s")
    parser.add_argument("--ner_dir", type=str, default="data/ner")
    parser.add_argument("--wiki_dir", type=str, default="data/UD_RUN/wikis")
    # =====
    # -- basics
    # running
    parser.add_argument("--run_mode", type=str, required=True)  # pre/ppp0/ppp1/...
    parser.add_argument("--pre_mode", type=str, default="orp")  # plm/mlm/orp
    parser.add_argument("--do_test", type=int, default=1, help="testing when training parser")
    parser.add_argument("--preload_prefix", type=str)  # pre-trained model for "par1"
    parser.add_argument("--pre_upe", type=int, default=1000, help="Update for fake epoch(valid)")
    # data
    parser.add_argument("-l", "--lang", type=str, required=True)
    parser.add_argument("--train_size", type=str, default="10k", help="Training size infix")
    # wvec and bert
    parser.add_argument("--use_wvec", type=int, default=0)
    parser.add_argument("--use_bert", type=int, default=0)
    parser.add_argument("--aug_wvec2", type=int, default=0)
    # model
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--head_count", type=int, default=8, help="Head count in MATT for trans/vrec")
    parser.add_argument("--enc_type", type=str, default="trans", choices=["vrec", "trans", "rnn"])
    parser.add_argument("--lm_pred_size", type=int, default=100000)
    parser.add_argument("--lm_tie_inputs", type=int, default=1)
    parser.add_argument("--use_biaffine", type=int, default=1)
    # env
    parser.add_argument("--cur_run", type=int, default=1)
    parser.add_argument("--rgpu", type=int, required=True)  # -1 for CPU, >=0 for GPU
    parser.add_argument("--debug", type=int, default=0)
    # extras
    parser.add_argument("--extras", type=str, default="")
    # -----
    args = parser.parse_args()
    return args

# --
def step0_dir(args):
    printing("# =====\n=> Step 0: determine dirs")
    run_dir = args.run_dir
    system(f"mkdir -p {run_dir}")
    os.chdir(run_dir)
    # check out dirs based on src
    base_dir = args.base_dir
    if base_dir is None:
        # search for src dir
        base_dir = ""
        _MAX_TIME = 4
        while not os.path.isdir(os.path.join(base_dir, args.src_dir)) and _MAX_TIME > 0:
            base_dir = os.path.join(base_dir, "..")  # upper one layer
            _MAX_TIME -= 1
        src_dir = os.path.join(base_dir, args.src_dir)
    else:
        src_dir = os.path.join(base_dir, args.src_dir)
    assert os.path.isdir(src_dir), f"Failed to find src dir: {args.src_dir}"
    printing(f"Finally run_dir={os.getcwd()}, base_dir={base_dir}, src_dir={src_dir}")
    dict_dir = os.path.join(base_dir, args.dict_dir)
    embed_dir = os.path.join(base_dir, args.embed_dir)
    ud_dir = os.path.join(base_dir, args.ud_dir)
    wiki_dir = os.path.join(base_dir, args.wiki_dir)
    ner_dir = os.path.join(base_dir, args.ner_dir)
    return run_dir, src_dir, dict_dir, embed_dir, ud_dir, wiki_dir, ner_dir

# -----
# todo(note): by default for par0
def get_basic_options(args):
    MODEL_DIM = args.model_dim
    ENC_TYPE = args.enc_type
    HEAD_COUNT = args.head_count
    HEAD_DIM = MODEL_DIM // HEAD_COUNT
    # =====
    base_opt = "conf_output:_conf"
    # => training and lrate schedule
    base_opt += " lrate.val:0.0004 lrate_warmup:250 lrate.min_val:0.000001 lrate.m:0.75"  # default lrate
    base_opt += " min_save_epochs:0 max_epochs:250 patience:8 anneal_times:10"
    base_opt += " train_min_length:5 train_max_length:80 train_batch_size:80 split_batch:5 test_batch_size:8"
    # => dropout
    base_opt += " drop_hidden:0.1 gdrop_rnn:0.33 idrop_rnn:0.33 fix_drop:0"
    # => model
    # embedder
    base_opt += " ec_word.comp_dim:300 ec_word.comp_drop:0.1 ec_word.comp_init_scale:1. ec_word.comp_init_from_pretrain:0"
    base_opt += " ec_char.comp_dim:50 ec_char.comp_drop:0.1 ec_char.comp_init_scale:1."
    # (NOPOS!!) base_opt+=" ec_pos.comp_dim:50 ec_pos.comp_drop:0.1 ec_pos.comp_init_scale:1."
    base_opt += f" emb_proj_dim:{MODEL_DIM} emb_proj_act:linear"  # proj after embedder
    # encoder
    if ENC_TYPE == "rnn":
        # RNN baseline
        base_opt += " matt_conf.head_count:256 default_attn_count:256 prepr_choice:rdist"  # this is not used, simply avoid err
        base_opt += f" enc_choice:original oenc_conf.enc_hidden:{MODEL_DIM} oenc_conf.enc_rnn_layer:3 oenc_conf.enc_rnn_sep_bidirection:1"
    else:
        # use VRec
        base_opt += " enc_choice:vrec"
        base_opt += " venc_conf.num_layer:6 venc_conf.attn_ranges:"  # vrec layers
        base_opt += " use_pre_norm:0 use_post_norm:1"  # layer norm
        # # no scale now!!
        # base_opt += " scorer_conf.param_init_scale:0.5 collector_conf.param_init_scale:0.5"  # other init
        # base_opt += " ec_word.comp_init_scale:0.5 ec_char.comp_init_scale:0.5"  # other init
        # use MAtt
        base_opt += f" matt_conf.head_count:{HEAD_COUNT}"  # head count
        base_opt += f" scorer_conf.d_qk:{HEAD_DIM} scorer_conf.use_rel_dist:1 scorer_conf.no_self_loop:1"  # matt scorer
        base_opt += f" normer_conf.use_noop:1 normer_conf.noop_fixed_val:0. normer_conf.norm_mode:cand normer_conf.attn_dropout:0.1"  # matt norm
        base_opt += f" collector_conf.d_v:{HEAD_DIM} collector_conf.collect_mode:ch collect_reducer_mode:aff collect_red_aff_out:{MODEL_DIM}"  # matt col
        if ENC_TYPE == "trans":
            base_opt += f" venc_conf.share_layers:0 comb_mode:affine ff_dim:{MODEL_DIM*2}"  # no share
        elif ENC_TYPE == "vrec":
            base_opt += " venc_conf.share_layers:1 comb_mode:lstm ff_dim:0"  # share layers
        else:
            raise NotImplementedError()
    # => prediction & loss
    # vocab
    base_opt += " lower_case:0 norm_digit:1 word_rthres:1000000 word_fthres:1"  # vocab
    base_opt += " ec_word.comp_rare_unk:0.5 ec_word.comp_rare_thr:10"  # word rare unk in training
    # masklm
    base_opt += " mlm_conf.loss_lambda.val:0. mlm_conf.max_pred_rank:100"  # masklm
    base_opt += " mlm_conf.hid_dim:300 mlm_conf.hid_act:elu"  # hid
    base_opt += f" mlm_conf.tie_input_embeddings:{args.lm_tie_inputs}"  # tie embeddings
    # plainlm
    base_opt += " plm_conf.loss_lambda.val:0. plm_conf.max_pred_rank:100"  # plainlm
    base_opt += " plm_conf.hid_dim:300 plm_conf.hid_act:elu"  # hid
    base_opt += f" plm_conf.tie_input_embeddings:{args.lm_tie_inputs}"  # tie embeddings
    # orderpr
    base_opt += " orp_conf.loss_lambda.val:0. orp_conf.disturb_mode:abs_bin orp_conf.bin_blur_bag_keep_rate:0."
    # base_opt += " orp_conf.disturb_ranges:8,10 orp_conf.bin_blur_bag_lbound:-2"
    base_opt += " orp_conf.disturb_ranges:7,8"  # original one
    # base_opt += " orp_conf.pred_range:1 orp_conf.cand_range:4 orp_conf.lambda_n1:0. orp_conf.lambda_n2:1."
    base_opt += " orp_conf.pred_range:1 orp_conf.cand_range:1000 orp_conf.lambda_n1:0. orp_conf.lambda_n2:1."  # AllCand
    base_opt += " orp_conf.ps_conf.use_input_pair:0"  # scorer
    # better ones for orp
    base_opt += " orp_conf.bin_blur_bag_keep_rate:0.5 orp_conf.pred_range:2" \
                " orp_conf.ps_conf.use_biaffine:1 orp_conf.ps_conf.use_ff2:0"
    # dpar
    base_opt += " dpar_conf.loss_lambda.val:0. dpar_conf.label_neg_rate:0.1"  # overall
    base_opt += " dpar_conf.pre_dp_space:512 dpar_conf.pre_dp_act:elu"  # pre-scoring
    base_opt += " dpar_conf.dps_conf.use_input_pair:0"  # scorer
    if args.use_biaffine:
        base_opt += " dpar_conf.dps_conf.use_biaffine:1 dpar_conf.dps_conf.use_ff1:1 dpar_conf.dps_conf.use_ff2:0"
    # pos
    base_opt += " upos_conf.loss_lambda.val:0."
    # --
    return base_opt

# ----
def find_model(model_prefix):
    model_dir = os.path.dirname(model_prefix)
    model_name_prefix = os.path.basename(model_prefix)
    model_name_full = None
    for f in os.listdir(model_dir):
        if f.endswith(".pr.json") and f.startswith(model_name_prefix):
            model_name_full = f[:-8]
    assert model_name_full is not None
    model_path = os.path.join(model_dir, model_name_full)
    printing(f"Find model at {model_path}")
    return model_path

# =====
def step_run(args, DIRs):
    run_dir, src_dir, dict_dir, embed_dir, ud_dir, wiki_dir, ner_dir = DIRs
    printing(f"# =====\n=> Step 1: running!!")
    # basic
    cur_opt = get_basic_options(args)
    # =====
    # specific mode
    CUR_LANG = args.lang
    RUN_MODE = args.run_mode
    PRE_MODE = args.pre_mode
    TRAIN_SIZE = args.train_size
    ENC_TYPE = args.enc_type
    # wvec & bert?
    if args.use_wvec:
        cur_opt += " read_from_pretrain:1 ec_word.comp_init_from_pretrain:1" \
                   f" pretrain_file:{embed_dir}/wiki.{CUR_LANG}.vec"
    if args.use_bert:
        cur_opt += " ec_word.comp_dim:0"  # no word if using bert (by default mbert)
        cur_opt += " ec_bert.comp_dim:768 ec_bert.comp_drop:0.1 ec_bert.comp_init_scale:1."
        cur_opt += " bert2_output_layers:4,5,6,7,8 bert2_training_mask_rate:0."
    if args.aug_wvec2:  # special mode!!
        cur_opt += f" aug_word2:1 aug_word2_pretrain:{embed_dir}/wiki.{CUR_LANG}.vec"
    # modes
    # -- pre-training
    if RUN_MODE == "pre":
        cur_opt += " min_save_epochs:0 max_epochs:200 patience:1000 anneal_times:5"
        cur_opt += f" train:{wiki_dir}/wiki_{CUR_LANG}.{TRAIN_SIZE}.txt dev: test: input_format:plain" \
                   f" cache_data:0 no_build_dict:1 dict_dir:{dict_dir}/voc_{CUR_LANG}/"
        # different pre-training tasks for different archs
        if PRE_MODE == "plm":
            cur_opt += f" plm_conf.loss_lambda.val:1. plm_conf.max_pred_rank:{args.lm_pred_size}"
            if ENC_TYPE != "rnn":
                cur_opt += f" plm_conf.split_input_blm:0"  # input list for vrec
        elif PRE_MODE == "mlm":
            cur_opt += f" mlm_conf.loss_lambda.val:1. mlm_conf.max_pred_rank:{args.lm_pred_size}"
            cur_opt += f" mlm_conf.mask_rate:0.15"
        elif PRE_MODE == "orp":
            cur_opt += " orp_conf.loss_lambda.val:1."
        elif PRE_MODE == "om":  # orp + mlm
            cur_opt += " orp_conf.loss_lambda.val:0.5 orp_conf.bin_blur_bag_keep_rate:0.5"  # keep for mlm
            cur_opt += f" mlm_conf.loss_lambda.val:0.5 mlm_conf.max_pred_rank:{args.lm_pred_size}"
            cur_opt += f" mlm_conf.mask_rate:0.15"
        else:
            raise NotImplementedError()
        # -----
        # # schedule
        # # (OldRun) when bs=80: (UPE*80 as one epoch): 600*CHECKPOINT seems to be enough (~50 epochs for 1m)
        # # (NewRun) when bs=240: (UPE*240 as one epoch): use 300*CHECKPOINT (~72 epochs)
        # UPE = 1000  # number of Update per Pseudo Epoch
        # cur_opt += " train_batch_size:240 split_batch:15"  # larger batch size for pre-training
        # cur_opt += f" report_freq:{UPE} valid_freq:{UPE} max_updates:{304*UPE}"
        # cur_opt += " save_freq:50 validate_epoch:0"  # save more points
        # cur_opt += f" lrate_warmup:{5*UPE} lrate.which_idx:uidx lrate.start_bias:{50*UPE} lrate.scale:{50*UPE}"
        # -----
        # schedule2
        # (NewRun2) when bs=480: (UPE*480 as one epoch): use 200*CHECKPOINT (~96 epochs)
        UPE = args.pre_upe  # number of Update per Pseudo Epoch
        cur_opt += " train_batch_size:480 split_batch:30"  # larger batch size for pre-training
        cur_opt += f" report_freq:{UPE} valid_freq:{UPE} max_updates:{201*UPE} save_freq:40 validate_epoch:0"
        cur_opt += f" lrate_warmup:{5*UPE} lrate.which_idx:uidx lrate.start_bias:{40*UPE} lrate.scale:{40*UPE} lrate.m:0.5"
    # -- parser/tagger training
    else:
        assert RUN_MODE in ["par0", "par1", "pos0", "pos1", "ppp0", "ppp1", "ner0", "ner1"]
        if RUN_MODE.startswith("ner"):
            cur_opt += " do_ner:1 input_format:ner div_by_tok:0"  # remember to add the extra flag!!
            cur_opt += f" train:{ner_dir}/{CUR_LANG}/train.{TRAIN_SIZE}.txt dev:{ner_dir}/{CUR_LANG}/dev.txt" \
                       f" test:{ner_dir}/{CUR_LANG}/test.txt"
            # extra settings for ner
            # cur_opt += " train_batch_size:80 split_batch:2 drop_hidden:0.5 max_epochs:10000000 validate_epoch:0 valid_freq:200" \
            #            " lrate_warmup:500 max_updates:30000 div_by_tok:0 ec_word.comp_drop:0.5 ec_char.comp_drop:0.5"
            # cur_opt += " drop_hidden:0.5 ec_word.comp_drop:0.5 ec_char.comp_drop:0.5"
        else:
            cur_opt += f" train:{ud_dir}/{CUR_LANG}_train.{TRAIN_SIZE}.conllu dev:{ud_dir}/{CUR_LANG}_dev.conllu" \
                       f" test:{ud_dir}/{CUR_LANG}_test.conllu"
        task_comp_names = {"par": ["dpar"], "pos": ["upos"], "ppp": ["dpar", "upos"], "ner": ["ner"]}[RUN_MODE[:3]]
        for one_tcn in task_comp_names:
            cur_opt += f" {one_tcn}_conf.loss_lambda.val:1."
        if RUN_MODE[-1] == "1":
            # use pretrain vocab
            cur_opt += f" no_build_dict:1 dict_dir:{dict_dir}/voc_{CUR_LANG}/"
            # find preload model
            cur_opt += f" load_pretrain_model_name:{find_model(args.preload_prefix)}"
            # use smaller lrate
            cur_opt += " lrate.val:0.0002"
        if ENC_TYPE == "rnn":
            # for rnn, better to use no warmup and larger dropout
            cur_opt += " lrate_warmup:0 drop_hidden:0.33"
    # gpu and seed
    cur_opt += f" device:{-1 if args.rgpu<0 else 0}" \
               f" niconf.random_seed:9347{args.cur_run} niconf.random_cuda_seed:9349{args.cur_run}"
    # finally extras
    cur_opt += " " + args.extras
    # =====
    # training
    CMD = f"CUDA_VISIBLE_DEVICES={args.rgpu} PYTHONPATH={src_dir} python3 " + (" -m pdb " if args.debug else "") \
          + f"{src_dir}/tasks/cmd.py zmlm.main.train {cur_opt} " + ("" if args.debug else f"2>&1 | tee _log")
    system(CMD)
    # testing
    if not args.debug and args.do_test and (not RUN_MODE.startswith("pre")):
        for wset in ["dev", "test"]:
            if RUN_MODE.startswith("ner"):
                CMD2 = f"CUDA_VISIBLE_DEVICES={args.rgpu} PYTHONPATH={src_dir} python3 {src_dir}/tasks/cmd.py zmlm.main.test" \
                       f" _conf input_format:ner do_ner:1 test:{ner_dir}/{CUR_LANG}/{wset}.txt output_file:zout.{wset}" \
                       f" model_load_name:zmodel.best"
            else:
                CMD2 = f"CUDA_VISIBLE_DEVICES={args.rgpu} PYTHONPATH={src_dir} python3 {src_dir}/tasks/cmd.py zmlm.main.test" \
                       f" _conf input_format:conllu test:{ud_dir}/{CUR_LANG}_{wset}.conllu output_file:zout.{wset}" \
                       f" model_load_name:zmodel.best"
            system(CMD2)
    # running with pre-train
    if not args.debug and args.do_test and (RUN_MODE == "pre"):
        CMD = f"python3 {SCRIPT_ABSPATH} -l {args.lang} --rgpu {args.rgpu} --run_dir _ppp1 --run_mode ppp1 --preload_prefix ../zmodel.c200"
        system(CMD)
    # =====

# =====
def main():
    printing(f"Run with {sys.argv}")
    args = parse()
    for k in sorted(dir(args)):
        if not k.startswith("_"):
            printing(f"--{k} = {getattr(args, k)}")
    # --
    DIRs = step0_dir(args)
    step_run(args, DIRs)

# --
if __name__ == '__main__':
    main()
