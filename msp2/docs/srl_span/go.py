#

# one easy-to-run script to run them all

# --
# add path!!
import sys
sys.path.extend(["../"*i+"src" for i in range(5)])
# --

import os
from msp2.utils import Conf, ConfEntryChoices, zlog, init_everything, mkdir_p, zglob1, system

# --
class DataConf(Conf):
    def __init__(self):
        self._choice = ""
        # should be one set!
        self.data_dir = ""
        self.train_files = []
        self.dev_files = []
        self.test_file = ""
        # special ones
        self.frame_file = ""
        self.all_dt_files = []  # for files for testing
        self.repl_ud_infix = "ud"
        # ms_train
        self.ms_train_files = []
        self.ms_props = ""

    def get_data_str(self, data_dir: str, do_ms_train: bool):
        train_files, dev_files, test_file = \
            ','.join([f"{data_dir}/{z}" for z in self.train_files]), \
            ','.join([f"{data_dir}/{z}" for z in self.dev_files]), \
            f"{data_dir}/{self.test_file}"
        ret = " "
        if do_ms_train:
            train_files = ','.join([f"{data_dir}/{z}" for z in self.ms_train_files])
            ret += self.ms_props
        # dt_file = ','.join([f"{data_dir}/{z}" for z in self.all_dt_files])
        ret = f" train:{train_files} dev:{dev_files} test:{test_file}" + ret
        # --
        if self.repl_ud_infix != "ud":
            # replace infix of "ud" with this, for example, with "ud2"
            fields = [(z if z!="ud" else self.repl_ud_infix) for z in ret.split(".")]
            ret = ".".join(fields)
        # --
        return ret
# --

# --
PRESET_DATA = {
    "fn15": DataConf.direct_conf(
        data_dir="data/fn/parsed", train_files=["fn15_fulltext.train.json"], dev_files=["fn15_fulltext.dev.json"],
        test_file="fn15_fulltext.test.json", frame_file="fn15_frames.json",
        ms_train_files=["fn15_fulltext.train.json", "fn15_exemplars.filtered.json"],
        ms_props=" train_props:[{},{\\'sent_loss_weight_non\\':0.1}] ms_train_budget1.val:4",
        all_dt_files=["fn15_fulltext.dev.json", "fn15_fulltext.test.json"],
    ),
    "fn17": DataConf.direct_conf(
        data_dir="data/fn/parsed", train_files=["fn17_fulltext.train.json"], dev_files=["fn17_fulltext.dev.json"],
        test_file="fn17_fulltext.test.json", frame_file="fn17_frames.json",
        ms_train_files=["fn17_fulltext.train.json", "fn17_exemplars.filtered.json"],
        ms_props=" train_props:[{},{\\'sent_loss_weight_non\\':0.1}] ms_train_budget1.val:4",
        all_dt_files=["fn17_fulltext.dev.json", "fn17_fulltext.test.json"],
    ),
    "pb05": DataConf.direct_conf(
        data_dir="data/pb/conll05", train_files=["train.conll.ud.json"], dev_files=["dev.conll.ud.json"],
        test_file="test.wsj.conll.ud.json",
        all_dt_files=["dev.conll.ud.json", "test.wsj.conll.ud.json", "test.brown.conll.ud.json"],
    ),
    "pb12": DataConf.direct_conf(
        data_dir="data/pb/conll12b", train_files=["train.conll.ud.json"],
        # dev_files=["dev2.conll.ud.json"],  # use smaller dev2 to speed up
        dev_files=["dev.conll.ud.json"],
        test_file="test.conll.ud.json",
        ms_train_files=["train.conll.ud.json", "train2.conll.ud.json"],  # extra one
        ms_props=" train_props:[{},{\\'sent_loss_weight_non\\':0.25}] ms_train_budget1.val:0.5",
        # all_dt_files=["dev2.conll.ud.json", "dev.conll.ud.json", "test.conll.ud.json"],
        all_dt_files=["dev.conll.ud.json", "test.conll.ud.json"],
    ),
    "pbewt": DataConf.direct_conf(
        data_dir="data/pb/pb", train_files=["ewt.train.conll.ud.json"], dev_files=["ewt.dev.conll.ud.json"],
        test_file="ewt.test.conll.ud.json",
        all_dt_files=["ewt.dev.conll.ud.json", "ewt.test.conll.ud.json"],
    ),
    # ==
    "pb12c": DataConf.direct_conf(
        data_dir="data/pb/conll12c", train_files=["en.nw.train.conll.ud.json"], dev_files=["en.nw.dev.conll.ud.json"],
        test_file="en.nw.test.conll.ud.json",
        # all_dt_files=["en.nw.dev.conll.ud.json", "en.nw.test.conll.ud.json"] + [f"en.{z}.train.conll.ud.json" for z in ["bc", "bn", "mz", "pt", "tc", "wb"]] + [f"zh.{z}.train.conll.json" for z in ["bc", "bn", "mz", "nw", "tc", "wb"]],
        all_dt_files=["en.nw.dev.conll.ud.json", "en.nw.test.conll.ud.json"] + [f"en.{z}.test.conll.ud.json" for z in ["bc", "bn", "mz", "pt", "tc", "wb"]],
    ),
    # cross-lingual ones
    "pb12cl": DataConf.direct_conf(
        data_dir="data/pb/conll12c", train_files=["en.nw.train.conll.ud.json"], dev_files=["en.nw.dev.conll.ud.json"],
        test_file="en.nw.test.conll.ud.json",
        all_dt_files=["en.nw.dev.conll.ud.json", "en.nw.test.conll.ud.json"] + [f"en.{z}.test.conll.ud.json" for z in ["bc", "bn", "mz", "pt", "tc", "wb"]] + ["ar.nw.dev.conll.json", "ar.nw.train.conll.json"] + [f"zh.{z}.dev.conll.json" for z in ["bc", "bn", "mz", "nw", "tc", "wb"]] + [f"zh.{z}.train.conll.json" for z in ["bc", "bn", "mz", "nw", "tc", "wb"]],
    ),
    # # v2: on conll12d, the newly prepared one!
    # "pb12cl2": DataConf.direct_conf(
    #     data_dir="data/pb/conll12d", train_files=[], dev_files=[], test_file="",
    #     # all_dt_files=["ar.nw.dev.conll.json", "ar.nw.train.conll.json"],
    #     all_dt_files=["ar.nw.dev.conll.json", "ar.nw.train.conll.json"] + [f"zh.{z}.{wset}.conll.json" for z in ["bc", "bn", "mz", "nw", "tc", "wb"] for wset in ["train", "dev", "test"]],
    # ),
    # v3: on conll12d, with all english data!
    "pb12cl3": DataConf.direct_conf(
        data_dir="data/pb/conll12d", train_files=["en.train.conll.ud.json"], dev_files=["en.dev2.conll.ud.json"],
        test_file="en.test.conll.ud.json",
        all_dt_files=["ar.nw.train.conll.json", "ar.nw.dev.conll.json", "ar.nw.test.conll.json"] + [f"zh.{z}.{wset}.conll.json" for z in ["bc", "bn", "mz", "nw", "tc", "wb"] for wset in ["train", "dev", "test"]] + ["zh.train.conll.json", "zh.dev.conll.json", "zh.test.conll.json"],
    ),
}


class RunConf(Conf):
    def __init__(self):
        # basic
        self.conf_output = "_conf"
        self.rgpu = -1
        self.cur_run = "1"
        self.debug = False
        self.do_train = True  # training
        self.do_test = False  # testing
        self.do_test_all = False  # testing all_df_files
        self.train_extras = ""
        self.test_extras = ""
        # paths
        self.run_dir = ""  # by default current one!
        self.src_dir = "src"
        self.voc_dir = "voc"
        self.dataset: DataConf = ConfEntryChoices(PRESET_DATA, "pb05")
        self.log_prefix = "_log"
        self.out_prefix = "_zout"
        # --
        # specific
        self.use_word_input = True  # input with word embeddings?
        self.use_bert_input = False  # input featured bert?
        self.use_rel_posi = True  # relative position
        self.assume_frame = False  # assume both trg and frame type
        self.assume_trg = False  # assume trg
        self.no_frame_label = False  # no frame label
        self.arg_mode = "seq"  # seq/span/head
        self.arg_seq_mod = "crf"  # mle/crf
        self.do_ms_train = False  # multi-source training?

def main(*args):
    conf: RunConf = init_everything(RunConf(), args, add_utils=False, add_nn=False)
    # =====
    # get paths
    RUN_DIR = conf.run_dir
    if RUN_DIR:
        mkdir_p(RUN_DIR, raise_error=True)
        os.chdir(RUN_DIR)  # change to it!!
    SRC_DIR = zglob1(conf.src_dir, check_prefix="..", check_iter=10)
    VOC_DIR = zglob1(conf.voc_dir, check_prefix="..", check_iter=10)
    DATA_DIR = zglob1(conf.dataset.data_dir, check_prefix="..", check_iter=10)
    zlog(f"RUN with RUN={RUN_DIR}, SRC={SRC_DIR}, VOC={VOC_DIR}, DATA={DATA_DIR}")
    # =====
    # modes
    dataset_choice = conf.dataset._choice
    is_pb, is_fn = [dataset_choice.startswith(z) for z in ["pb", "fn"]]
    assert is_pb or is_fn
    # =====
    # options
    # --
    # base ones
    base_opt = "conf_output:_conf"
    # eval
    if is_pb:
        base_opt += f" eval_conf:pb"
    elif is_fn:
        base_opt += f" dict_frame_file:{DATA_DIR}/{conf.dataset.frame_file}"
        base_opt += f" eval_conf:fn eval_conf.frame_file:{DATA_DIR}/{conf.dataset.frame_file}"  # eval
    # --
    # =====
    # modeling
    if conf.use_word_input:
        base_opt += " ec_word.dim:300 ec_word.drop_rate:0.2 ec_word.init_from_pretrain:1 ec_word.rare_unk_thr:2"  # word
    # base_opt += " ec_posi.dim:512"  # posi?
    # base_opt += " ec_char.dim:50 ec_char.init_scale:5."  # char?
    if conf.use_bert_input:
        base_opt += " ec_bert.dim:768 bert_model:bert-base-cased bert_output_layers:7,8,9"  # bert features?
    base_opt += " eproj_dim:512"  # --
    if conf.use_rel_posi:
        base_opt += " enc_conf.enc_att.n_layers:2 enc_conf.enc_att.use_posi:0 enc_conf.clip_dist:16"  # enc1
    else:
        base_opt += " enc_conf.enc_att.n_layers:2 enc_conf.enc_att.use_posi:1 enc_conf.clip_dist:0"  # enc1
    # base_opt += " enc_conf.enc_tatt.n_layers:2 enc_conf.enc_tatt.use_posi:1"  # enc1
    # base_opt += " enc_conf.enc_rnn.n_layers:1 enc_conf.enc_hidden:1024"  # enc1
    # --
    # frame
    base_opt += " loss_evt:0.5 pred_evt:1"  # with evts
    base_opt += " evt_conf.cand_label_smoothing:0.05 evt_conf.label_smoothing:0.1"  # label smooth
    base_opt += " evt_conf.lookup_conf.use_emb:0"  # no adding frame embeddings?
    base_opt += " evt_conf.span_conf.sconf.hid_nlayer:1"  # pred scorer?
    if conf.assume_frame:  # no need for the evt module!!
        base_opt += " loss_evt:0 pred_evt:0 eval_conf.weight_frame:0."
    elif conf.assume_trg:  # no need for cand, but still need to identify frame types
        base_opt += " evt_conf.loss_cand:0. evt_conf.loss_use_posi:1 evt_conf.pred_use_posi:1"  # use-posi for evt
        base_opt += " evt_conf.pred_addition_non_score:-100000."  # NEGINF-non
        if is_fn:  # further use cons for fn
            base_opt += f" evt_cons_lex_file:{VOC_DIR}/cons_lex_{dataset_choice}.json evt_conf.pred_use_cons:1 evt_conf.pred_use_lu:1 evt_conf.loss_use_cons:0 evt_conf.loss_use_lu:0"  # cons & use-lu for evt
    else:
        # evt_conf -> direct
        base_opt += " evt_conf.loss_cand:1.0 evt_conf.loss_lab:1.0"  # loss_cand
        base_opt += " evt_conf.span_train_sample_rate:1.0 evt_conf.span_topk_rate:1.0 evt_conf.span_train_sample:1"  # some rates
        # --
        if is_pb:  # lab is aux for pb
            base_opt += " evt_conf.loss_lab:0.5 evt_conf.pred_score_prune:0. evt_conf.pred_addition_non_score:-100000."
        elif is_fn:  # lab is essential for fn
            base_opt += " loss_evt:1 evt_conf.loss_cand:0.5 evt_conf.span_train_sample_rate:0.33 evt_conf.span_topk_rate:0.4 evt_conf.span_train_sample:1"
        # --
        if conf.no_frame_label:
            base_opt += " evt_conf.loss_lab:0. evt_conf.pred_score_prune:0. evt_conf.pred_addition_non_score:-100000."
    # --
    # arg
    base_opt += " arg_use_finput:0"
    base_opt += f" fenc_conf.enc_att.n_layers:8 fenc_conf.clip_dist:{16 if conf.use_rel_posi else 0}"  # fenc
    # base_opt += " fenc_conf.enc_tatt.n_layers:6"  # fenc
    # base_opt += " fenc_conf.enc_rnn.n_layers:3 fenc_conf.enc_hidden:1024"  # enc1
    base_opt += " loss_arg:1. pred_arg:1"  # with args
    base_opt += " arg_conf.label_smoothing:0.1"  # label smooth
    if conf.arg_mode in ["span", "head"]:
        # arg_conf -> direct
        base_opt += " arg_conf.loss_cand:0.5"  # loss_cand
        # base_opt+=" arg_conf.span_train_sample_rate:0.33 arg_conf.span_topk_rate:0.4"  # some rates
        base_opt += " arg_conf.span_topk_rate:1. arg_conf.span_topk_count:10 arg_conf.span_train_sample:0"  # some rates
        base_opt += " arg_conf.loss_weight_non:1."  # less penalizing this?
        base_opt += " arg_conf.pp_check_more:1"  # check non-overlapping
        if conf.arg_mode == "span":
            base_opt += " arg_conf.max_width:30 arg_conf.softhead_topk:5 arg_conf.pred_non_overlapping:1"  # span
        elif conf.arg_mode == "head":
            base_opt += " arg_conf.core_span_mode:shead arg_conf.max_width:1"  # head
            # extender
            base_opt += " arg_conf.loss_ext:0.5 arg_conf.pred_ext:1 arg_conf.ext_use_finput:0"
            base_opt += f" arg_conf.ext_conf.eenc_conf.enc_att.n_layers:1 arg_conf.ext_conf.eenc_conf.enc_att.aconf.clip_dist:{16 if conf.use_rel_posi else 0}"
        else:
            raise NotImplementedError()
    elif conf.arg_mode == "soft":
        base_opt += " arg_conf:soft"
        base_opt += " arg_conf.loss_ext:0.5 arg_conf.pred_ext:1 arg_conf.ext_use_finput:0"
        base_opt += f" arg_conf.ext_conf.eenc_conf.enc_att.n_layers:1 arg_conf.ext_conf.eenc_conf.enc_att.aconf.clip_dist:{16 if conf.use_rel_posi else 0}"
        base_opt += " arg_conf.pp_check_more:1"
    elif conf.arg_mode in ["anchor", "anchor2"]:
        base_opt += " arg_conf:anchor"
        if conf.arg_mode == "anchor2":  # yet another head mode!
            base_opt += " arg_conf.core_span_mode:shead"
        base_opt += " arg_conf.loss_ext:0.5 arg_conf.pred_ext:1 arg_conf.ext_use_finput:0"
        base_opt += f" arg_conf.ext_conf.eenc_conf.enc_att.n_layers:1 arg_conf.ext_conf.eenc_conf.enc_att.aconf.clip_dist:{16 if conf.use_rel_posi else 0}"
        base_opt += " arg_conf.pp_check_more:1"
    elif conf.arg_mode in ["seq", "seq0"]:
        # arg_conf -> seq
        base_opt += " arg_conf:seq arg_conf.seq_scheme:BIO"  # use seq mode!
        base_opt += " arg_conf.loss_weight_non:1."  # less penalizing this?
        # --
        if conf.arg_mode == "seq":
            base_opt += " arg_conf.beam_k:150 arg_conf.use_bigram:0 arg_conf.pred_use_seq_cons:1"  # viterbi with constraints
            if conf.arg_seq_mod == "crf":  # crf-mode
                base_opt += " arg_conf.loss_mode:crf arg_conf.use_bigram:1 arg_conf.local_normalize:0"
        elif conf.arg_mode == "seq0":  # greedy mode: no crf and no viterbi
            base_opt += " arg_conf.pred_use_seq_cons:0 arg_conf.loss_mode:mle arg_conf.use_bigram:0 arg_conf.local_normalize:1"
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    # --
    # =====
    # training
    base_opt += " ema_decay:0. ema_update_step:1"  # ema
    if 1:
        UPE = 1000  # 1000 update as one epoch
        base_opt += " lrate.val:0.0002 anneal_times:10 anneal_patience:10 lrate.m:0.75"
        base_opt += f" valid_ufreq:{UPE} valid_epoch:0 max_uidx:{UPE*150} lrate_warmup_uidx:{8*UPE} lrate_decrease_alpha:0."
        if conf.use_rel_posi:
            base_opt += " train_count_mode:ftok train_batch_size:4096 accu_batch:1"  # actually bs=bs*accu
            base_opt += " test_count_mode:ftok test_batch_size:2048"
        else:
            base_opt += " train_count_mode:ftok train_batch_size:4096 accu_batch:1"  # actually bs=bs*accu
            base_opt += " test_count_mode:ftok test_batch_size:2048"
        base_opt += " df_hdrop:0.2"  # general dropout
    else:  # possibly for rnn
        base_opt += " lrate.val:0.002 anneal_times:10 anneal_patience:10"
        base_opt += " train_count_mode:frame max_eidx:100 train_batch_size:32"
        base_opt += " df_hdrop:0.33"  # general dropout
    if is_pb:
        base_opt += " train_skip_noevt_rate:0.0"
    elif is_fn:
        base_opt += " train_skip_noevt_rate:1.0"  # skip sents where no targets!
    # data
    base_opt += " " + conf.dataset.get_data_str(DATA_DIR, conf.do_ms_train)
    base_opt += f" pretrain_wv_file:{VOC_DIR}/hits_{dataset_choice}.vec pretrain_scale:10."  # filtered pretrain file
    # nn
    base_opt += f" nn.device:0 nn.random_seed:9347{conf.cur_run} nn.random_cuda_seed:9349{conf.cur_run}"
    # =====
    # note: base_opt is only for training!!
    _L_PRE = conf.log_prefix
    DEBUG_OPTION = "-m pdb" if conf.debug else ""
    TRAIN_CMD = f"CUDA_VISIBLE_DEVICES={conf.rgpu} PYTHONPATH={SRC_DIR}:$PYTHONPATH python3 {DEBUG_OPTION} -m msp2.tasks.zsfp.main.train {base_opt} log_file:{_L_PRE}_train {conf.train_extras}"
    # --
    TEST_CMD = f"CUDA_VISIBLE_DEVICES={conf.rgpu} PYTHONPATH={SRC_DIR}:$PYTHONPATH python3 {DEBUG_OPTION} -m msp2.tasks.zsfp.main.test {conf.conf_output} log_file:{_L_PRE}_test {conf.test_extras}"
    # --
    if conf.do_train:
        system(TRAIN_CMD, pp=True)
    # --
    if conf.do_test:
        system(TEST_CMD, pp=True)
    # --
    if conf.do_test_all:
        for tfile in conf.dataset.all_dt_files:
            _TMP_CMD = f"CUDA_VISIBLE_DEVICES={conf.rgpu} PYTHONPATH={SRC_DIR}:$PYTHONPATH python3 {DEBUG_OPTION} -m msp2.tasks.zsfp.main.test {conf.conf_output} test:{DATA_DIR}/{tfile} output:{conf.out_prefix}.{tfile} log_file:{_L_PRE}.{tfile} test_extra_pretrain_wv_files:{VOC_DIR}/hits_{dataset_choice}.vec {conf.test_extras}"
            system(_TMP_CMD, pp=True)
    # --
    # =====

# --
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
