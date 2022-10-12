#

# some common confs for zmtl3

from msp2.utils import zwarn

# --
# generally three modes: 1) plain training, 2) extra training, 3) special test cases
def conf_getter_trig(data='ace', extra_data='', fs_seed=0, fs_k='5', fs_k2='', test_case='', training=True, **kwargs):
    # --
    fs_seed = int(fs_seed)
    if not fs_k2:
        fs_k2 = fs_k  # if no diff between train/test, then simply make them the same!
    if len(kwargs) > 0:
        zwarn(f"conf_getter_train gets extra kwargs: {kwargs}")
    # --
    args = []
    # --
    # task & model
    args.extend("evt0:yes evt0.b_model:roberta-base".split())
    if extra_data:
        args.extend("evt2:yes evt2.bconf:shared evt2.shared_bmod_name:evt0".split())  # simply share evt0's bmod!
        args.extend("model_save_dels:Mevt2.bmod".split())  # not saving that for convenience
    # --
    # data
    if data:
        # no file names given to train/dev!
        for wset in ["train0", "dev0", "dev1", "test0", "test1"]:
            args.extend(f"{wset}.group_name:ee {wset}.input_dir:events/data/data21f "
                        f"{wset}.group_files: {wset}.group_tasks:evt0".split())
        # only give test filenames
        args.extend(f"test0.group_files:en.{data}2.dev.json,en.{data}2.test.json".split())  # test on both
        args.extend(f"test1.group_files:en.{data}2.train.json".split())
        for wset in ["dev1", "test1"]:  # special (sparse+sampled(balanced)_neg) for dev1/test1!
            args.extend(f"{wset}.special_test:1 {wset}.group_name:eeS {wset}.inst_f:sent_evt2 '{wset}.fs_str:{data}.+*:{fs_k2};seed:{fs_seed}'".split())
        # which batcher to use for train?
        # args.extend("train0.batcher:frame train0.type_sample_alpha:0.5 train0.batcher.n_type:8 train0.batcher.k_shot:5".split())
        # args.extend("train0.batcher:plain train0.batch_size:1024 accu_batch:2".split())
        # test on train?
        # testM.input_dir:events/data/data21f testM.group_files:en.ace2.train.json testM.group_tasks:evt0 'testM.fs_str:ace.+*:5;seed:0' testM.inst_f:sent_evt
        # tune thrs
        # evt0.setup_do_prep:1 evt0.prep_ranges:0,0.01,0.02,0.03,0.04,0.05
    # --
    if extra_data:
        for wset in ["train2", "dev2"]:
            args.extend(f"{wset}.group_name:ee2 {wset}.input_dir:events/data/data21f "
                        f"{wset}.group_files: {wset}.group_tasks:evt2".split())
            args.extend(f"{wset}.batcher:frame {wset}.type_sample_alpha:0. {wset}.batcher.n_type:8 {wset}.batcher.k_shot:5".split())
        args.extend("train2.ignore_fcount:0".split())
        args.extend("dev2.test_with_loss:50".split())
        if extra_data in ['ace', 'ere']:  # remove original ones and put lemma
            args.extend(f"train2.fl_str:NOUN,VERB dev2.fl_str:NOUN,VERB".split())
    # --
    # setup train?
    # args.extend("evt0.setup_dname_train:ee_en.ace2.train.json".split())
    # --
    # optim?
    if training:
        UPE = 1000
        args += " optim:adam lrate.val:0.00002".split()
        args += f" lrate.mode:linear lrate.which_idx:uidx lrate.b:1 lrate.k:-1 lrate.m:1 lrate.idx_bias:{10*UPE} lrate.idx_scale:{50*UPE}".split()  # derterminstic annealing
        args += f" valid_ufreq:{UPE} max_uidx:{UPE*50}".split()
    # --
    # special testings
    if not training and test_case:
        if test_case == "case1":
            args += conf_trig_test_case1(data, extra_data, fs_seed, fs_k, fs_k2)
    return args

def conf_trig_test_case1(data, extra_data, fs_seed, fs_k, fs_k2):
    args = []
    # --
    # clear them all
    for wset0 in ['train', 'dev', 'test']:
        args.extend([f"{wset0}{z}.group_files:" for z in range(10)])
    # --
    for wset in ["train9", "test0", "test1"]:
        args.extend(f"{wset}.group_name:ee {wset}.batcher:plain {wset}.input_dir:events/data/data21f "
                    f"{wset}.group_files:en.{data}2.{wset[:-1]}.json {wset}.group_tasks:evt0".split())
    args.extend(f"test0.group_files:en.{data}2.dev.json,en.{data}2.test.json".split())  # test on both
    for wset in ["test1"]:  # special (sparse+sampled(balanced)_neg) for dev1/test1!
        args.extend(
            f"{wset}.special_test:1 {wset}.group_name:eeS {wset}.group_files:en.{data}2.train.json {wset}.inst_f:sent_evt2 '{wset}.fs_str:{data}.+*:{fs_k2};seed:{fs_seed}'".split())
    # --
    args.extend(f"train9.inst_f:sent_evt 'train9.fs_str:ace.+*:{fs_k};seed:{fs_seed}' evt0.setup_dname_test:ee_en.{data}2.train.json evt0.neg_number:5 'evt0.upos_filter:+NV,-*' evt0.func:cos evt0.sim.scale.init:1. evt0.bert_lidx:10".split())
    args.extend("vocab_force_rebuild:1 model_load_name:".split())
    # for example: ...
    # args.extend(["st_args:[f'stn:{m},{z/100} model_load_name:zmodel.{m}###DMevt0.lab evt0.neg_delta.init:{z/100}' for m in ['curr','best'] for z in range(6)]"])
    return args

# for example:
# python3 -mpdb -m msp2.tasks.zmtl3.main.train "conf_sbase:data:ace;task:arg" device:0 train0.group_files:en.ace2.train.json dev0.group_files:en.ace2.dev.json
def conf_getter_arg(data='ace', s_seed=0, s_k=1., fs_seed=0, fs_k='', train_batcher='frame', training=True, **kwargs):
    # --
    # note: by default, no down-sample
    s_seed, fs_seed = [int(z) for z in [s_seed, fs_seed]]
    s_k = float(s_k)
    if len(kwargs) > 0:
        zwarn(f"conf_getter_train gets extra kwargs: {kwargs}")
    # --
    args = []
    # --
    # task & model
    args.extend("arg0:yes arg0.b_model:roberta-base".split())
    if data:
        args.extend(f"arg0.build_oload:{data}".split())  # pre-defined onto
        # no file names given to train/dev!
        for wset in ["train0", "dev0", "test0"]:
            args.extend(f"{wset}.group_name:ee {wset}.input_dir:events/data/data21f "
                        f"{wset}.group_files: {wset}.group_tasks:arg0".split())
        # sample train/dev?
        for wset in ["train0", "dev0"]:
            args.extend(f"{wset}.presample:{s_k} {wset}.presample_seed:{s_seed}".split())
            if fs_k:
                args.extend(f"'{wset}.fs_str:{data}.+*:{fs_k};seed:{fs_seed}'".split())
        # batcher?
        for wset in ["train0"]:
            if train_batcher == 'plain':
                args.extend(f"{wset}.batcher:plain {wset}.inst_f:frame {wset}.batch_size:40 {wset}.batch_size_f:num {wset}.bucket_interval:10000".split())
            elif train_batcher == 'frame':
                args.extend(f"{wset}.batcher:frame {wset}.type_sample_alpha:0.75 {wset}.batcher.n_type:8 {wset}.batcher.k_shot:4".split())
        for wset in ["dev0", "test0"]:
            args.extend(f"{wset}.batcher:plain {wset}.inst_f:frame {wset}.batch_size:10 {wset}.batch_size_f:num {wset}.bucket_interval:10000".split())
        # only give test filenames
        args.extend(f"test0.group_files:en.{data}2.dev.json,en.{data}2.test.json".split())  # test on both
    # --
    # optim?
    if training:
        UPE = 1000
        args += " optim:adam lrate.val:0.00002".split()
        args += f" lrate.mode:linear lrate.which_idx:uidx lrate.b:1 lrate.k:-1 lrate.m:1 lrate.idx_bias:{10*UPE} lrate.idx_scale:{50*UPE}".split()  # derterminstic annealing
        args += f" valid_ufreq:{UPE} max_uidx:{UPE*50}".split()
    # --
    return args

# --
def conf_getter_mat(training=True, **kwargs):
    args = []
    # args.extend("matM:yes matM.b_model:roberta-base".split())
    # args.extend("matM.b_model:__mat/matbert-base-cased".split())
    # args.extend("matM.ctx_nsent_rates:1,1".split())
    # args.extend("matM.crf:yes".split())
    # args.extend("matR:yes matR.b_model:roberta-base".split())
    for wset in ["train0", "dev0", "test0"]:
        args.extend(f"{wset}.group_name:matD {wset}.input_dir: {wset}.group_files:".split())
        args.extend(f"{wset}.batcher:plain {wset}.batch_size:512 {wset}.bucket_interval:10000".split())
    # --
    # optim?
    if training:
        UPE = 1000
        E0, EMAX = 1, 20
        args += " optim:adam lrate.val:0.00002".split()
        args += f" lrate.mode:linear lrate.which_idx:uidx lrate.b:1 lrate.k:-1 lrate.m:1 lrate.idx_bias:{E0*UPE} lrate.idx_scale:{EMAX*UPE}".split()  # derterminstic annealing
        args += f" valid_ufreq:{UPE} max_uidx:{EMAX*UPE}".split()
    # --
    return args

# --
def conf_getter(task='trig', **kwargs):
    if task == 'trig':
        return conf_getter_trig(**kwargs)
    elif task == 'arg':
        return conf_getter_arg(**kwargs)
    elif task == 'mat':
        return conf_getter_mat(**kwargs)
    else:
        raise NotImplementedError(f"UNK task of {task}!!")

def conf_getter_train(**kwargs):
    return conf_getter(training=True, **kwargs)

def conf_getter_test(**kwargs):
    return conf_getter(training=False, **kwargs)
