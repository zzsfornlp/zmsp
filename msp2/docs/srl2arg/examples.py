#

# some example running confs

dict(
train_super=[
    ["conf_sbase:data:ace;task:arg"],
    [' '.join([f"{z}0.input_dir:data/data_evt" for z in ['train', 'dev', 'test']])],
    ["train0.group_files:en.ace.train.json dev0.group_files:en.ace.dev.json test0.group_files:en.ace.dev.json,en.ace.test.json"],
    ["arg0.mix_evt_ind:0.5"],
    ["train0.batcher.n_type:4 train0.batcher.k_shot:4 arg0.train_neg_ratios:0.5,0.5,0.5"],  # bs=16
    ["train0.ignore_fcount:0"],  # no ignore!
    ["arg0.arg_mode:tpl"],
],
train_srl=[
    ["conf_sbase:data:pbfn;task:arg"],
    ["arg0.mix_evt_ind:0.5"],
    ["train0.batcher.n_type:4 train0.batcher.k_shot:4 arg0.train_neg_ratios:0.5,0.5,0.5"],
    ["train0.ignore_fcount:2"],
    ["train0.type_sample_alpha:0.5"],
    ["arg0.extend_span:0 arg0.span_mode_arg:shead"],
    # --
    ["'arg0.filter_noncore:+spec,-*'"],
    ["dev0.group_files:data_pb/ewt.dev.conll.ud.json test0.group_files:data_pb/ewt.dev.conll.ud.json,data_pb/ewt.test.conll.ud.json"],
    ["train0.preprocessors:fn_norm_arg"],  # pre-process
    ["arg0.add_arg_conj:1"],  # add_conj
    [f"arg0.build_oload:pbfn train0.filter_onto:pbfn arg0.np_getter:fn"],
    ["arg0.arg_mode:tpl"],
    # --
    [' '.join([f"{z}0.input_dir:data" for z in ['train', 'dev', 'test']])],
    ["train0.group_files:data_pb/ontonotes.train.conll.ud.json,data_pb/ewt.train.conll.ud.json,data_nb/nb_f0.train.ud.json,data_fn/parsed/fn17_exemplars.filtered.json"],
],
train_qa=[
    ["conf_sbase:data:qa;task:arg"],
    ["train0.batcher.n_type:1 train0.batcher.k_shot:32 arg0.train_neg_ratios:0.5,0.5,0.5"],  # only one type!
    ["arg0.extend_span:0 arg0.span_mode_arg:shead"],
    # --
    [' '.join([f"{z}0.input_dir:data/data_qa/" for z in ['train', 'dev', 'test']])],
    ["dev0.group_files:en.squadR.dev.ud2.json dev0.presample:0.1 test0.group_files:en.squadR.dev.ud2.json"],
    ["arg0.add_arg_conj:1"],  # still add_conj
    ["arg0.build_oload:qa arg0.arg_mode:mrc arg0.mrc_use_eques:1"],  # qa-mode
    # --
    ["train0.group_files:en.squadR.train.ud2.json,en.qamrR.all.ud2.json,en.qasrl.all.ud2.json,en.qanom.all.ud2.json"],
    ["arg0.mix_evt_ind:0.5"],
    [f"train0.group_sample_betas:2,1,1,1"],
],
train_aug_srl=[
    ["conf_sbase:data:pbfn;task:arg"],
    ["arg0.mix_evt_ind:0.5"],
    ["train0.batcher.n_type:4 train0.batcher.k_shot:4 arg0.train_neg_ratios:0.5,0.5,0.5"],
    ["train0.ignore_fcount:2"],
    ["train0.type_sample_alpha:0.5"],
    ["arg0.extend_span:0 arg0.span_mode_arg:shead"],
    # --
    ["'arg0.filter_noncore:+spec,-*'"],
    ["dev0.group_files:data_pb/ewt.dev.conll.ud.json test0.group_files:data_pb/ewt.dev.conll.ud.json,data_pb/ewt.test.conll.ud.json"],
    ["train0.preprocessors:fn_norm_arg"],  # pre-process
    ["arg0.add_arg_conj:1"],  # add_conj
    [f"arg0.build_oload:pbfn train0.filter_onto:pbfn arg0.np_getter:fn"],
    ["arg0.arg_mode:tpl"],
    # --
    [' '.join([f"{z}0.input_dir:data" for z in ['train', 'dev', 'test']])],
    [f"train0.input_dir:data/qadistill train0.group_files:ontonotes.train.conll.ud.q1.json,ewt.train.conll.ud.q1.json,fn17_exemplars.filtered.q1.json,nb_f0.train.ud.q1.json"  # distilled data
     + " arg0.distill_alphas:1,0 arg0.distill_tau:2.511 arg0.distill_topk_thr:0.5 arg0.distill_topk_k:2"  # dist. options
     + " arg0.aug_rate:0.5 arg0.aug_combs:1,4"]  # aug. options
],
)
