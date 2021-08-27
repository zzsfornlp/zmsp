#

# some example running confs

dict(
# cl0
cl0_base_xlmr=[
    ["bert_model:xlm-roberta-base"],
    ["reg0.mod_trg:Menc.bert reg0.reg_method:update reg0.l2_reg:1"],
],
cl0_syn_xlmr=[
    ["tconf.udep:yes idec_udep_lab.app_layers:12 udep.loss_udep_lab:0.5"],
    ["train1.group_tasks:udep train1.input_dir:ud/cl0 train1.group_joint:1 train1.batch_size:1024"],
    ["train1.group_name:ud train1.group_files:en.ud.train.json,de.ud.train.json,fr.ud.train.json,it.ud.train.json,es2.ud.train.json,pt_bosque.ud.dev.json,fi.ud.train.json"],
    [f"lrate.val:0.00002 max_uidx:150000 lrate.idx_bias:15000 lrate.idx_scale:150000"],
    ["train1.group_sample_rate.val:0.5 record_best_start_cidx:120"],
    # --
    ["bert_model:xlm-roberta-base"],
],
# cl1
cl1_both=[
    ["bert_model:xlm-roberta-base"],
    ["train1.group_tasks:pb1 dev1.group_tasks:pb1 test1.group_tasks:pb1 dev0.group_files:"],
    # --
    ["tconf.udep:yes idec_udep_lab.app_layers:12 udep.loss_udep_lab:0.5"],
    ["train2.group_files:_ud14/en0,_ud14/fi0"],
    ["train2.input_format:conllu train2.batch_size:1024 train2.group_sample_rate.val:0.5"],
    # --
    ["train2.group_files:"],  # en-srl + fi-srl
    [f"train1.presample:{z} dev1.presample:{(z // 10) if z > 10 else z}" for z in [1000]],
],
cl1_syn=[
    ["bert_model:xlm-roberta-base"],
    ["train1.group_tasks:pb1 dev1.group_tasks:pb1 test1.group_tasks:pb1 dev0.group_files:"],
    # --
    ["tconf.udep:yes idec_udep_lab.app_layers:12 udep.loss_udep_lab:0.5"],
    ["train2.group_files:_ud14/en0,_ud14/fi0"],
    ["train2.input_format:conllu train2.batch_size:1024 train2.group_sample_rate.val:0.5"],
    # --
    ["train2.group_files:_ud14/en0,_ud14/fi0"],  # en-srl + fi-srl + en-ewt + fi-tdt
    [f"train1.presample:{z} dev1.presample:{(z // 10) if z > 10 else z}" for z in [1000]],
],
# cl2 (for example, lang=zh)
cl2_zh_both=[
    ["bert_model:xlm-roberta-base"],
    ["dev0.group_files: dev1.presample:1000"],
    ["train1.group_tasks:pb1 dev1.group_tasks:pb1 test1.group_tasks:pb1 pb1.non_overlapping:1"],
    # --
    ["tconf.udep:yes idec_udep_lab.app_layers:12 udep.loss_udep_lab:0.5"],
    ["train2.input_dir:ud/cl2 train2.group_files:en.train.ud.json,zh.train.ud.json"],
    ["train2.input_format:zjson train2.batch_size:1024 train2.group_sample_rate.val:0.5"],
    ["train2.presample:10000 train2.presample_reverse:0"],  # take some!
    # --
    ["train2.group_files:"],  # w/o syn
    [f"train1.presample:{z} dev1.presample:{(z // 10) if z > 10 else z}" for z in [1000]],
],
cl2_zh_syn=[
    ["bert_model:xlm-roberta-base"],
    ["dev0.group_files: dev1.presample:1000"],
    ["train1.group_tasks:pb1 dev1.group_tasks:pb1 test1.group_tasks:pb1 pb1.non_overlapping:1"],
    # --
    ["tconf.udep:yes idec_udep_lab.app_layers:12 udep.loss_udep_lab:0.5"],
    ["train2.input_dir:ud/cl2 train2.group_files:en.train.ud.json,zh.train.ud.json"],
    ["train2.input_format:zjson train2.batch_size:1024 train2.group_sample_rate.val:0.5"],
    ["train2.presample:10000 train2.presample_reverse:0"],  # take some!
    # --
    [f"train1.presample:{z} dev1.presample:{(z // 10) if z > 10 else z}" for z in [1000]],
],
# cl3 (for example, lang=es)
cl3_es_both=[
    ["dev0.group_files: dev1.presample:1000"],
    # --
    ["tconf.udep:yes udep.idec_udep_lab.app_layers:12 udep.loss_udep_lab:0.5"],
    ["train2.input_dir:ud/cl3 train2.group_files:UNK"],
    ["train2.input_format:zjson train2.batch_size:1024 train2.group_sample_rate.val:0.5"],
    ["train2.presample:10000 train2.presample_reverse:0"],  # take some!
    # --
    [f"train1.presample:{z} dev1.presample:{(z//10) if z>10 else z}" for z in [1000]],
    ["train2.group_files:"],  # w/o syn
],
cl3_es_syn=[
    ["dev0.group_files: dev1.presample:1000"],
    # --
    ["tconf.udep:yes udep.idec_udep_lab.app_layers:12 udep.loss_udep_lab:0.5"],
    ["train2.input_dir:ud/cl3 train2.group_files:UNK"],
    ["train2.input_format:zjson train2.batch_size:1024 train2.group_sample_rate.val:0.5"],
    ["train2.presample:10000 train2.presample_reverse:0"],  # take some!
    # --
    [f"train1.presample:{z} dev1.presample:{(z//10) if z>10 else z}" for z in [1000]],
    ["train2.group_files:en.train.ud.json,es.train.ud.json"],  # w/ syn
],
)
