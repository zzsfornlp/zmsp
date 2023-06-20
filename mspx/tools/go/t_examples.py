# --

# ==
# for clf

#

# an example of the "table_tasks"

_base_task = {
    'req_rs': {'cpu': 2, 'gpu': 1},
    'verbose': '[[V]]',
    'no_popen': '[[NP]]',
    'cmd': """
mkdir -p run_[[ID]]; 
cd run_[[ID]];
python3 [[PYOPTS]] -m mspx.tasks.zclf.main conf_sbase:bert_name:roberta-base conf_output:_conf log_stderr:0 log_file:_log device:0 conf_sbase:bert_name:roberta-base d_input_dir:__data/glue train0.group_files: dev0.group_files:sst2.dev.json test0.group_files:sst2.dev.json,sst2.test.json,orig_sst2.dev.json,orig_sst2.test.json fs:build,train,test [[ARGS]];
""",
    'cmd_res': "[py]self._read_res('run_[[ID]]/_log')",
}

table_tasks = dict(
_debug220705=(
    _base_task,
    {'ARGS': [
        [f"train0.group_files:{z}sst2.train.json" for z in ["", "orig_"]],
        [f"conf_sbase:bert_name:roberta-base::bmod2:{bmod2}" for bmod2 in [0,1]],
        [f"lrate.val:0.0000{lrate}" for lrate in [1,2,3]],
        # python3 -m mspx.scripts.tools.print_res input_path:./run*/_log t_dims:2,6 "t_ff:[x[f'test0_{z}']['zres'] for z in [2,3]]" t_lp:zzztestfinal
        # 0  0.9427/0.9516  0.9427/0.9495  0.9381/0.9478  0.9461/0.9484  0.9415/0.9516  0.9381/0.9478
        # 1  0.9381/0.9440  0.9312/0.9429  0.9312/0.9302  0.9346/0.9440  0.9335/0.9445  0.9312/0.9401
        # -> generally slightly worse without aug
    ]}
),
)

# ==

# ==
# for mlm

#

# an example for mlm

_base_task = {
    'req_rs': {'cpu': 2, 'gpu': 2},
    'verbose': '[[V]]',
    'no_popen': '[[NP]]',
    'cmd': """
mkdir -p run_[[ID]]; 
cd run_[[ID]];
# pretrain
python3 -m mspx.cli.run_ddp mspx.tasks.zmlm.main conf_sbase:bert_name:bert-base-cased::do_sub:0 conf_output:_conf log_stderr:0 log_file:_log nn.device:0 train0.group_files:__train0.tok.txt.bz2 dev0.group_files:__dev.tok.txt train0.input_format:plain_sent dev0.input_format:plain_sent fs:build,train use_torch_amp:1 dist_port:1236[[TIDX]] [[ARGS]];
# finetune
python3 -m mspx.tasks.zclf.main conf_sbase:bert_name:bert-base-cased::bmod2:1 clf0.init_with_bmodel: conf_output:_confF log_stderr:0 log_file:_logF device:0 d_input_dir:__data/glue train0.group_files:sst2.train.json dev0.group_files:sst2.dev.json test0.group_files:sst2.dev.json,sst2.test.json,orig_sst2.dev.json,orig_sst2.test.json fs:build,train,test train_preload_model:zmodel.curr.m,,,SMmlm0==Mclf0 model_save_prefix:zmodelF model_load_name:zmodelF.best.m;
""",
    'cmd_res': "[py]self._read_res('run_[[ID]]/_logF')",
}

table_tasks = dict(
try220708=(
    _base_task,
    {'ARGS': [
        ["save_special_start_cidx:0 save_special_cfreq:50"],
        [f"lrate.val:0.000{z}" for z in [1,4]],
        [f"train0.batch_size:{bs} dev0.batch_size:{bs}" for bs in [25000, 50000]],  # x2 for ddp=2
        # 4e-4 NAN, 1e-4 seems not good enough: 87/85
    ]}
),
)

# ==
