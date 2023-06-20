#

# some running examples

CUDA_VISIBLE_DEVICES=3 python3 -mpdb -m mspx.znew.prompt.proc.train global_cache_dir:___cache task_conf:clf0 train0.path::glue:sst2:train dev0.path::glue:sst2:validation eval.metrics:z_f1 test0.path::glue:sst2:validation template_choice:sst2 info_choice:sst2

# other data
# train0.path:__data/sst2/train.jsonl dev0.path: test0.path:__data/sst2/dev_subsample.jsonl

# clf_modes:head,head template_choice:sst2M model_name:roberta-base
# train0.shuffle_times:1 train0.sample_range:,128
# max_uidx:0
# pred_knn_lambda:1. sim_f:kl use_what:logits
# pred_knn_lambda:1. sim_f:ndist use_what:hiddens
# demo_select.sel_k:8 "pass_cidx_f:cidx==-1"

# knn-prompt
#clf_modes:head,head template_choice:sst2 train0.shuffle_times:1 train0.sample_range:,128 max_uidx:0 pred_knn_lambda:1. sim_f:kl demo_select.sel_k:8 "pass_cidx_f:cidx==-1"
# model_name:gpt2-xl

# icl
CUDA_VISIBLE_DEVICES=2 python3 -mpdb -m mspx.znew.prompt.proc.train global_cache_dir:___cache task_conf:clf0 train0.path::glue:sst2:train dev0.path::glue:sst2:validation eval.metrics:z_f1 test0.path::glue:sst2:validation load_half:1 mixed_precision:fp16
# MYTASK=sst2
# common0: train0.path:__data/${MYTASK}/train.jsonl dev0.path: test0.path:__data/${MYTASK}/dev_subsample.jsonl info_choice:${MYTASK} template_choice:${MYTASK}
# common: max_uidx:0 train0.shuffle_times:1 train0.sample_count:256 demo_select.sel_k:4 "pass_cidx_f:cidx==-1" sel_balance_label:1 model_name:gpt2-xl
# icl: clf_modes:head,head head_init_stra:tok
# knn: clf_modes:head,head pred_knn_lambda:1. sim_f:kl

# plain testing
CUDA_VISIBLE_DEVICES=0 python3 -mpdb -m mspx.znew.prompt.proc.test global_cache_dir:___cache load_half:1 task_conf:clf0 template_choice:sst2 info_choice:sst2 test0.path::glue:sst2:validation eval.metrics:z_f1

# model_load_name:zmodel.curr
# model_name:google/t5-v1_1-base
# model_name:decapoda-research/llama-7b-hf
# model_name:EleutherAI/pythia-70m
# load_in_8bit:1 load_half:1 peft_type:lora mixed_precision:fp16

# --
# for instruction tuning
CUDA_VISIBLE_DEVICES=3 python3 -mpdb -m mspx.znew.prompt.proc.train log_file:_log conf_output:_conf global_cache_dir:___cache model_save_suffix_bestn:.bestn task_conf:gen0 template_choice:alpaca info_choice:alpaca train0.path:__data/alpaca_data.json dev0.path: test0.path: model_name:gpt2-xl max_uidx:5000 load_in_8bit:1 load_half:1 peft_type:lora mixed_precision:fp16 init_lrate:1e-4 train0.batch_size:4 accu_batch:32 max_seq_len:512
# 'train0.bucket_f:(-1 if len(inst["_cache"]["_C"]["input"][0])>=510 else (len(inst["_cache"]["_C"]["input"][0])+2)//128)'
# model_name:gpt2-xl load_in_8bit:1 peft_type:lora mixed_precision:fp16
# model_load_name:zmodel.bestn

# demo
CUDA_VISIBLE_DEVICES=2 python3 -mpdb -m mspx.znew.prompt.scripts.gen0 global_cache_dir:___cache task_conf:gen0 template_choice:alpaca info_choice:alpaca
# template_choice:flan info_choice:flan model_name:google/flan-t5-base load_half:1
# model_name:decapoda-research/llama-7b-hf load_in_8bit:1 load_half:1
# --
# load_in_8bit:1 load_half:1 peft_type:lora mixed_precision:fp16
# --
# encode instructions
MDIR=
CUDA_VISIBLE_DEVICES=2 python3 -mpdb -m mspx.znew.prompt.scripts.gen0 $MDIR/_conf model_load_name:$MDIR/zmodel.bestn
CUDA_VISIBLE_DEVICES=3 python3 -m mspx.znew.prompt.scripts.gen0 $MDIR/_conf model_load_name:$MDIR/zmodel.bestn quiet:1 output_file:dA.pkl do_batch_enc:1 binput.path:../data/alpaca_data.json

# instruction with clf
# MYTASK=sst2
MDIR=
CUDA_VISIBLE_DEVICES=2 python3 -mpdb -m mspx.znew.prompt.proc.test $MDIR/_conf model_load_name:$MDIR/zmodel.bestn load_half:1 mixed_precision:fp16 test0.path:__data/${MYTASK}/dev_subsample.jsonl info_choice:i_${MYTASK} eval.metrics:z_f1
# --
MDIR=./
for MYTASK in sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do
  CUDA_VISIBLE_DEVICES=0 python3 -m mspx.znew.prompt.proc.test $MDIR/_conf model_load_name:$MDIR/zmodel.bestn load_half:1 mixed_precision:fp16 test0.path:__data/${MYTASK}/dev_subsample.jsonl info_choice:i_${MYTASK} eval.metrics:z_f1 test0.output_path:out.${MYTASK}.json |& tee _log.${MYTASK}
done
# --
