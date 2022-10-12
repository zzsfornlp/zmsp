#

# prepare ar/zh

# --
# ace
for ff in ../../events/data/split/{ar,zh}.ace.{train,dev,test}.json; do
python3 -m msp2.tasks.zmtl3.scripts.data.prep_ace_arzh input_file:$ff "output_file:$(basename $ff)"
done |& tee _log_ace3
for cl in ar zh; do
for wset in train dev test; do
  cp ${cl}.ace.${wset}.json ../../events/data/data21f/${cl}.ace3.${wset}.json
done
done

# --
# prepare data/frames from c12
echo "See msp2/docs/srl_cl/prep.sh for the preparations of conll12 data ..."
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_prep_c09 ../../events/data/data21f/en.ewt.train.ud.json en.ewt.train.ud.json
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_prep_c09 ../../ud/pb12/ar.train.conll.ud.json ar.c12.train.ud.json
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_prep_c09 ../../ud/pb12/zh.train.conll.ud.json zh.c12.train.ud.json
#Arg: Counter({'ARG1': 26864, 'ARG0': 15270, 'ARG2': 9710, 'ARGM-LOC': 1625, 'ARG3': 637, 'ARG4': 401, 'ARG1-DSP': 6, 'ARG5': 4})
#Arg: Counter({'ARG1': 17351, 'ARG0': 10427, 'ARG2': 6146, 'ARGM-LOC': 1671, 'ARG3': 424, 'ARG4': 75})
#Arg: Counter({'ARG1': 73001, 'ARG0': 67142, 'ARG2': 7888, 'ARGM-LOC': 6554, 'ARG3': 491, 'ARG4': 42})

# --
# prepare syn
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn2 input_file:en.ewt.train.ud.json output_file:en.ewt.train.ud_syn2.json language:en
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn2 input_file:ar.c12.train.ud.json output_file:ar.c12.train.ud_syn2.json language:ar
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn2 input_file:zh.c12.train.ud.json output_file:zh.c12.train.ud_syn2.json language:zh
#Create syn frames en.ewt.train.ud.json -> en.ewt.train.ud_syn2.json: Counter({'tok': 204586, 'args': 49756, 'pred': 33550, 'sent': 12543, 'inst': 540})
#Create syn frames ar.c12.train.ud.json -> ar.c12.train.ud_syn2.json: Counter({'tok': 242702, 'args': 88848, 'pred': 59118, 'inst': 7422, 'sent': 7422})
#Create syn frames zh.c12.train.ud.json -> zh.c12.train.ud_syn2.json: Counter({'tok': 756063, 'args': 202830, 'pred': 140483, 'inst': 36487, 'sent': 36487})

# collect onto (no further aug here!)
{
# syn
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:en.ewt.train.ud_syn2.json language:en output_onto:frames.en.syn2.json add_dummy_tpl:1 rm_no_args:1
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:ar.c12.train.ud_syn2.json language:ar output_onto:frames.ar.syn2.json add_dummy_tpl:1 rm_no_args:1
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:zh.c12.train.ud_syn2.json language:zh output_onto:frames.zh.syn2.json add_dummy_tpl:1 rm_no_args:1
# sem
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:ar.c12.train.ud.json language:ar output_onto:frames.ar.sem.json add_dummy_tpl:1 rm_no_args:1
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:zh.c12.train.ud.json language:zh output_onto:frames.zh.sem.json add_dummy_tpl:1 rm_no_args:1
} |& tee _log_collect

# merge them
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:frames.ar.sem.json,frames.zh.sem.json output_onto:merged_cl3sem0.json
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:frames.en.syn2.json,frames.ar.syn2.json,frames.zh.syn2.json output_onto:merged_cl3syn.json
#Write Onto:frames=11814,roles=22059,core=22059 to merged_cl3sem0.json
#Write Onto:frames=6,roles=18,core=18 to merged_cl3syn.json

# --
# map fn role names & qadistill
mkdir _xfnl
# --
# fn-pred
ii=0
for ff in ar.c12.train.ud.json zh.c12.train.ud.json ; do
mdir=../run_zzf_xfnl_ALL/run_zzf_xfnl_0/
CUDA_VISIBLE_DEVICES=$ii python3 -m msp2.tasks.zmtl3.main.test $mdir/_conf model_load_name:$mdir/zmodel.best.m vocab_load_dir:$mdir/ log_stderr:1 log_file: test0.input_dir:./ test0.output_dir:_xfnl test0.group_files:$ff arg0.default_frame_name:Event |& tee _xfnl/_log_${ff:0:2} &
ii=$((ii+1))
done
# --
# stat and map
python3 -m msp2.tasks.zmtl3.scripts.srl.s22_map_fn 'input_files:_xfnl/_zout*.json' input_onto:merged_cl3sem0.json output_onto:_xfnl/fn_pred_onto.json
#Stat = {'n_frames': 12923, 'n_roles': 19869, 'c_frames': 137090, 'c_roles': 182987}
#Load (json) from merged_cl3sem0.json: Onto:frames=11814,roles=22059,core=22059
#Convert roles:
#all_frame: 11814
#all_frame_map1: 11814 (1.00)
#all_roleC: 19758
#all_roleCH: 19758
#all_roleCH_c01: 363 (0.02)
#all_roleCH_perc=0: 141 (0.01)
#all_roleCH_perc=1: 1727 (0.09)
#all_roleCH_perc=2: 4638 (0.23)
#all_roleCH_perc=3: 1875 (0.09)
#all_roleCH_perc=4: 11377 (0.58)
#all_roleM: 2301
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:_xfnl/fn_pred_onto.json output_onto:merged_cl3semX.json
# --
# qadistill
ii=0
for ff in ar.c12.train.ud.json zh.c12.train.ud.json ; do
CUDA_VISIBLE_DEVICES=$ii python3 -m msp2.tasks.zmtl3.main.stream device:0 'conf_sbase:data:pbfn;task:arg' arg0.pred_store_scores:1 arg0.pred_no_change:1 arg0.extend_span:0 arg0.np_getter:fn 'arg0.filter_noncore:+spec,-*' arg0.arg_mode:mrc 'model_load_name:../run_zzf_xqa_ALL/run_zzf_xqa_0/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role' arg0.mix_evt_ind:0.5 arg0.b_model:xlm-roberta-base arg0.build_oload:merged_cl3semX.json stream_input:$ff stream_output:${ff%.json}.q1X.json &
ii=$((ii+1))
done
# --
