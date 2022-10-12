#

# prepare data/frames for zh/es (from conll09)
# (dir: @frames2)

# get data
echo "See msp2/docs/srl_cl/prep.sh for the preparations of conll09 data ..."
cat ../../ud/conll09/convert/es.{train,dev,test}.ud.json | python3 -m msp2.tasks.zmtl3.scripts.srl.sz_prep_c09 '' es.c09.all.ud.json
cat ../../ud/conll09/convert/zh.{train,dev,test}.ud.json | python3 -m msp2.tasks.zmtl3.scripts.srl.sz_prep_c09 '' zh.c09.all.ud.json
cat ../../events/data/data21f/en.ewt.train.ud.json | python3 -m msp2.tasks.zmtl3.scripts.srl.sz_prep_c09 '' en.ewt.train.ud.json
#Prep es  -> es.c09.all.ud.json: Counter({'tok': 528440, 'arg': 122478, 'arg_OK': 85187, 'evt': 54075, 'arg_No': 26546, 'tok_C': 18973, 'inst': 17709, 'sent': 17709, 'tok_F': 10778, 'arg_D': 10745})
#Arg: Counter({'ARG1-PAT': 24794, 'ARG0-AGT': 16243, 'ARG1-TEM': 14739, 'ARG2-ATR': 9965, 'ARGM-LOC': 6102, 'ARG2-NULL': 3008, 'ARG2-BEN': 2740, 'ARG1-NULL': 2137, 'ARG2-LOC': 1936, 'ARG0-CAU': 1213, 'ARG4-DES': 743, 'ARG2-EXT': 348, 'ARG3-ORI': 268, 'ARG2-EFI': 230, 'ARG3-BEN': 176, 'ARG4-EFI': 130, 'ARG2-EXP': 108, 'ARG3-FIN': 85, 'ARG3-EIN': 65, 'ARG1-LOC': 40, 'ARG1-EXT': 37, 'ARG2-INS': 22, 'ARG3-EXP': 19, 'ARG0-EXP': 12, 'ARG0-SRC': 9, 'ARG3-INS': 5, 'ARG0-NULL': 3, 'ARG3-LOC': 3, 'ARG2-TEM': 2, 'ARG3-ATR': 2, 'ARG0-PAT': 1, 'ARG3-NULL': 1, 'ARG3-DES': 1})
#arg_D: 10745 (0.09) / arg_No: 26546 (0.22) / arg_OK: 85187 (0.70)
#Prep es  -> zh.c09.all.ud.json: Counter({'tok': 731833, 'arg': 278135, 'arg_OK': 176862, 'evt': 123198, 'arg_No': 101167, 'inst': 26595, 'sent': 26595, 'arg_D': 106})
#Arg: Counter({'ARG1': 84859, 'ARG0': 73434, 'ARG2': 9952, 'ARGM-LOC': 7857, 'ARG3': 694, 'ARG4': 65, 'ARG5': 1})
#arg_D: 106 (0.00) / arg_No: 101167 (0.36) / arg_OK: 176862 (0.64)
#Prep es  -> en.ewt.train.ud.json: Counter({'tok': 204586, 'arg': 122036, 'arg_No': 67341, 'arg_OK': 54517, 'evt': 40486, 'sent': 12543, 'inst': 540, 'arg_D': 178, 'tok_C': 76, 'tok_F': 5})
#Arg: Counter({'ARG1': 26864, 'ARG0': 15270, 'ARG2': 9710, 'ARGM-LOC': 1625, 'ARG3': 637, 'ARG4': 401, 'ARG1-DSP': 6, 'ARG5': 4})
#arg_D: 178 (0.00) / arg_No: 67341 (0.55) / arg_OK: 54517 (0.45)

# create syn frames
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn input_file:en.ewt.train.ud.json output_file:en.ewt.train.ud_syn.json language:en
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn input_file:es.c09.all.ud.json output_file:es.c09.all.ud_syn.json language:es
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn input_file:zh.c09.all.ud.json output_file:zh.c09.all.ud_syn.json language:zh
#Create syn frames en.ewt.train.ud.json -> en.ewt.train.ud_syn.json: Counter({'tok': 204586, 'args': 46352, 'pred': 30813, 'sent': 12543, 'inst': 540})
#Create syn frames es.c09.all.ud.json -> es.c09.all.ud_syn.json: Counter({'tok': 528440, 'args': 115209, 'pred': 73777, 'inst': 17709, 'sent': 17709})
#Create syn frames zh.c09.all.ud.json -> zh.c09.all.ud_syn.json: Counter({'tok': 731833, 'args': 168585, 'pred': 117565, 'inst': 26595, 'sent': 26595})

# read frames for zh
# first read zh frame files (ref-onto) from c09
# note: manually fix ["0413-cheng.xml(missing <frameset>)", "12843-qi-cheng.xml(missing space in attrib)", "1736-jiang.xml(repeated frames?)", "8635-hua-zuo.xml(L5 strange attrib)"]
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_read_c09zhf ../../ud/conll09/2009_conll/CoNLL2009-ST-Chinese/frames/ c09_ref.json |& tee _log.c09zhf
#Read frames from ../../ud/conll09/2009_conll/CoNLL2009-ST-Chinese/frames/: Counter({'role': 26479, 'roleN': 17374, 'frame': 15184, 'file': 14128, 'role1': 8703, 'role?': 207, 'role0': 195})

# prepare word translations
#wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec
#wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.es.align.vec
#wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.zh.align.vec
# nope, these emb-aligns are not accurate
# first export all frame names
#export_frame_names("es.c09.all.ud_syn.json", "_listf.es.syn")
#export_frame_names("zh.c09.all.ud_syn.json", "_listf.zh.syn")
#export_frame_names("es.c09.all.ud.json", "_listf.es.sem")
#export_frame_names("zh.c09.all.ud.json", "_listf.zh.sem")
# ok, simply put into a docx and use google!
#docx2plain(['_listf.es.syn.docx', '_listf.es_en.syn.docx'], '_dict.es_en.syn')
#docx2plain(['_listf.es.sem.docx', '_listf.es_en.sem.docx'], '_dict.es_en.sem')
#docx2plain(['_listf.zh.syn.docx', '_listf.zh_en.syn.docx'], '_dict.zh_en.syn')
#docx2plain(['_listf.zh.sem.docx', '_listf.zh_en.sem.docx'], '_dict.zh_en.sem')

# collect ontos
{
# syn
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:en.ewt.train.ud_syn.json language:en output_onto:frames.en.syn.json
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:es.c09.all.ud_syn.json language:es output_onto:frames.es.syn.json tfile:_dict.es_en.syn
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:zh.c09.all.ud_syn.json language:zh output_onto:frames.zh.syn.json tfile:_dict.zh_en.syn
# sem
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:es.c09.all.ud.json language:es output_onto:frames.es.sem.json tfile:_dict.es_en.sem
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_collect_onto input_file:zh.c09.all.ud.json language:zh output_onto:frames.zh.sem.json ref_onto:c09_ref.json tfile:_dict.zh_en.sem
} |& tee _log_collect
#Collect onto Onto:frames=6773,roles=11411,core=11411 from en.ewt.train.ud_syn.json to frames.en.syn.json: Counter({'arg': 46352, 'evt': 30813, 'sent': 12543, 'oroleA': 11607, 'oroleI': 11411, 'oframe': 6773, 'inst': 540, 'oroleE': 196})
#Collect onto Onto:frames=16984,roles=29798,core=29798 from es.c09.all.ud_syn.json to frames.es.syn.json: Counter({'arg': 115209, 'evt': 73777, 'oroleA': 30297, 'oroleI': 29801, 'inst': 17709, 'sent': 17709, 'oframe': 16986, 'oroleE': 496, 'oframeTF': 2})
#Collect onto Onto:frames=16716,roles=25284,core=25284 from zh.c09.all.ud_syn.json to frames.zh.syn.json: Counter({'arg': 168585, 'evt': 117565, 'inst': 26595, 'sent': 26595, 'oroleA': 25650, 'oroleI': 25284, 'oframe': 16716, 'oroleE': 366})
#Collect onto Onto:frames=4913,roles=9138,core=9138 from es.c09.all.ud.json to frames.es.sem.json: Counter({'arg': 85187, 'evt': 54075, 'inst': 17709, 'sent': 17709, 'oroleA': 10287, 'oroleI': 9150, 'oframe': 4919, 'oroleE': 1137, 'oframeTF': 6})
#Collect onto Onto:frames=13459,roles=20564,core=20564 from zh.c09.all.ud.json to frames.zh.sem.json: Counter({'arg': 176862, 'evt': 123198, 'inst': 26595, 'sent': 26595, 'oroleA': 21447, 'oroleI': 20566, 'oframe': 13625, 'oroleE': 881, 'oframeRF': 164, 'oframeTF': 2})

# aug onto
# v0
#{
## syn
#python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:en.ewt.train.ud_syn.json language:en input_onto:frames.en.syn.json output_onto:framesA.en.syn.json
#python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:es.c09.all.ud_syn.json language:es input_onto:frames.es.syn.json output_onto:framesA.es.syn.json tpl_prep_trans:1
#python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:zh.c09.all.ud_syn.json language:zh input_onto:frames.zh.syn.json output_onto:framesA.zh.syn.json tpl_prep_trans:1
## sem
#python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:es.c09.all.ud.json language:es input_onto:frames.es.sem.json output_onto:framesA.es.sem.json tpl_prep_trans:1
#python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:zh.c09.all.ud.json language:zh input_onto:frames.zh.sem.json output_onto:framesA.zh.sem.json tpl_prep_trans:1
#} |& tee _log_aug
{
# syn
python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:en.ewt.train.ud_syn.json language:en input_onto:frames.en.syn.json output_onto:framesA.en.syn.json rm_frame_noverb:1 rm_frame_norole:1 tpl_frozen_dist:1
python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:es.c09.all.ud_syn.json language:es input_onto:frames.es.syn.json output_onto:framesA.es.syn.json tpl_prep_trans:1 rm_frame_noverb:1 rm_frame_norole:1 tpl_frozen_dist:1
python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:zh.c09.all.ud_syn.json language:zh input_onto:frames.zh.syn.json output_onto:framesA.zh.syn.json tpl_prep_trans:1 rm_frame_noverb:1 rm_frame_norole:1 tpl_frozen_dist:1
# sem
python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:es.c09.all.ud.json language:es input_onto:frames.es.sem.json output_onto:framesA.es.sem.json tpl_prep_trans:1 rm_frame_noverb:1 rm_frame_norole:1 tpl_in_amloc:1 tpl_core_nameorder:1
python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_files:zh.c09.all.ud.json language:zh input_onto:frames.zh.sem.json output_onto:framesA.zh.sem.json tpl_prep_trans:1 rm_frame_noverb:1 rm_frame_norole:1 tpl_in_amloc:1 tpl_core_nameorder:1
} |& tee _log_aug

# merge everything together
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:framesA.es.sem.json,framesA.zh.sem.json output_onto:merged_cl3sem.json
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:framesA.en.syn.json,framesA.es.syn.json,framesA.zh.syn.json output_onto:merged_cl3syn.json
#gzip -c merged_cl3sem.json >onto_cl3sem.json.gz
#gzip -c merged_cl3syn.json >onto_cl3syn.json.gz
#Write Onto:frames=14563,roles=29702,core=27565 to merged_cl3sem.json
#Write Onto:frames=24119,roles=66493,core=48522 to merged_cl3syn.json

# --
# align/map sem frames to English ones (for aug info)
# cp pb data
cat ../../events/data/data21f/en.{ewt,ontoC}* >_en.pb.all.ud.json
for mm in bert-base-multilingual-cased xlm-roberta-base; do
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_align_onto device:0 b_model:${mm} src_onto:pbfn src_files:_en.pb.all.ud.json src_repr:_repr${mm:0:1}_en.pkl trg_onto:merged_cl3sem.json trg_files:es.c09.all.ud.json,zh.c09.all.ud.json trg_repr:_repr${mm:0:1}_sem.pkl |& tee _log_align${mm:0:1}
done
# note: xlmr looks better, use it!
ii=0
for trig_ws in "0,0,1" "0,1,0"; do
for role_ws in "1,0" "0,1"; do
echo ZRUN $ii ${trig_ws} ${role_ws}
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_align_onto device:0 src_onto:pbfn src_repr:_reprx_en.pkl trg_onto:merged_cl3sem.json trg_repr:_reprx_sem.pkl trig_ws:${trig_ws} role_ws:${role_ws} aligned_onto:mapped${ii}_cl3sem.json
ii=$((ii+1))
done
done |& tee _log_align2
#vim -d mapped0_cl3sem.json.txt merged_cl3sem.json.txt
# --
# qadistill on these!
for ff in es.c09.all.ud.json zh.c09.all.ud.json; do
python3 -m msp2.tasks.zmtl3.main.stream device:0 'conf_sbase:data:pbfn;task:arg' arg0.pred_store_scores:1 arg0.pred_no_change:1 arg0.extend_span:0 arg0.np_getter:fn 'arg0.filter_noncore:+spec,-*' arg0.arg_mode:mrc 'model_load_name:../run_zzf_xqa_ALL/run_zzf_xqa_0/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role' arg0.mix_evt_ind:0.5 arg0.b_model:xlm-roberta-base arg0.build_oload:mapped1_cl3sem.json stream_input:$ff stream_output:${ff%.json}.q1.json  # stream_show:1
done

# --
# syn2
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn2 input_file:en.ewt.train.ud.json output_file:en.ewt.train.ud_syn2.json language:en
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn2 input_file:es.c09.all.ud.json output_file:es.c09.all.ud_syn2.json language:es
python3 -m msp2.tasks.zmtl3.scripts.srl.sz_create_syn2 input_file:zh.c09.all.ud.json output_file:zh.c09.all.ud_syn2.json language:zh
#Create syn frames en.ewt.train.ud.json -> en.ewt.train.ud_syn2.json: Counter({'tok': 204586, 'args': 53011, 'pred': 33839, 'sent': 12543, 'inst': 540})
#Create syn frames es.c09.all.ud.json -> es.c09.all.ud_syn2.json: Counter({'tok': 528440, 'args': 131390, 'pred': 75938, 'inst': 17709, 'sent': 17709})
#Create syn frames zh.c09.all.ud.json -> zh.c09.all.ud_syn2.json: Counter({'tok': 731833, 'args': 244314, 'pred': 160158, 'inst': 26595, 'sent': 26595})

# --
# map fn & qadistill
# see "s22_map_fn.py"
#Stat = {'n_frames': 18544, 'n_roles': 27744, 'c_frames': 177273, 'c_roles': 248090}
#Load (json) from merged_cl3sem.json: Onto:frames=14563,roles=29702,core=27565
#Convert roles:
#all_frame: 14563
#all_frame_map1: 14563 (1.00)
#all_roleC: 24632
#all_roleCH: 24632
#all_roleCH_c01: 347 (0.01)
#all_roleCH_perc=0: 108 (0.00)
#all_roleCH_perc=1: 1855 (0.08)
#all_roleCH_perc=2: 5303 (0.22)
#all_roleCH_perc=3: 2442 (0.10)
#all_roleCH_perc=4: 14924 (0.61)
#all_roleM: 2933
ls _xfnl/merged_cl3sem.json
ln -s _xfnl/merged_cl3sem.json merged_cl3semX.json
# --
# distill
ii=0
for ff in es.c09.all.ud.json zh.c09.all.ud.json; do
CUDA_VISIBLE_DEVICES=$ii python3 -m msp2.tasks.zmtl3.main.stream device:0 'conf_sbase:data:pbfn;task:arg' arg0.pred_store_scores:1 arg0.pred_no_change:1 arg0.extend_span:0 arg0.np_getter:fn 'arg0.filter_noncore:+spec,-*' arg0.arg_mode:mrc 'model_load_name:../run_zzf_xqa_ALL/run_zzf_xqa_0/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role' arg0.mix_evt_ind:0.5 arg0.b_model:xlm-roberta-base arg0.build_oload:merged_cl3semX.json stream_input:$ff stream_output:${ff%.json}.q1X.json &
ii=$((ii+1))
done
# --
