#

# run the qa model on srl datasets

# --
CCMD="python3 -m msp2.tasks.zmtl3.main.stream device:0 'conf_sbase:data:pbfn;task:arg' arg0.pred_store_scores:1 arg0.pred_no_change:1 arg0.extend_span:0 arg0.np_getter:fn 'arg0.filter_noncore:+spec,-*' arg0.build_oload:pbfn arg0.arg_mode:mrc"
CCMD_Q1="$CCMD 'model_load_name:../run_zzf_qa_ALL/zmodel.qa.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role' arg0.mix_evt_ind:0.5"
# --
DDIR="../../events/data/data21f/"
# first run these on EWT
#for ff in en.ewt.train.ud.json; do
#$CCMD_Q1 stream_input:$DDIR/$ff stream_output:${ff%.json}.q1t0.json
#$CCMD_Q1 arg0.mrc_use_tques:1 stream_input:$DDIR/$ff stream_output:${ff%.json}.q1t1.json
#$CCMD_Q1 arg0.mrc_use_tques:1 arg0.mrc_add_t:1 stream_input:$DDIR/$ff stream_output:${ff%.json}.q1t2.json
#done
# --
# run them
for ff in en.ewt.train.ud.json en.ewt.dev.ud.json en.ontoC.train.ud.json en.nb_f0.train.ud.json en.fn17.exemplars.ud2.json; do
CUDA_VISIBLE_DEVICES=2 $CCMD_Q1 stream_input:$DDIR/$ff stream_output:${ff%.json}.q1.json
done |& tee _log
#for ff in en.amr.all.ud2.json en.msamr.all.ud2.json; do
#  CUDA_VISIBLE_DEVICES=2 $CCMD_Q1 stream_input:$DDIR/$ff stream_output:${ff%.json}.q1.json
#done |& tee _logamr
# --
# run with qa+msent (nope for fn-exemplars)
for ff in en.ewt.train.ud.json en.ewt.dev.ud.json en.ontoC.train.ud.json en.nb_f0.train.ud.json; do
CUDA_VISIBLE_DEVICES=2 $CCMD_Q1 'model_load_name:../run_zzf_mqa_ALL/run_zzf_mqa_0/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role' arg0.ctx_nsent_rates:0,0,1 arg0.max_content_len:256 stream_input:$DDIR/$ff stream_output:${ff%.json}.q2.json
done |& tee _log2
# --
