#

# prepare frames and data of srl
# (dir: @frames)

# --
# step 0: what to prepare?
echo "First prepare all the frames: 'fndata-1.7'(https://framenet.icsi.berkeley.edu/) 'propbank-frames-3.1'(https://github.com/propbank/propbank-frames) 'nombank.1.0'(https://nlp.cs.nyu.edu/meyers/nombank/nombank.1.0/)"
echo "Then also prepare all the data at \$DATA_DIR/en.DATASET.SPLIT.ud?.json"
# in our run, we use a specific DATA_DIR
DATA_DIR=../../events/data/data21f/

# --
# step 1: read raw frames
python3 -m msp2.tasks.zmtl3.scripts.srl.s1_read_frames onto:fn output:f_fn.json dir:fndata-1.7/frame/ |& tee _logs1_fn
python3 -m msp2.tasks.zmtl3.scripts.srl.s1_read_frames onto:pb output:f_pb.json dir:propbank-frames-3.1/frames/ |& tee _logs1_pb
python3 -m msp2.tasks.zmtl3.scripts.srl.s1_read_frames onto:nb output:f_nb.json dir:nombank.1.0/frames/ |& tee _logs1_nb
#Read Onto:frames=679,roles=917,core=2143 from fndata-1.7/frame/: Counter({'frame_all': 1221, 'role': 917, 'frame_inc': 679, 'frame_noverb': 522, 'frame_rm': 20})
#Read Onto:frames=8752,roles=23166,core=23214 from propbank-frames-3.1/frames/: Counter({'role_core': 23145, 'role_core_vn': 10825, 'frame_all': 10687, 'frame_all_v1': 8752, 'frame_all_v0': 1935, 'frame_rm': 1935, 'frame_rm_j': 1646, 'frame_rm_n': 357})
#Read Onto:frames=5573,roles=14937,core=14916 from nombank.1.0/frames/: Counter({'frame_all': 5577, 'frame_all_v0': 5577, 'frame_skip': 4})

# --
# step 1.5: deal with nb (filter frames and convert data)
# rm ./en.nb*
for rm_reuse in 0 1; do
  python3 -m msp2.tasks.zmtl3.scripts.srl.s15_filter_nb onto_nb:f_nb.json onto_pb:f_pb.json input_files:${DATA_DIR}/en.nb.train.ud.json,${DATA_DIR}/en.nb.dev.ud.json,${DATA_DIR}/en.nb.test.ud.json "output_sub:en.nb,en.nb_f${rm_reuse}" rm_reuse:${rm_reuse}
done |& tee _logs15_nb
#Process Onto:frames=5573,roles=14937,core=14916: Counter({'nb_all': 5573, 'rr_ok': 2573, 'nb_nosrc': 2090, 'nb_good': 1417, 'nb_reuse': 1358, 'nb_fine': 587, 'rr_smaller': 496, 'rr_differ': 293, 'nb_badsrc': 121})
#Read ../../events/data/data21f//en.nb.train.ud.json: Counter({'all_frame': 92652, 'all_sent': 39832, 'frame_change': 38737, 'frame_kept_nb_good': 33955, 'frame_rm_nb_nosrc': 31447, 'frame_kept_nb_reuse': 15124, 'frame_kept_nb_fine': 7984, 'frame_rm_unk': 2492, 'frame_rm_nb_badsrc': 1650})
#Read ../../events/data/data21f//en.nb.dev.ud.json: Counter({'all_frame': 3869, 'all_sent': 1700, 'frame_change': 1634, 'frame_kept_nb_good': 1530, 'frame_rm_nb_nosrc': 1122, 'frame_kept_nb_reuse': 659, 'frame_kept_nb_fine': 338, 'frame_rm_unk': 145, 'frame_rm_nb_badsrc': 75})
#Read ../../events/data/data21f//en.nb.test.ud.json: Counter({'all_frame': 5382, 'all_sent': 2416, 'frame_change': 2164, 'frame_kept_nb_good': 2036, 'frame_rm_nb_nosrc': 1781, 'frame_kept_nb_reuse': 844, 'frame_kept_nb_fine': 451, 'frame_rm_unk': 154, 'frame_rm_nb_badsrc': 116})

# --
# step 2: prepare extra aug info (qwords, preps, dist) for onto
python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_onto:f_pb.json output_onto:f_pbA.json "input_files:$DATA_DIR/en.ontoC.*,$DATA_DIR/en.ewt.*" |& tee _logs2_pbA  # aug
python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_onto:f_pb.json output_onto:f_pbN.json tpl_core_nameorder:1 "input_files:$DATA_DIR/en.ontoC.*,$DATA_DIR/en.ewt.*" |& tee _logs2_pbN  # aug + name-order
python3 -m msp2.tasks.zmtl3.scripts.srl.s2_aug_onto input_onto:f_fn.json output_onto:f_fnA.json "input_files:$DATA_DIR/en.fn17.*" |& tee _logs2_fnA  # aug
# pb / fn
#all_frame: 8752                          all_frame: 679
#all_frame_fok: 5453 (0.62)               all_frame_fok: 643 (0.95)
#all_frame_frm: 3299 (0.38)               all_frame_frm: 36 (0.05)
#all_frame_nocount: 3299 (0.38)           all_frame_nocount: 36 (0.05)
#all_frame_noverb: 3566 (0.41)            all_frame_noverb: 50 (0.07)
#all_frame_noverbpass: 6768 (0.77)        all_frame_noverbpass: 243 (0.36)
#all_frame_ok_disagreename: 890 (0.10)    all_frame_ok_disagreename: 423 (0.62)
#all_role: 23166                          all_role: 917
#all_role_nocount: 11501 (0.50)           all_role_nocount: 84 (0.09)
#all_role_nopron: 18533 (0.80)            all_role_nopron: 515 (0.56)
#all_role_yeswho: 2155 (0.09)             all_role_yeswho: 179 (0.20)

# --
# step 2.*: add vn/fn role names for pb
# ==
# => vn, note: no need for this since the differences are little
# get semlink2 maps
#wget https://raw.githubusercontent.com/cu-clear/semlink/3d6e3d7581984a59a36b9ad8037daca16f2555d9/instances/pb-vn2.json
#python3 -m msp2.tasks.zmtl3.scripts.srl.s21_map_vn input_onto:./f_pbA.json output_onto: pb2vn:./pb-vn2.json "check_input_files:$DATA_DIR/en.ontoC.*,$DATA_DIR/en.ewt.*"
#Load (json) from ./f_pbA.json: Onto:frames=5453,roles=23166,core=14595
#Convert roles:
#all_frame: 5453
#all_frame_map0: 2634 (0.48)
#all_frame_map1: 2152 (0.39)
#all_frame_map2: 667 (0.12)
#all_frame_mapC: 319 (0.06)
#all_roleC: 14535
#all_roleC_C: 373 (0.03)
#all_roleC_O0M0: 7263 (0.50)
#all_roleC_O0M1: 137 (0.01)
#all_roleC_O1M0: 1719 (0.12)
#all_roleC_O1M1: 5416 (0.37)
#all_roleM: 60
# --
# -> overall, the coverage seems reasonable (all_arg_hasvn/all_arg_hit=0.8141):
# Read all files: Counter({'all_arg': 993567, 'all_arg_hit': 452898, 'all_arg_hasvn': 368699, 'all_frame': 327355, 'all_inst': 110848, 'all_sent': 110848, 'all_frame_None': 6666})
# ==
# -> fn:
# see 's22_map_fn.py' for the preparation of maps
for aa in 'A' 'N'; do
for vv in 0 1; do
python3 -m msp2.tasks.zmtl3.scripts.srl.s22_map_fn input_stat:fn_pred_files${vv}.json input_onto:f_pb${aa}.json output_onto:f_pb${aa}m${vv}.json
done
done |& tee _logs25_pb2fn
# ->
#Stat = {'n_frames': 6043, 'n_roles': 12403, 'c_frames': 327355, 'c_roles': 456864}
#Load (json) from f_pbA.json: Onto:frames=5453,roles=23166,core=14595
#Convert roles:
#all_frame: 5453
#all_frame_map1: 5453 (1.00)
#all_roleC: 14535
#all_roleCH: 11646
#all_roleCH_c01: 159 (0.01)       all_roleCH_c01: 137 (0.01)
#all_roleCH_perc=0: 68 (0.01)     all_roleCH_perc=0: 52 (0.00)
#all_roleCH_perc=1: 1104 (0.09)   all_roleCH_perc=1: 1109 (0.10)
#all_roleCH_perc=2: 2813 (0.24)   all_roleCH_perc=2: 2774 (0.24)
#all_roleCH_perc=3: 1974 (0.17)   all_roleCH_perc=3: 2060 (0.18)
#all_roleCH_perc=4: 5687 (0.49)   all_roleCH_perc=4: 5651 (0.49)
#all_roleC_miss: 2889 (0.20)
#all_roleM: 60

# --
# step 3: merge the pb/fn frames together to make things easier
for pp in f_pb{A,N}m{0,1}.json; do
  python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:f_fnA.json,$pp output_onto:merged_$pp
done

# --
# step 4: (specifically) change pb data from sents into docs
# @DATA_DIR
cd $(readlink -f "$DATA_DIR")
if [[ -d pb_sents ]]; then
  echo "Already there!!"
else
  mkdir -p pb_sents
  mv en.{ewt,ontoC}* pb_sents
fi
for ff in pb_sents/*.json; do
  python3 -m msp2.tasks.zmtl3.scripts.srl.s4_pbs2d $ff "$(basename $ff)"
done
# ->
#pbs2d from pb_sents/en.ewt.dev.ud.json to en.ewt.dev.ud.json: Counter({'sent': 1974, 'doc': 297})
#pbs2d from pb_sents/en.ewt.test.ud.json to en.ewt.test.ud.json: Counter({'sent': 2062, 'doc': 308})
#pbs2d from pb_sents/en.ewt.train.ud.json to en.ewt.train.ud.json: Counter({'sent': 12543, 'doc': 540})
#pbs2d from pb_sents/en.ontoC.dev.ud.json to en.ontoC.dev.ud.json: Counter({'sent': 9603, 'doc': 222})
#pbs2d from pb_sents/en.ontoC.test.ud.json to en.ontoC.test.ud.json: Counter({'sent': 9479, 'doc': 222})
#pbs2d from pb_sents/en.ontoC.train.ud.json to en.ontoC.train.ud.json: Counter({'sent': 75187, 'doc': 1940})

# ==
# extraly including multiple fns
python3 -m msp2.tasks.zmtl3.scripts.srl.s22_map_fn input_stat:fn_pred_files0.json input_onto:f_pbA.json output_onto:f_pbAm0v2.json topk:5 topp:0.9
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:f_fnA.json,f_pbAm0v2.json output_onto:merged_f_pbAm0v2.json
#all_roleCH_NUM=1: 6389 (0.55); all_roleCH_NUM=2: 2767 (0.24); all_roleCH_NUM=3: 1266 (0.11); all_roleCH_NUM=4: 595 (0.05); all_roleCH_NUM=5: 629 (0.05)
# ==
