#

# prepare all UD related data

# --
# UD v1.4
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1827/ud-treebanks-v1.4.tgz
tar -zxvf ud-treebanks-v1.4.tgz
# further make a simpl-zh
mkdir ud-treebanks-v1.4/UD_Chinese-S/
for ff in ud-treebanks-v1.4/UD_Chinese/*.conllu; do
  python3 zh_t2s.py <$ff >"ud-treebanks-v1.4/UD_Chinese-S/`basename ${ff}`"
done

# --
# UD v2.7
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz
tar -zxvf ud-treebanks-v2.7.tgz
# specific for Arabic-NYUAD
cd ud-treebanks-v2.7/UD_Arabic-NYUAD/
ln -s ../../../pb/patb/integrated/ .  # note: PATB files
for ff in *.conllu; do
  java -jar merge.jar $ff integrated
  mv $ff $ff.blank
  mv ${ff%.conllu}.merged.conllu $ff
done
cd ${OLDPWD}

# --
# upb1.0
# (21.01.24): 60e2fb824e304c90cbee692aa3adadcf54f5c73f
git clone https://github.com/System-T/UniversalPropositions
# zh-simpl
mkdir UniversalPropositions/UP_Chinese-S/
for ff in UniversalPropositions/UP_Chinese/*.conllu; do
  python3 zh_t2s.py <$ff >"UniversalPropositions/UP_Chinese-S/`basename ${ff}`"
done
# convert to zjson
export PYTHONPATH="`readlink -f ../../zsp2021/src/`:${PYTHONPATH}"
mv UniversalPropositions/UP_English-EWT UniversalPropositions/_UP_English-EWT
for ff in UniversalPropositions/UP*/*.conllu; do
  python3 -m msp2.cli.change_format R.input_path:$ff R.input_format:conllup W.output_path:${ff%.conllu}.json
done |& tee _log_up_change
# upb1.0 styled ewt
# (21.02.25): b677ea28ef4dc049027b2a5fc6e002a4c692d0a7
git clone https://github.com/scofield7419/XSRL-ACL
for ff in XSRL-ACL/UP_English/*.conllu; do
  python3 -m msp2.cli.change_format R.input_path:$ff R.input_format:conllup W.output_path:${ff%.conllu}.json
done
ln -s ../XSRL-ACL/UP_English/ UniversalPropositions/UP_English2-EWT
# simple combine the two spanish ones
for wset in train dev test; do
  cat ../UP_Spanish/es-up-${wset}.json ../UP_Spanish-AnCora/es_ancora-up-${wset}.json > es2-up-${wset}.json;
done

# --
# ewt + ud1.4
# combine ewt with udv1.4 and make it dep-srl
# note: currently convert all roles, may prune out some later ...
cp ../../pb/pb/ewt.*.conll .
for wset in train dev test; do
  python3 assign_anns.py input.input_path:ewt.${wset}.conll aux.input_path:../ud-treebanks-v1.4/UD_English/en-ud-${wset}.conllu output.output_path:en_span.ewt.${wset}.json input.input_format:conllpb aux.input_format:conllu output_sent_and_discard_nonhit:1
  python3 span2dep.py en_span.ewt.${wset}.json en.ewt.${wset}.json
done |& tee _log.ewt  # hit for train/dev/test -> 0.9999/1.0000/0.9990

# --
# fi-pb (simply read it into zjson)
# (21.01.26): 77a694a765a93d4f944bb9302ea5d1f2132d9cdd
for wset in train dev test; do
  # note: it seems that other fields are exactly the same as udv14
  wget https://raw.githubusercontent.com/TurkuNLP/Finnish_PropBank/data/fipb-ud-${wset}.conllu
  python3 read_fipb.py fipb-ud-${wset}.conllu fipb-ud-${wset}.json
done |& tee _log.fipb
# --

# --
# pb-conll12 + ud2
# note: first get those without deps
for wset in train dev test; do
for cl in en zh ar; do
#  ln -s ../../pb/conll12d/${cl}.${wset}.conll.json .
  ln -s ../../pb/conll12e/${cl}.${wset}.conll.json .
done; done
# then collect trees from
#bash assign_p2d/assign.sh
# further auto-parse with ud3 (see tune0207ud.py for zgo)
#for cl in en zh ar; do
#  for ff in ${cl}.*.ud.json; do
#    echo "#==" $ff
#    zgo ../../parsers/models2/m_${cl}/ zjson $ff ${ff%.ud.json}.ud3.json
#  done
#done |& tee _log.pb12.ud3
# arg-head-agree => en[~96/~90], zh[~90/~75], ar[...]
# --

# --
# pbfn (pb3(ewt/onto), fn1.5/1.7) (together with ud2)
# first pb3
PB3_DIR="../../pb/pb/"
for wset in train dev test; do
#  python3 assign_anns.py input.input_path:${PB3_DIR}/ewt.${wset}.conll aux.input_path:../ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud-${wset}.conllu output.output_path:en.ewt.${wset}.json input.input_format:conllpb aux.input_format:conllu output_sent_and_discard_nonhit:1
  python3 assign_anns.py input.input_path:${PB3_DIR}/ewt.${wset}.conll.ud.json aux.input_path:../ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud-${wset}.conllu output.output_path:en.ewt.${wset}.ud.json input.input_format:zjson aux.input_format:conllu output_sent_and_discard_nonhit:0
#  python3 -m msp2.cli.analyze ud gold:en.ewt.${wset}.json preds:${PB3_DIR}/ewt.${wset}.conll.ud.json
  python3 -m msp2.cli.evaluate dpar gold.input_path:en.ewt.${wset}.ud.json pred.input_path:${PB3_DIR}/ewt.${wset}.conll.ud.json
  python3 eval_arg_head.py gold.input_path:en.ewt.${wset}.ud.json pred.input_path:${PB3_DIR}/ewt.${wset}.conll.ud.json
done |& tee _log.ewt
# then simply ln ontonotes!
for wset in train dev test conll12-test; do
  ln -s ${PB3_DIR}/ontonotes.${wset}.conll.ud.json en.onto.${wset}.ud.json
done
# note: also resplit ontonotes for conll12 splits! (see resplit_onto.py)
# note: stat for onto data: slightly fewer f/s for UNK(extra) ones: ~2.6 vs ~2.8
# --
# then fn15/17
FN_DIR="../../fn/parsed/"  # these are already parsed by stanza
for vfn in 15 17; do
  ln -s ${FN_DIR}/fn${vfn}_exemplars.filtered.json en.fn${vfn}.exemplars.ud2.json  # use filtered version!
  for wset in train dev test; do
    ln -s ${FN_DIR}/fn${vfn}_fulltext.${wset}.json en.fn${vfn}.${wset}.ud2.json  # split!
  done
done
## then further parse "en.fn*" to ud3 with our own parser... (see tune0207ud.py for zgo)
#for ff in en.fn*.ud2.json; do
#    echo "#==" $ff
#    zgo ../../parsers/models2/m_en/ zjson $ff ${ff%.ud2.json}.ud3.json
#done |& tee _log.fn.ud3
# --

# --
# stat
for ff in UniversalPropositions/*/*.json pb2/*.json pb12/*.ud.json pbfn/*.ud.json pbfn/*.ud3.json ; do
  echo "ZZ ${ff}"
  python3 stat_udsrl.py zjson $ff
done |& tee _log.stat.udsrl
# cat _log.stat.udsrl | grep -E "CC|ZZ"
# build frames
python3 -m msp2.scripts.ud.frames.read_frames onto:pb dir:../../pb/pb/propbank-frames-3.1/frames/ save_txt:frames.pb3.txt save_pkl:frames.pb3.pkl
python3 -m msp2.scripts.ud.frames.read_frames onto:fn dir:../../fn/fndata-1.5/frame/ save_txt:frames.fn15.txt save_pkl:frames.fn15.pkl
python3 -m msp2.scripts.ud.frames.read_frames onto:fn dir:../../fn/fndata-1.7/frame/ save_txt:frames.fn17.txt save_pkl:frames.fn17.pkl
# python3 -m msp2.scripts.ud.frames.read_frames query:1 load_pkl:? onto:?
