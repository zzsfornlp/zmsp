#

# prepare for clsrl0

export PYTHONPATH="`readlink -f ../../src/`:${PYTHONPATH}"

# first change ewt's data to short arg labels
for wset in train dev test; do
  python3 change_arg_label.py input_path:../pb2/en.ewt.${wset}.json output_path:en.ewt.${wset}.json
done

# then get others' ud and shuffle!
declare -A UDTABLE=(
  [en]="UD_English" [zh]="UD_Chinese-S" [fi]="UD_Finnish" [fr]="UD_French" [de]="UD_German"
  [it]="UD_Italian" [pt_bosque]="UD_Portuguese-Bosque" [es_ancora]="UD_Spanish-AnCora" [es]="UD_Spanish"
)
for CL in "${!UDTABLE[@]}"; do
  FULL_CL=${UDTABLE[${CL}]}
  echo ${CL} ${FULL_CL};
  for wset in train dev test; do
    python3 -m msp2.cli.change_format R.input_path:../ud-treebanks-v1.4/${FULL_CL}/${CL}-ud-${wset}.conllu R.input_format:conllu W.output_path:_tmp.json  # change to json
    python3 sample_shuffle.py input:_tmp.json output:${CL}.ud.${wset}.json shuffle:1
  done
done |& tee _log

# combine the spanish ones
for wset in train dev test; do
  cat es_ancora.ud.${wset}.json es.ud.${wset}.json >_tmp.json
  python3 sample_shuffle.py input:_tmp.json output:es2.ud.${wset}.json shuffle:1
done
