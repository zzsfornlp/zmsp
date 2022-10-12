#

# prepare for clsrl1

export PYTHONPATH="`readlink -f ../../src/`:${PYTHONPATH}"

# simply cp cl0's en set!
cp ../pb2/en.ewt.* .
# and shuffle fi ones
for wset in train dev test; do
  python3 sample_shuffle.py input:../pb2/fipb-ud-${wset}.json output:fipb.${wset}.json shuffle:1
done
# and shuffle the train/dev multiple times
for ii in {1..5}; do
  for wset in train dev; do
    python3 sample_shuffle.py input:../pb2/fipb-ud-${wset}.json output:fipb.${wset}.s${ii}.json shuffle_times:${ii}
  done
done
# --
# also get ontonotes
for wset in train dev test; do
  python3 span2dep.py ../pbfn/en.ontoC.${wset}.ud.json en.ontoC.${wset}.json
done |& tee _log.s2d.onto
