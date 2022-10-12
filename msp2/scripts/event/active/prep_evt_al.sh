#

# prepare the low-resource & minimal-supervision settings for evt

# --
WDIR="../../"
export PYTHONPATH="${WDIR}/src/"

# --
# sample 10% of ace training
cat ${WDIR}/events/data/splitS/en.ace.train.json | head -53 >en.ace53.train.json
# filter fn/pb ones
for _fc in 1 2 3; do
  python3 filter_data.py lemma_filter_count:${_fc} input_evt:en.ace53.train.json frame_file:${WDIR}/ud/pbfn/frames.fn17.pkl input_data:${WDIR}/ud/pbfn/en.fn17.train.ud3.json output_data:en.fn17.train.f53c${_fc}.ud3.json
  python3 filter_data.py lemma_filter_count:${_fc} input_evt:en.ace53.train.json frame_file:${WDIR}/ud/pbfn/frames.fn17.pkl input_data:${WDIR}/ud/pbfn/en.fn17.exemplars.ud3.json output_data:en.fn17.exemplars.f53c${_fc}.ud3.json
  python3 filter_data.py lemma_filter_count:${_fc} input_evt:en.ace53.train.json frame_file:${WDIR}/ud/pbfn/frames.pb3.pkl input_data:${WDIR}/ud/pbfn/en.ewt.train.ud.json output_data:en.ewt.train.f53c${_fc}.ud.json
  python3 filter_data.py lemma_filter_count:${_fc} input_evt:en.ace53.train.json frame_file:${WDIR}/ud/pbfn/frames.pb3.pkl input_data:${WDIR}/ud/pbfn/en.onto.train.ud.json output_data:en.onto.train.f53c${_fc}.ud.json
done |& tee _log.filter.f53
# less _log.filter.f53 | grep -E -- "-stat|All frames|Read from"
# --
#Read from en.ace53.train.json: Counter({'num_sent': 1464, 'num_evt': 551, 'num_inst': 53})
#Lemma-stat: evt=27,itemsA=235,itemsS=213 => 213/82/49
#All frames = 1221
#Frame-stat: evt=26,itemsA=464,itemsS=295 => 295/148/93
#Read from ../..//ud/pbfn/en.fn17.train.ud3.json: Counter({'num_evt': 17064, 'num_evtV': 7304, 'num_sent': 3139, 'num_inst': 47, 'num_instV': 47})
#Read from ../..//ud/pbfn/en.fn17.train.ud3.json: Counter({'num_evt': 17064, 'num_evtV': 4479, 'num_sent': 3139, 'num_inst': 47, 'num_instV': 47})
#Read from ../..//ud/pbfn/en.fn17.train.ud3.json: Counter({'num_evt': 17064, 'num_sent': 3139, 'num_evtV': 2569, 'num_inst': 47, 'num_instV': 47})
# --
#Read from en.ace53.train.json: Counter({'num_sent': 1464, 'num_evt': 551, 'num_inst': 53})
#Lemma-stat: evt=27,itemsA=235,itemsS=213 => 213/82/49
#All frames = 1221
#Frame-stat: evt=26,itemsA=464,itemsS=295 => 295/148/93
#Read from ../..//ud/pbfn/en.fn17.exemplars.ud3.json: Counter({'num_evt': 173391, 'num_inst': 173029, 'num_sent': 173029, 'num_evtV': 83455, 'num_instV': 83333})
#Read from ../..//ud/pbfn/en.fn17.exemplars.ud3.json: Counter({'num_evt': 173391, 'num_inst': 173029, 'num_sent': 173029, 'num_evtV': 49125, 'num_instV': 49047})
#Read from ../..//ud/pbfn/en.fn17.exemplars.ud3.json: Counter({'num_evt': 173391, 'num_inst': 173029, 'num_sent': 173029, 'num_evtV': 35295, 'num_instV': 35264})
# --
#Read from en.ace53.train.json: Counter({'num_sent': 1464, 'num_evt': 551, 'num_inst': 53})
#Lemma-stat: evt=27,itemsA=235,itemsS=213 => 213/82/49
#All frames = 10687
#Frame-stat: evt=27,itemsA=579,itemsS=419 => 419/205/108
#Read from ../..//ud/pbfn/en.ewt.train.ud.json: Counter({'num_evt': 40486, 'num_inst': 12543, 'num_sent': 12543, 'num_evtV': 6312, 'num_instV': 4527})
#Read from ../..//ud/pbfn/en.ewt.train.ud.json: Counter({'num_evt': 40486, 'num_inst': 12543, 'num_sent': 12543, 'num_evtV': 3342, 'num_instV': 2711})
#Read from ../..//ud/pbfn/en.ewt.train.ud.json: Counter({'num_evt': 40486, 'num_inst': 12543, 'num_sent': 12543, 'num_evtV': 1599, 'num_instV': 1401})
# --
#Read from en.ace53.train.json: Counter({'num_sent': 1464, 'num_evt': 551, 'num_inst': 53})
#Lemma-stat: evt=27,itemsA=235,itemsS=213 => 213/82/49
#All frames = 10687
#Frame-stat: evt=27,itemsA=579,itemsS=419 => 419/205/108
#Read from ../..//ud/pbfn/en.onto.train.ud.json: Counter({'num_evt': 314158, 'num_inst': 111105, 'num_sent': 111105, 'num_evtV': 47476, 'num_instV': 36057})
#Read from ../..//ud/pbfn/en.onto.train.ud.json: Counter({'num_evt': 314158, 'num_inst': 111105, 'num_sent': 111105, 'num_evtV': 24904, 'num_instV': 21029})
#Read from ../..//ud/pbfn/en.onto.train.ud.json: Counter({'num_evt': 314158, 'num_inst': 111105, 'num_sent': 111105, 'num_evtV': 12405, 'num_instV': 11088})
# then run with the filtered ones, like "go0505_f53" ...
