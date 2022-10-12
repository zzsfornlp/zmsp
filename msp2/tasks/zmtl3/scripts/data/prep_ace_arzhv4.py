#

# prepare these according to:
# https://github.com/PlusLabNLP/X-Gear/tree/main/preprocessing

import os
import sys
from collections import OrderedDict, Counter
from msp2.utils import zopen, zlog, default_json_serializer, zwarn
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc
from msp2.data.rw import ReaderGetterConf, WriterGetterConf


def convert_xgear(data: dict, stat):
    sent = Sent.create(data['tokens'], id=data['wnd_id'])
    stat['sent'] += 1
    # --
    ef_map = {}  # id->frame
    for v_ef in data['entity_mentions']:
        ef = sent.make_entity_filler(v_ef['start'], v_ef['end']-v_ef['start'], type=v_ef['entity_type'], id=v_ef['id'])
        ef_map[v_ef['id']] = ef
        stat['ef'] += 1
    for v_evt in data['event_mentions']:
        evt = sent.make_event(v_evt['trigger']['start'], v_evt['trigger']['end']-v_evt['trigger']['start'],
                              type=v_evt['event_type'], id=v_evt['id'])
        stat['evt'] += 1
        for v_arg in v_evt['arguments']:
            _arg = ef_map[v_arg['entity_id']]
            evt.add_arg(_arg, role=v_arg['role'])
            stat['arg'] += 1
    # --
    return sent

def convert_ar(data: dict, stat):
    sents = []
    stat['doc'] += 1
    for v_sent in data['sentences']:
        stat['sent'] += 1
        sent = Sent.create([z['text'] for z in v_sent['tokens']])
        ef_map = {}  # id->frame
        for v_ef in v_sent['mentions']:
            span_info0 = v_ef['grounded_span']
            if span_info0['full_span'] != span_info0['head_span']:
                zwarn("It seems that these two are the same in these files?")
            span_info = span_info0['full_span']
            ef = sent.make_entity_filler(
                span_info['start_token'], span_info['end_token']-span_info['start_token']+1, type=v_ef['entity_type'])
            stat['ef'] += 1
            ef_map[v_ef['mention_id']] = ef
        for v_evt in v_sent['basic_events']:
            if len(v_evt['anchors']['spans']) > 1:
                zwarn("More than 1 spans for trigger!")
            span_info0 = v_evt['anchors']['spans'][0]['grounded_span']
            if span_info0['full_span'] != span_info0['head_span']:
                zwarn("It seems that these two are the same in these files?")
            span_info = span_info0['full_span']
            evt = sent.make_event(
                span_info['start_token'], span_info['end_token']-span_info['start_token']+1,
                type=v_evt['event_type'].replace(".", ":"))
            stat['evt'] += 1
            for v_arg in v_evt['arguments']:
                if len(v_arg['span_set']['spans']) > 1:
                    zwarn("More than one spans for arg!")
                stat['arg'] += 1
                evt.add_arg(ef_map[v_arg['span_set']['spans'][0]['grounded_span']['mention_id']], role=v_arg['role'])
        # --
        sents.append(sent)
    doc = Doc.create(sents, id=data['doc_id'])
    return doc

# --
# count the stat from the orig/split80 files
# https://github.com/PlusLabNLP/X-Gear/tree/main/preprocessing/Dataset/ace_2005_ar/*
"""
# {original_length,split_80}/{train,dev,test}.json
Counter({'ef': 16151, 'sent': 2723, 'arg': 2506, 'evt': 1743, 'doc': 317})
Counter({'ef': 1246, 'sent': 289, 'arg': 174, 'evt': 117, 'doc': 20})
Counter({'ef': 1517, 'arg': 287, 'sent': 272, 'evt': 198, 'doc': 32})
Counter({'ef': 16150, 'sent': 3219, 'arg': 2506, 'evt': 1743, 'doc': 317})
Counter({'ef': 1246, 'sent': 335, 'arg': 174, 'evt': 117, 'doc': 20})
Counter({'ef': 1517, 'sent': 313, 'arg': 287, 'evt': 198, 'doc': 32})
"""
def count_stat(file):
    import json
    from collections import Counter
    with open(file) as fd:
        dds = json.load(fd)
    cc = Counter()
    for dd in dds.values():
        cc['doc'] += 1
        for dd2 in dd['sentences']:
            cc['sent'] += 1
            cc['ef'] += len(dd2['mentions'])
            cc['evt'] += len(dd2['basic_events'])
            cc['arg'] += sum([len(z['arguments']) for z in dd2['basic_events']])
    print(cc)
def count_stat2(file):
    import json
    from collections import Counter
    with open(file) as fd:
        dds = [json.loads(line) for line in fd]
    cc = Counter()
    for dd in dds:
        cc['doc'] += 1
        cc['sent'] += len(dd['sentences'])
        cc['ef'] += sum(len(z) for z in dd['ner'])
        cc['evt'] += sum(len(z) for z in dd['events'])
        cc['arg'] += sum((len(z2)-1) for z in dd['events'] for z2 in z)
    print(cc)
# --

# --
def main(input_file: str, output_file: str, format='xgear'):
    cc = Counter()
    docs = []
    if format == 'xgear':
        for dd in default_json_serializer.load_list(input_file):
            one_doc = convert_xgear(dd, cc)
            docs.append(one_doc)
    elif format == 'ar':
        dds = default_json_serializer.from_file(input_file)
        for dd in dds.values():
            one_doc = convert_ar(dd, cc)
            docs.append(one_doc)
    else:
        raise NotImplementedError()
    zlog(f"Read from {input_file} to {output_file}: {cc}")
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(docs)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.data.prep_ace_arzhv4 input_file:?? output_file:??
if __name__ == '__main__':
    main(*sys.argv[1:])

# --
"""
# prepare data
git clone https://github.com/PlusLabNLP/X-Gear
cd X-Gear/preprocessing
conda env create -f ../environment.yml
conda activate xgear
git clone https://github.com/fe1ixxu/Gradual-Finetune
cp -r Gradual-Finetune/dygiepp/data/ace-event/processed-data/ Dataset/ace_2005_Xuetal
# note: need to cp ace05 ...
echo "cp -r ??/ace_2005_td_v7 ./Dataset/"
bash process_ace.sh
conda deactivate
cd ../..
# --
for cl in en ar zh; do
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.data.prep_ace_arzhv4 X-Gear/processed_data/ace05_${cl}_mT5/${wset}.json ${cl}.ace4.${wset}.json
done
done |& tee _log_ace4
for wset in train dev test; do
mv zh.ace4.${wset}.json zh.tmp_ace4.${wset}.json
done
# retokenize!
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.data2.retok_zh zh.tmp_ace4.${wset}.json zh.ace4.${wset}.json
done
for cl in en ar zh; do
for wset in train dev test; do
ff=${cl}.ace4.${wset}.json
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:${cl} stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:${ff} "output_path:${ff%.json}.ud2.json"
done
done
cp *.ace*.ud2.json ../../events/data/data21f/
# --
# orig data for AR
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.data.prep_ace_arzhv4 X-Gear/preprocessing/Dataset/ace_2005_ar/original_length/${wset}.json ar.ace4r.${wset}.json ar
done |& tee _log_ace4r
# Read from X-Gear/preprocessing/Dataset/ace_2005_ar/original_length/train.json to ar.ace4r.train.json: Counter({'ef': 16151, 'sent': 2723, 'arg': 2506, 'evt': 1743, 'doc': 317})
# Read from X-Gear/preprocessing/Dataset/ace_2005_ar/original_length/dev.json to ar.ace4r.dev.json: Counter({'ef': 1246, 'sent': 289, 'arg': 174, 'evt': 117, 'doc': 20})
# Read from X-Gear/preprocessing/Dataset/ace_2005_ar/original_length/test.json to ar.ace4r.test.json: Counter({'ef': 1517, 'arg': 287, 'sent': 272, 'evt': 198, 'doc': 32})
for cl in ar; do
for wset in train dev test; do
ff=${cl}.ace4r.${wset}.json
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:${cl} stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:${ff} "output_path:${ff%.json}.ud2.json"
done
done
cp ar.ace4r*.ud2.json ../../events/data/data21f/
"""
