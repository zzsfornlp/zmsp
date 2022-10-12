#

# col format to json

import os
import sys
from collections import OrderedDict, Counter
from msp2.utils import zopen, zlog
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, CharIndexer
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.data.vocab import SeqSchemeHelperStr

# --
def yield_block(stream):
    rets = []
    for line in stream:
        line = line.rstrip()
        if len(line) == 0:
            if len(rets) > 0:
                yield rets
                rets = []
        else:
            rets.append(line)
    if len(rets) > 0:
        yield rets
    # --

def main(input_file: str, output_file: str, scheme='BIO'):
    cc = Counter()
    sents = []
    _SEP = None
    helper = SeqSchemeHelperStr(scheme)
    with zopen(input_file) as fd:
        for lines in yield_block(fd):
            words = [z.split(_SEP)[0] for z in lines]
            tags = [z.split(_SEP)[-1] for z in lines]
            spans = helper.tags2spans(tags)
            sent = Sent.create(words)
            for widx, wlen, label in spans:
                sent.make_event(widx, wlen, type=label)  # note: make it as event!
                cc['item'] += 1
            cc['sent'] += 1
            cc['word'] += len(sent)
            sents.append(sent)
    # --
    zlog(f"Read from {input_file} to {output_file}: {cc}")
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(sents)
    # --

# --
# python3 -m msp2.tasks.zmtl3.mat.prep.col2json ...
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# --
# try it with NCRFpp
git clone https://github.com/jiesutd/NCRFpp
cd NCRFpp
mv sample_data sample_data0
mkdir -p sample_data
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.mat.prep.json2col ../_split_v0307/mat.v0307.0.${wset}.json sample_data/${wset}.bmes BIO " "
done
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
# unzip -q -p crawl-300d-2M.vec.zip | tail -n +2 >sample_data/sample.word.emb
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip -p glove.6B.zip 'glove.6B.50d.txt' >sample_data/sample.word.emb
# modify config files...
python main.py --config demo.train.config |& tee _log
# ok, not good, let's just find a bert-ner ...
# --
wget https://raw.githubusercontent.com/huggingface/transformers/v3.1.0/examples/token-classification/run_ner.py
wget https://raw.githubusercontent.com/huggingface/transformers/v3.1.0/examples/token-classification/scripts/preprocess.py
wget https://raw.githubusercontent.com/huggingface/transformers/v3.1.0/examples/token-classification/utils_ner.py
wget https://raw.githubusercontent.com/huggingface/transformers/v3.1.0/examples/token-classification/tasks.py
MAX_LENGTH=128
BERT_MODEL=roberta-base
for wset in train dev test; do
python3 preprocess.py sample_data/${wset}.bmes $BERT_MODEL $MAX_LENGTH > ${wset}.txt
done
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
CUDA_VISIBLE_DEVICES=0 python3 run_ner.py --data_dir ./ \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir output \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs 100 \
--per_device_train_batch_size 32 \
--save_steps 50 \
--seed 1 \
--do_train \
--do_eval \
--do_predict
# => {'eval_loss': 2.2005312740802765, 'eval_accuracy_score': 0.6572164948453608, 'eval_precision': 0.215, 'eval_recall': 0.2792207792207792, 'eval_f1': 0.24293785310734461, 'epoch': 100.0, 'step': 400}
# --
# try it with conll03
for wset in train valid test; do
wget https://raw.githubusercontent.com/ningshixian/NER-CONLL2003/master/data/${wset}.txt
python3 -m msp2.tasks.zmtl3.mat.prep.col2json ${wset}.txt ${wset}.json
done
ln -s valid.json dev.json
Read from train.txt to train.json: Counter({'word': 204567, 'item': 23499, 'sent': 14987})
Read from valid.txt to valid.json: Counter({'word': 51578, 'item': 5942, 'sent': 3466})
Read from test.txt to test.json: Counter({'word': 46666, 'item': 5648, 'sent': 3684})
python3 -m msp2.tasks.zmtl3.main.train 'conf_sbase:task:mat' train0.input_dir:./ dev0.input_dir:./ test0.input_dir:./ train0.group_files:train.json dev0.group_files:dev.json test0.group_files:dev.json,test.json matM.crf:yes train0.batch_size:1024 lrate.val:0.00002 device:0 |& tee _log
# => seems ok, after 1 ckp: dev=0.9550, test=0.9183; and final (20ckp): dev=0.9634, test=0.9210
"""
