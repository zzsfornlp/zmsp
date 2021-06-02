#

# to build vocab: first step of training

#
from collections import Counter
from msp import utils
from msp.data import MultiCatStreamer, InstCacher

from ..run.confs import OverallConf, init_everything, build_model
from ..run.run import get_data_reader, PreprocessStreamer, index_stream, batch_stream, MltTrainingRunner
from ..run.vocab import MLMVocabPackage

#
def print_vocab_txt(vocab, dev_counter=None):
    lines = []
    lines.append(f"# Details of Vocab {vocab.name}")
    if dev_counter is None:
        dev_counter = {}
    # iterate through vocab
    all_count_voc = sum(z for z in vocab.final_vals if z is not None)
    all_count_dev = sum(dev_counter.values())
    all_hit_dev = sum(1 for z in dev_counter.values() if z>0)
    accu_count_voc, accu_count_dev = 0, 0
    accu_hit_dev_set = set()
    for idx, key in enumerate(vocab.final_words):
        # in vocab
        count_voc = vocab.final_vals[idx]
        if count_voc is None:
            count_voc = 0
        accu_count_voc += count_voc
        # in dev
        count_dev = dev_counter.get(key, 0)
        accu_count_dev += count_dev
        if count_dev > 0:
            accu_hit_dev_set.add(key)
        # -----
        added_line = [
            str(idx), key,
            f"{count_voc}({count_voc/all_count_voc:.4f})", f"{accu_count_voc}({accu_count_voc/all_count_voc:.4f})",
            f"{count_dev}({count_dev/all_count_dev:.4f})", f"{accu_count_dev}({accu_count_dev/all_count_dev:.4f})",
        ]
        lines.append("\t".join(added_line))
    lines.append(f"Coverage: type={len(accu_hit_dev_set)}/{all_hit_dev}={len(accu_hit_dev_set)/max(1, all_hit_dev)}, "
                 f"count={accu_count_dev}/{all_count_dev}={accu_count_dev/max(1, all_count_dev)}")
    unhit_dev_counter = Counter({k:v for k,v in dev_counter.items() if k not in accu_hit_dev_set})
    lines.append(f"Unhit dev top-50: {unhit_dev_counter.most_common(50)}")
    return "\n".join(lines)

# -----
def main(args):
    conf = init_everything(args)
    dconf, mconf = conf.dconf, conf.mconf
    # =====
    if dconf.train:  # build
        train_streamer = PreprocessStreamer(get_data_reader(dconf.train, dconf.input_format),
                                            lower_case=dconf.lower_case, norm_digit=dconf.norm_digit)
        vpack = MLMVocabPackage.build_from_stream(dconf.vconf, train_streamer, [])
        vpack.save(dconf.dict_dir)
    else:  # read
        vpack = MLMVocabPackage.build_by_reading(dconf.dict_dir)
    # check dev
    if dconf.dev:
        dev_streamer = PreprocessStreamer(get_data_reader(dconf.dev, dconf.dev_input_format),
                                          lower_case=dconf.lower_case, norm_digit=dconf.norm_digit)
        dev_counter = Counter()
        for inst in dev_streamer:
            for w in inst.word_seq.vals:
                dev_counter[w] += 1
        print(print_vocab_txt(vpack.get_voc("word"), dev_counter))

# examples
"""
SRC_DIR=../../../src/
PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zmlm.main.vocab_utils train:../wikis/wiki_en.100k.txt dev:../ud24s/en_train.-1.conllu input_format:plain dev_input_format:conllu norm_digit:1 >vv.list
"""
