#

from msp.utils import zlog, Conf, JsonRW, PickleRW, zopen
import json, glob

from .svocab import StatVocab
from .scorpus import DataReader
from .smodel import StatModel, StatApplyConf

#
class StatConf(Conf):
    def __init__(self, args):
        self.main_prefix = ""  # main_dir/main_name
        # step 1: build the overall vocab
        self.input_files = []  # can be glob
        self.lc = True
        self.thr_minlen = 5
        self.thr_maxlen = 80
        self.word_minc = 5  # word-count thresh
        self.word_softcut = 10000  # general vocab size (todo(note): should also decide on corpus size)
        self.vocab_prefix = "vocab"
        self.no_build_vocab = False  # directly load and skip building
        # step 2: build the model (feature and feature counts)
        self.feat_compact = True  # whether compact the feat to save space?
        self.win_size = 10  # max distance of center & context tokens
        self.bins = [1, 2, 4, 10]  # how to bin the distances
        self.neg_dist = True  # use negative distance
        self.binary_nb = 1  # whether in the mode of binary nb
        self.fc_max = 0  # cutting for the nearby word for Center (0 as off)
        self.ft_max = 0  # cutting for the nearby word for conText (0 as off)
        self.lex_min = 1  # only consider those >= this for tok (center)
        self.lex_min2 = 1  # only consider those >= this for tok2 (context)
        self.meet_punct_max = 1  # how many puncts as features (<=0 as off)
        self.meet_lex_thresh = 0  # only lex min <thresh
        self.meet_lex_freq_max = 0  # how many freq words met as features (0 as off)
        self.model_prefix = "model"
        self.no_build_model = False
        # step 3: apply the model
        # not at building time, see smodel.ApplyConf
        self.update_from_args(args)
        self.validate()

    def do_validate(self):
        # extend glob
        all_files = []
        for one in self.input_files:
            all_files.extend(glob.glob(one))
        self.input_files = sorted(all_files)
        # int bins
        self.bins = [int(z) for z in self.bins]

#
def load_model(file) -> StatModel:
    import pickle
    # return PickleRW.from_file(file)
    with zopen(file, 'rb') as fd:
        return pickle.load(fd)

def main(args):
    conf = StatConf(args)
    input_files = conf.input_files
    main_prefix = conf.main_prefix
    # step 1: get vocab
    zlog("Step 1: vocab")
    vocab_prefix = main_prefix + conf.vocab_prefix
    if conf.no_build_vocab:
        vocab = StatVocab()
        JsonRW.from_file(vocab, vocab_prefix+".json")
    else:
        reader0 = DataReader(conf)
        vocab_orig = StatVocab(name="word")
        for tokens in reader0.yield_data(input_files):
            vocab_orig.add_all(tokens)
        # cut and write vocab
        # first sort and write overall one
        vocab_orig.sort_and_cut(sort=True)
        vocab_orig.write_txt((main_prefix + vocab_prefix + "_orig.txt"))
        JsonRW.to_file(vocab_orig, (main_prefix + vocab_prefix + "_orig.json"))
        # also write msp.Vocab for analysis usage
        JsonRW.to_file(vocab_orig.to_zvoc(), (main_prefix + vocab_prefix + "_orig.voc"))
        #
        vocab_cut = vocab_orig.copy()
        vocab_cut.sort_and_cut(mincount=conf.word_minc, soft_cut=conf.word_softcut, sort=True)
        vocab_cut.write_txt((main_prefix + vocab_prefix + ".txt"))
        JsonRW.to_file(vocab_cut, (main_prefix + vocab_prefix + ".json"))
        vocab = vocab_cut
    # step 2: build corpus and raw counts
    # todo(note): next:
    #  look at which words (freq) are good/bad for cross-lingual results;
    #  collect features one pass from left to right: count(word), count(word, featured_word), count(f_word)
    #  how to normalize? and how to penalize according to features, in train or test?
    zlog("Step 2: model")
    model_prefix = main_prefix + conf.model_prefix
    if conf.no_build_model:
        model = PickleRW.from_file(model_prefix+".pic")
    else:
        reader1 = DataReader(conf)
        model = StatModel(conf, vocab)
        for tokens in reader1.yield_data(input_files):
            model.add_sent(tokens)
        #
        PickleRW.to_file(model, model_prefix+".pic")
    # step 3: apply model
    # todo(note): at using time ...
    zlog("Finish")
    return model

# PYTHONPATH=../src/ python3 ../src/tasks/cmd.py zdpar.stat.collect input_files:?
