#

import msp
from msp import utils
from msp.utils import Helper
from msp.data import MultiHelper, WordVectors, VocabBuilder

from ..common.confs import DepParserConf, init_everything, build_model
from ..common.data import get_data_reader
from ..common.vocab import DConf, ParserVocabPackage
from ..common.run import index_stream, batch_stream, ParserTestingRunner

#
def iter_hit_words(parse_streamer, emb):
    for inst in parse_streamer:
        for one in inst.words.vals:
            if emb.has_key(one):
                yield one

#
def main(args):
    # conf
    conf = init_everything(args)
    dconf, pconf = conf.dconf, conf.pconf
    iconf = pconf.iconf
    # vocab
    vpack = ParserVocabPackage.build_by_reading(dconf)
    # prepare data
    test_streamer = get_data_reader(dconf.test, dconf.input_format, dconf.code_test, dconf.use_label0)
    # model
    model = build_model(conf.partype, conf, vpack)
    model.load(dconf.model_load_name)
    # =====
    # augment with extra embeddings
    extra_embed_files = dconf.test_extra_pretrain_files
    if len(extra_embed_files) > 0:
        # get embeddings
        extra_codes = dconf.test_extra_pretrain_codes
        if len(extra_codes) == 0:
            extra_codes = [""] * len(extra_embed_files)
        extra_embedding = WordVectors.load(extra_embed_files[0], aug_code=extra_codes[0])
        extra_embedding.merge_others([WordVectors.load(one_file, aug_code=one_code) for one_file, one_code in zip(extra_embed_files[1:], extra_codes[1:])])
        # get extra dictionary (only those words hit in extra-embed)
        extra_vocab = VocabBuilder.build_from_stream(iter_hit_words(test_streamer, extra_embedding), sort_by_count=True)
        # give them to the model
        new_vocab = model.aug_words_and_embs(extra_vocab, extra_embedding)
        vpack.put_voc("word", new_vocab)
    # =====
    # No Cache!!
    test_iter = batch_stream(index_stream(test_streamer, vpack, False), iconf, False)
    # go
    rr = ParserTestingRunner(model, vpack, dconf.output_file, dconf.test, dconf.output_format)
    x = rr.run(test_iter)
    utils.printing("The end.")

# b tasks/zdpar/main/test.py:46
