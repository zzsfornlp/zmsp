#

import msp
from msp import utils
from msp.utils import Helper, zlog, zwarn
from msp.data import MultiHelper, WordVectors, VocabBuilder

from ..common.confs import DepParserConf, init_everything, build_model
from ..common.data import get_data_reader
from ..common.vocab import DConf, ParserVocabPackage
from ..common.run import index_stream, batch_stream, ParserTestingRunner

#
def iter_hit_words(parse_streamer, emb):
    words = set()
    hit_count = 0
    #
    for inst in parse_streamer:
        for one in inst.words.vals:
            # todo(note): only those hit in the embeddings are added
            if emb.has_key(one):
                yield one
                # count how many hit in embeddings
                if one not in words:
                    hit_count += 1
            words.add(one)
    zlog(f"Iter hit words: all={len(words)}, hit={hit_count}")

#
def prepare_test(args, ConfType=None):
    # conf
    conf = init_everything(args, ConfType)
    dconf, pconf = conf.dconf, conf.pconf
    iconf = pconf.iconf
    # vocab
    vpack = ParserVocabPackage.build_by_reading(dconf)
    # prepare data
    test_streamer = get_data_reader(dconf.test, dconf.input_format, dconf.code_test, dconf.use_label0,
                                    dconf.aux_repr_test, dconf.aux_score_test)
    # model
    model = build_model(conf.partype, conf, vpack)
    if dconf.model_load_name != "":
        model.load(dconf.model_load_name)
    else:
        zwarn("No model to load, Debugging mode??")
    # =====
    # augment with extra embeddings
    extra_embed_files = dconf.test_extra_pretrain_files
    if len(extra_embed_files) > 0:
        # get embeddings
        extra_codes = dconf.test_extra_pretrain_codes
        if len(extra_codes) == 0:
            extra_codes = [""] * len(extra_embed_files)
        extra_embedding = WordVectors.load(extra_embed_files[0], aug_code=extra_codes[0])
        extra_embedding.merge_others([WordVectors.load(one_file, aug_code=one_code) for one_file, one_code in
                                      zip(extra_embed_files[1:], extra_codes[1:])])
        # get extra dictionary (only those words hit in extra-embed)
        extra_vocab = VocabBuilder.build_from_stream(iter_hit_words(test_streamer, extra_embedding),
                                                     sort_by_count=True, pre_list=(), post_list=())
        # give them to the model
        new_vocab = model.aug_words_and_embs(extra_vocab, extra_embedding)
        vpack.put_voc("word", new_vocab)
    # =====
    # No Cache!!
    test_inst_preparer = model.get_inst_preper(False)
    test_iter = batch_stream(index_stream(test_streamer, vpack, False, False, test_inst_preparer), iconf, False)
    return conf, model, vpack, test_iter

#
def main(args):
    conf, model, vpack, test_iter = prepare_test(args)
    dconf = conf.dconf
    # go
    rr = ParserTestingRunner(model, vpack, dconf.output_file, dconf.test, dconf.output_format)
    x = rr.run(test_iter)
    utils.printing("The end.")

# b tasks/zdpar/main/test.py:46

# explanations for aug_code and multi-lang
"""
This shares similar function to scripts/dpar/multi/merge_data_and_emb.py, both can add lang-code to words:
-Option 1: with `merge_data_and_emb.py`, the tokens in both embeddings and conllu files can get aug_code previously, thus no need to add again here.
-Option 2: no pre-processing and add here with dconf options.
# the backoff to src (for eg., English) feature should be done in pre-processing!
-Option 3: no lang-code, let new embeddings override old src embeddings.
##
But all these options have problems that no way to indicate what embeddings to override (for eg., share punct and NE even if they appear to target embeddings). But I guess this goes too far and we have to believe the mapped CL embeds!
"""

# extra notes about embeddings
"""
1. training
For training, the vocab is created by all the words in Training set and possible extra words in dev/test that are hit in pretrained-embeddings. The embeddings are adopted from pre-trained if hit, otherwise init randomly (or 0).
// WARN: notice that speicial tokens like UNK,EOS,ROOT are usually not found in embeddings!
=> dconf.pretrain_init_nohit, dconf.pretrain_file, ...
2. testing 
// (currently 19.03.07, prefer using the Option 3 above since no need to use special data)
For testing, can plus augment embeddings and vocabs.
=> dconf.test_extra_pretrain_files, dconf.code_*, ...
"""
