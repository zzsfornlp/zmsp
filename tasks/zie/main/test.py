#

import msp
from msp import utils
from msp.utils import Helper, zlog, zwarn
from msp.data import MultiHelper, WordVectors, VocabBuilder

from ..common.confs import OverallConf, init_everything, build_model, get_berter
from ..common.data import get_data_reader, BerterDataAuger
from ..common.vocab import DConf, IEVocabPackage
from ..common.run import index_stream, batch_stream, MyIETestingRunner

#
def iter_hit_words(doc_streamer, emb):
    words = set()
    hit_count = 0
    #
    for inst in doc_streamer:
        for sent in inst.sents:
            for one in sent.words.vals:
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
    conf: OverallConf = init_everything(args, ConfType)
    dconf, mconf = conf.dconf, conf.mconf
    iconf = mconf.iconf
    # vocab
    vpack = IEVocabPackage.build_by_reading(conf)
    # prepare data
    test_streamer = get_data_reader(dconf.test, dconf.input_format, dconf.use_label0, dconf.noef_link0, dconf.aux_repr_test, max_evt_layers=dconf.max_evt_layers)
    # model
    model = build_model(conf.model_type, conf, vpack)
    if dconf.model_load_name != "":
        model.load(dconf.model_load_name)
    else:
        zwarn("No model to load, Debugging mode??")
    # =====
    # augment with extra embeddings
    extra_embed_files = dconf.test_extra_pretrain_files
    if len(extra_embed_files) > 0:
        # get embeddings
        extra_codes = []  # todo(note): ignore this mode for this project
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
    # use bert? todo(note): no pre-compute here in testing!
    if dconf.use_bert:
        bmodel = get_berter(dconf.bconf)
        test_streamer = BerterDataAuger(test_streamer, bmodel, "aux_repr")
    #
    # No Cache!!
    test_inst_preparer = model.get_inst_preper(False)
    test_iter = batch_stream(index_stream(test_streamer, vpack, False, False, test_inst_preparer), iconf, False)
    return conf, model, vpack, test_iter

#
def main(args):
    conf, model, vpack, test_iter = prepare_test(args)
    dconf = conf.dconf
    # go
    rr = MyIETestingRunner(model, vpack, dconf.output_file, dconf.test, dconf.output_format, dconf.eval_conf,
                           release_resources=True)
    x = rr.run(test_iter)
    utils.printing("The end.")
