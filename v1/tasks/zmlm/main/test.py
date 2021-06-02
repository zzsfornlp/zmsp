#

import msp
from msp import utils
from msp.utils import Helper, zlog, zwarn
from msp.data import MultiHelper, WordVectors, VocabBuilder

from ..run.confs import OverallConf, init_everything, build_model
from ..run.run import get_data_reader, PreprocessStreamer, index_stream, batch_stream, MltTrainingRunner, MltTestingRunner
from ..run.vocab import MLMVocabPackage

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

# simply use an external one as here
def aug_words_and_embs(model, aug_vocab, aug_wv):
    orig_vocab = model.word_vocab
    word_emb_node = model.embedder.get_node('word')
    if word_emb_node is not None:
        orig_arr = word_emb_node.E.detach().cpu().numpy()
        # todo(+2): find same-spelling words in the original vocab if not-hit in the extra_embed?
        # todo(warn): here aug_vocab should be find in aug_wv
        aug_arr = aug_vocab.filter_embed(aug_wv, assert_all_hit=True)
        new_vocab, new_arr = MultiHelper.aug_vocab_and_arr(orig_vocab, orig_arr, aug_vocab, aug_arr, aug_override=True)
        # assign
        model.word_vocab = new_vocab  # there should be more to replace? but since only testing maybe no need...
        word_emb_node.replace_weights(new_arr)
    else:
        zwarn("No need to aug vocab since delexicalized model!!")
        new_vocab = orig_vocab
    return new_vocab

#
def prepare_test(args, ConfType=None):
    # conf
    conf = init_everything(args, ConfType)
    dconf, mconf = conf.dconf, conf.mconf
    # vocab
    vpack = MLMVocabPackage.build_by_reading(dconf.dict_dir)
    # prepare data
    test_streamer = PreprocessStreamer(get_data_reader(dconf.test, dconf.input_format),
                                       lower_case=dconf.lower_case, norm_digit=dconf.norm_digit)
    # model
    model = build_model(conf, vpack)
    if dconf.model_load_name != "":
        model.load(dconf.model_load_name)
    else:
        zwarn("No model to load, Debugging mode??")
    # -----
    # augment with extra embeddings for test stream?
    extra_embed_files = dconf.vconf.test_extra_pretrain_files
    if len(extra_embed_files) > 0:
        # get embeddings
        extra_codes = dconf.vconf.test_extra_pretrain_codes
        if len(extra_codes) == 0:
            extra_codes = [""] * len(extra_embed_files)
        extra_embedding = WordVectors.load(extra_embed_files[0], aug_code=extra_codes[0])
        extra_embedding.merge_others([WordVectors.load(one_file, aug_code=one_code) for one_file, one_code in
                                      zip(extra_embed_files[1:], extra_codes[1:])])
        # get extra dictionary (only those words hit in extra-embed)
        extra_vocab = VocabBuilder.build_from_stream(iter_hit_words(test_streamer, extra_embedding),
                                                     sort_by_count=True, pre_list=(), post_list=())
        # give them to the model
        new_vocab = aug_words_and_embs(model, extra_vocab, extra_embedding)
        vpack.put_voc("word", new_vocab)
    # =====
    # No Cache!!
    test_inst_preparer = model.get_inst_preper(False)
    backoff_pos_idx = dconf.backoff_pos_idx
    test_iter = batch_stream(index_stream(test_streamer, vpack, False, False, test_inst_preparer, backoff_pos_idx),
                             mconf.test_batch_size, mconf, False)
    return conf, model, vpack, test_iter

#
def main(args):
    conf, model, vpack, test_iter = prepare_test(args)
    dconf = conf.dconf
    # go
    rr = MltTestingRunner(model, vpack, dconf.output_file, dconf.test, dconf.output_format)
    x = rr.run(test_iter)
    utils.printing("The end.")

# b tasks/zmlm/main/test.py:46
