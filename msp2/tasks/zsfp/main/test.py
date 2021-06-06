#

# test

from msp2.utils import zlog, zwarn, init_everything, Timer
from msp2.data.inst import yield_sents
from msp2.data.vocab import WordVectors, SimpleVocab
from msp2.nn.layers import EmbeddingNode
from msp2.nn import BK
from ..confs import OverallConf, build_model
from ..run import ZsfpTestingRunner, ZsfpVocabPackage, index_stream, batch_stream

# =====
# some helpers

# special operation: get extra and hit words!
def get_extra_hit_words(input_stream, emb, voc):
    word_counts = {}
    hit_words = set()
    # --
    for sent in yield_sents(input_stream):
        for one in sent.seq_word.vals:
            word_counts[one] = word_counts.get(one, 0) + 1
            if emb.find_key(one) is not None:
                hit_words.add(one)
    # --
    extra_hit_words = [z for z in hit_words if z not in voc]
    extra_hit_words.sort(key=lambda x: -word_counts[x])
    zlog(f"Iter hit words: all={len(word_counts)}, hit={len(hit_words)}, extra_hit={len(extra_hit_words)}")
    return extra_hit_words

# todo(note): special operation!
def aug_words_and_embs(emb_node: EmbeddingNode, orig_vocab: SimpleVocab,
                       aug_vocab: SimpleVocab, aug_wv: WordVectors, aug_scale: float=1.0):
    orig_arr = emb_node.E.E.detach().cpu().numpy()
    # todo(+2): find same-spelling words in the original vocab if not-hit in the extra_embed?
    # todo(warn): here aug_vocab should be find in aug_wv
    aug_arr = aug_vocab.filter_embed(aug_wv, scale=aug_scale, assert_all_hit=True)
    new_vocab, new_arr = SimpleVocab.aug_vocab_and_arr(orig_vocab, orig_arr, aug_vocab, aug_arr, aug_override=True)
    # assign
    BK.set_value(emb_node.E.E, new_arr, resize=True)
    return new_vocab

# =====
# common prepare
def prepare_test(args):
    conf: OverallConf = init_everything(OverallConf(), args)
    dconf, tconf = conf.dconf, conf.tconf
    # vocab
    vpack = ZsfpVocabPackage.build_by_reading(dconf)
    # prepare data
    test_streamer = dconf.R.get_reader(input_path=dconf.test)
    # model
    model = build_model(conf, vpack=vpack)
    if dconf.model_load_name != "":
        model.load(dconf.model_load_name, strict=dconf.model_load_strict)
    else:
        zwarn("No model to load, Debugging mode??")
    # =====
    # augment with extra embeddings
    extra_embed_files = dconf.test_extra_pretrain_wv_files
    if model.emb is not None:
        _embedder = model.emb.eg.get_embedder("word")
        if len(extra_embed_files) > 0 and _embedder is not None:  # has extra_emb and need_emb
            # get embeddings
            extra_embedding = WordVectors.load(extra_embed_files[0])
            extra_embedding.merge_others([WordVectors.load(one_file) for one_file in extra_embed_files[1:]])
            # get extra dictionary (only those words hit in extra-embed)
            extra_vocab = SimpleVocab.build_by_static(
                get_extra_hit_words(test_streamer, extra_embedding, vpack.get_voc("word")), pre_list=None, post_list=None)
            # give them to the model
            new_vocab = aug_words_and_embs(_embedder, vpack.get_voc("word"),
                                           extra_vocab, extra_embedding, aug_scale=dconf.pretrain_scale)
            vpack.put_voc("word", new_vocab)
    # =====
    # No Cache!!
    test_inst_preparer = model.get_inst_preper(False)
    test_iter, _ = batch_stream(index_stream(test_streamer, vpack, False, False, test_inst_preparer), tconf, False)
    return conf, model, vpack, test_iter

# -----
def main(args):
    conf, model, vpack, test_iter = prepare_test(args)
    dconf = conf.dconf
    # go
    rr = ZsfpTestingRunner(model, test_iter, conf, dconf.output, dconf.test)
    res = rr.run()
    zlog(f"zzzfinal: {res}")
    zlog("The end of testing.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Testing", print_date=True) as et:
        main(sys.argv[1:])
