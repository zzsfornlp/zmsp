#

# test

from collections import defaultdict
from msp2.utils import zlog, zwarn, init_everything, Timer
from msp2.data.inst import yield_sents
from msp2.data.vocab import WordVectors, SimpleVocab
from msp2.data.stream import IterStreamer
from msp2.nn.layers import EmbeddingNode
from msp2.nn import BK
from ..confs import OverallConf, build_model
from ..run import ZmtlTestingRunner, ZmtlVocabPackage, index_stream, batch_stream

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
    vpack = ZmtlVocabPackage.build_by_reading(dconf)
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
    model_emb = model.get_emb()
    if model_emb is not None:
        _embedder = model_emb.eg.get_embedder("word")
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
    # --
    if conf.tconf.test_do_oracle_batching:
        # note: special mode!!
        zlog("First decode to get oracle!")
        all_insts = []  # simply decode and get them
        grouped_insts = defaultdict(list)
        for insts in test_iter:
            model.predict_on_batch(insts)
            all_insts.extend(insts)
            for inst in insts:
                for frame in inst.events:
                    _key = frame.info["exit_lidx"]
                    grouped_insts[_key].append(frame)
        group_info = {k: len(grouped_insts[k]) for k in sorted(grouped_insts.keys())}
        zlog(f"group: {group_info}")
        # then feed them within groups
        rr = ZmtlTestingRunner(model, None, conf, dconf.output, dconf.test, do_score=dconf.test_do_score)
        rec = rr.test_recorder
        with Timer(info="Run-test", print_date=True):
            tconf = conf.tconf
            tconf.test_count_mode = "tok"  # note: here we already get the frames!!
            for frames in grouped_insts.values():
                stream, _ = batch_stream(IterStreamer(frames), tconf, False)
                for binsts in stream:
                    with rec.go():
                        res0 = rr._run_batch(binsts)
                    rec.record(res0)
            rr.all_insts = all_insts  # replace by sents!!
            res = rr._run_end()
    else:
        # go
        rr = ZmtlTestingRunner(model, test_iter, conf, dconf.output, dconf.test, do_score=dconf.test_do_score)
        res = rr.run()
    zlog(f"zzzfinal: {res}")
    zlog("The end of testing.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Testing", print_date=True) as et:
        main(sys.argv[1:])

# --
"""
CUDA_VISIBLE_DEVICES= PYTHONPATH=../src:../../src/:../../../src python3 -m msp2.tasks.zmtl.main.test _conf
"""
