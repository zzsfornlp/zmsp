#

# the vocab package

from typing import Dict

from msp.utils import Conf, zlog, zwarn
from msp.data import VocabPackage, VocabBuilder, WordVectors, VocabHelper

# -----
# UD2 pre-defined types
UD2_POS_LIST = ["NOUN", "PUNCT", "VERB", "PRON", "ADP", "DET", "PROPN", "ADJ", "AUX", "ADV", "CCONJ", "PART", "NUM", "SCONJ", "X", "INTJ", "SYM"]
UD2_LABEL_LIST = ["punct", "case", "nsubj", "det", "root", "nmod", "advmod", "obj", "obl", "amod", "compound", "aux", "conj", "mark", "cc", "cop", "advcl", "acl", "xcomp", "nummod", "ccomp", "appos", "flat", "parataxis", "discourse", "expl", "fixed", "list", "iobj", "csubj", "goeswith", "vocative", "reparandum", "orphan", "dep", "dislocated", "clf"]
# for POS UNK backoff
UD2_POS_UNK_MAP = {p: VocabHelper.convert_special_pattern("UNK_"+p) for p in UD2_POS_LIST}
# -----

#
class MLMVocabPackageConf(Conf):
    def __init__(self):
        self.add_ud2_prevalues = True  # add for UD2 pos and labels
        self.add_ud2_pos_backoffs = False
        # pretrain
        self.pretrain_file = []
        self.read_from_pretrain = False
        self.pretrain_scale = 1.0
        self.pretrain_init_nohit = 1.0
        self.pretrain_codes = []
        self.test_extra_pretrain_files = []  # extra embeddings for testing
        self.test_extra_pretrain_codes = []
        # thresholds for word
        self.word_rthres = 1000000  # rank <= this
        self.word_fthres = 1  # freq >= this
        self.ignore_thresh_with_pretrain = True  # if hit in pretrain, also include and ignore thresh

#
class MLMVocabPackage(VocabPackage):
    def __init__(self, vocabs: Dict, embeds: Dict):
        super().__init__(vocabs, embeds)
        # -----

    @staticmethod
    def build_by_reading(prefix):
        zlog("Load vocabs from files.")
        possible_vocabs = ["word", "char", "pos", "deplabel", "ner", "word2"]
        one = MLMVocabPackage({n:None for n in possible_vocabs}, {n:None for n in possible_vocabs})
        one.load(prefix=prefix)
        return one

    @staticmethod
    def build_from_stream(build_conf: MLMVocabPackageConf, stream, extra_stream):
        zlog("Build vocabs from streams.")
        ret = MLMVocabPackage({}, {})
        # -----
        if build_conf.add_ud2_pos_backoffs:
            ud2_pos_pre_list = list(VocabBuilder.DEFAULT_PRE_LIST) + [UD2_POS_UNK_MAP[p] for p in UD2_POS_LIST]
            word_builder = VocabBuilder("word", pre_list=ud2_pos_pre_list)
        else:
            word_builder = VocabBuilder("word")
        char_builder = VocabBuilder("char")
        pos_builder = VocabBuilder("pos")
        deplabel_builder = VocabBuilder("deplabel")
        ner_builder = VocabBuilder("ner")
        if build_conf.add_ud2_prevalues:
            zlog(f"Add pre-defined UD2 values for upos({len(UD2_POS_LIST)}) and ulabel({len(UD2_LABEL_LIST)}).")
            pos_builder.feed_stream(UD2_POS_LIST)
            deplabel_builder.feed_stream(UD2_LABEL_LIST)
        for inst in stream:
            word_builder.feed_stream(inst.word_seq.vals)
            for w in inst.word_seq.vals:
                char_builder.feed_stream(w)
            # todo(+N): currently we are assuming that we are using UD pos/deps, and directly go with the default ones
            # pos and label can be optional??
            # if inst.poses.has_vals():
            #     pos_builder.feed_stream(inst.poses.vals)
            # if inst.deplabels.has_vals():
            #     deplabel_builder.feed_stream(inst.deplabels.vals)
            if hasattr(inst, "ner_seq") and inst.ner_seq.has_vals():
                ner_builder.feed_stream(inst.ner_seq.vals)
        # ===== embeddings
        w2vec = None
        if build_conf.read_from_pretrain:
            # todo(warn): for convenience, extra vocab (usually dev&test) is only used for collecting pre-train vecs
            extra_word_set = set(w for inst in extra_stream for w in inst.word_seq.vals)
            # ----- load (possibly multiple) pretrain embeddings
            # must provide build_conf.pretrain_file (there can be multiple pretrain files!)
            list_pretrain_file, list_code_pretrain = build_conf.pretrain_file, build_conf.pretrain_codes
            list_code_pretrain.extend([""] * len(list_pretrain_file))  # pad default ones
            w2vec = WordVectors.load(list_pretrain_file[0], aug_code=list_code_pretrain[0])
            if len(list_pretrain_file) > 1:
                w2vec.merge_others([WordVectors.load(list_pretrain_file[i], aug_code=list_code_pretrain[i])
                                    for i in range(1, len(list_pretrain_file))])
            # -----
            # first filter according to thresholds
            word_builder.filter(
                lambda ww, rank, val: (val >= build_conf.word_fthres and rank <= build_conf.word_rthres) or
                                      (build_conf.ignore_thresh_with_pretrain and w2vec.has_key(ww)))
            # then add extra ones
            if build_conf.ignore_thresh_with_pretrain:
                for w in extra_word_set:
                    if w2vec.has_key(w) and (not word_builder.has_key_currently(w)):
                        word_builder.feed_one(w)
            word_vocab = word_builder.finish()
            word_embed1 = word_vocab.filter_embed(w2vec, init_nohit=build_conf.pretrain_init_nohit,
                                                  scale=build_conf.pretrain_scale)
        else:
            word_vocab = word_builder.finish_thresh(rthres=build_conf.word_rthres, fthres=build_conf.word_fthres)
            word_embed1 = None
        #
        char_vocab = char_builder.finish()
        pos_vocab = pos_builder.finish(sort_by_count=False)
        deplabel_vocab = deplabel_builder.finish(sort_by_count=False)
        ner_vocab = ner_builder.finish()
        # assign
        ret.put_voc("word", word_vocab)
        ret.put_voc("char", char_vocab)
        ret.put_voc("pos", pos_vocab)
        ret.put_voc("deplabel", deplabel_vocab)
        ret.put_voc("ner", ner_vocab)
        ret.put_emb("word", word_embed1)
        #
        return ret

    # ======
    # special mode extend another word vocab
    # todo(+N): ugly patch!!
    def aug_word2_vocab(self, stream, extra_stream, extra_embed_file: str):
        zlog(f"Aug another word vocab from streams and extra_embed_file={extra_embed_file}")
        word_builder = VocabBuilder("word2")
        for inst in stream:
            word_builder.feed_stream(inst.word_seq.vals)
        # embeddings
        if len(extra_embed_file) > 0:
            extra_word_set = set(w for inst in extra_stream for w in inst.word_seq.vals)
            w2vec = WordVectors.load(extra_embed_file)
            for w in extra_word_set:
                if w2vec.has_key(w) and (not word_builder.has_key_currently(w)):
                    word_builder.feed_one(w)
            word_vocab = word_builder.finish()  # no filtering!!
            word_embed1 = word_vocab.filter_embed(w2vec, init_nohit=1.0, scale=1.0)
        else:
            zwarn("WARNING: No pretrain file for aug node!!")
            word_vocab = word_builder.finish()  # no filtering!!
            word_embed1 = None
        self.put_voc("word2", word_vocab)
        self.put_emb("word2", word_embed1)
    # =====

# b tasks/zmlm/run/vocab:146
