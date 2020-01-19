#

from typing import Dict

from msp.utils import Conf, zlog
from msp.data import VocabPackage, VocabBuilder, WordNormer, WordVectors, VocabHelper

from .confs import DConf

#
class ParserVocabPackage(VocabPackage):
    def __init__(self, vocabs: Dict, embeds: Dict, dconf: DConf):
        super().__init__(vocabs, embeds)
        #
        self.word_normer = WordNormer(lower_case=dconf.lower_case, norm_digit=dconf.norm_digit)

    @staticmethod
    def build_by_reading(dconf):
        zlog("Load vocabs from files.")
        possible_vocabs = ["word", "char", "pos", "label"]
        one = ParserVocabPackage({n:None for n in possible_vocabs}, {n:None for n in possible_vocabs}, dconf)
        one.load(dconf.dict_dir)
        return one

    @staticmethod
    def build_from_stream(dconf: DConf, stream, extra_stream):
        zlog("Build vocabs from streams.")
        ret = ParserVocabPackage({}, {}, dconf)
        #
        word_builder = VocabBuilder("word")
        char_builder = VocabBuilder("char")
        pos_builder = VocabBuilder("pos")
        label_builder = VocabBuilder("label")
        word_normer = ret.word_normer
        for inst in stream:
            # todo(warn): only do special handling for words
            word_builder.feed_stream(word_normer.norm_stream(inst.words.vals))
            for w in inst.words.vals:
                char_builder.feed_stream(w)
            pos_builder.feed_stream(inst.poses.vals)
            label_builder.feed_stream(inst.labels.vals)
        #
        w2vec = None
        if dconf.init_from_pretrain:
            # todo(warn): for convenience, extra vocab (usually dev&test) is only used for collecting pre-train vecs
            extra_word_set = set()
            for inst in extra_stream:
                for w in word_normer.norm_stream(inst.words.vals):
                    extra_word_set.add(w)
            # ----- load (possibly multiple) pretrain embeddings
            # must provide dconf.pretrain_file (there can be multiple pretrain files!)
            list_pretrain_file, list_code_pretrain = dconf.pretrain_file, dconf.code_pretrain
            list_code_pretrain.extend([""] * len(list_pretrain_file))  # pad default ones
            w2vec = WordVectors.load(list_pretrain_file[0], aug_code=list_code_pretrain[0])
            if len(list_pretrain_file) > 1:
                w2vec.merge_others([WordVectors.load(list_pretrain_file[i], aug_code=list_code_pretrain[i])
                                    for i in range(1, len(list_pretrain_file))])
            # -----
            # first filter according to thresholds
            word_builder.filter(lambda ww, rank, val: (val >= dconf.word_fthres and rank <= dconf.word_rthres) or w2vec.has_key(ww))
            # then add extra ones
            for w in extra_word_set:
                if w2vec.has_key(w) and (not word_builder.has_key_currently(w)):
                    word_builder.feed_one(w)
            word_vocab = word_builder.finish()
            word_embed1 = word_vocab.filter_embed(w2vec, init_nohit=dconf.pretrain_init_nohit, scale=dconf.pretrain_scale)
        else:
            word_vocab = word_builder.finish_thresh(rthres=dconf.word_rthres, fthres=dconf.word_fthres)
            word_embed1 = None
        #
        char_vocab = char_builder.finish()
        # todo(+1): extra pos/label symbols?
        TARGET_END = VocabHelper.convert_special_pattern("unk")
        pos_vocab = pos_builder.finish(target_range=(1, TARGET_END))       # only real tags
        label_vocab = label_builder.finish(target_range=(1, TARGET_END))
        # assign
        ret.put_voc("word", word_vocab)
        ret.put_voc("char", char_vocab)
        ret.put_voc("pos", pos_vocab)
        ret.put_voc("label", label_vocab)
        ret.put_emb("word", word_embed1)
        #
        return ret
