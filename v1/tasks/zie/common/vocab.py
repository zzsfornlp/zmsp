#

from typing import Dict
from collections import defaultdict

from msp.utils import Conf, zlog, PickleRW
from msp.data import VocabPackage, VocabBuilder, WordVectors, VocabHelper

from msp.zext.ie import HLabelVocab

from .confs import DConf, OverallConf
from .data import DocInstance

#
class IEVocabPackage(VocabPackage):
    def __init__(self, vocabs: Dict, embeds: Dict, dconf: DConf):
        super().__init__(vocabs, embeds)

    # =====
    # why not save them all at one place? (harder to view?)
    def load(self, prefix="./"):
        fname = prefix + "zvocab.pkl"
        self.vocabs, self.embeds = PickleRW.from_file(fname)

    def save(self, prefix="./"):
        fname = prefix + "zvocab.pkl"
        PickleRW.to_file([self.vocabs, self.embeds], fname)

    # =====
    # loading or building

    @staticmethod
    def build_by_reading(conf: OverallConf):
        dconf = conf.dconf
        zlog("Load vocabs from files.")
        one = IEVocabPackage({}, {}, dconf)
        one.load(dconf.dict_dir)
        return one

    @staticmethod
    def build_from_stream(conf: OverallConf, stream, extra_stream):
        dconf = conf.dconf
        zlog("Build vocabs from streams.")
        ret = IEVocabPackage({}, {}, dconf)
        # here, collect them all
        # -- basic inputs
        word_builder = VocabBuilder("word")
        char_builder = VocabBuilder("char")
        lemma_builder = VocabBuilder("lemma")
        upos_builder = VocabBuilder("upos")
        ulabel_builder = VocabBuilder("ulabel")
        # -- outputs (event type, entity/filler type, arg role type) (type -> count)
        evt_type_builder = defaultdict(int)
        ef_type_builder = defaultdict(int)
        arg_role_builder = defaultdict(int)
        for inst in stream:
            # -- basic inputs
            for sent in inst.sents:
                word_builder.feed_stream(sent.words.vals)
                for w in sent.words.vals:
                    char_builder.feed_stream(w)
                lemma_builder.feed_stream(sent.lemmas.vals)
                upos_builder.feed_stream(sent.uposes.vals)
                ulabel_builder.feed_stream(sent.ud_labels.vals)
            # -- outputs
            # assert inst.entity_fillers is not None, "For building vocabs, need to provide training instances!"
            assert inst.events is not None, "For building vocabs, need to provide training instances!"
            if inst.entity_fillers is not None:
                for one_ef in inst.entity_fillers:
                    ef_type_builder[one_ef.type] += 1
            for one_evt in inst.events:
                evt_type_builder[one_evt.type] += 1
                if one_evt.links is not None:
                    for one_arg in one_evt.links:
                        arg_role_builder[one_arg.role] += 1
        # build real hlabel-types
        hl_evt = HLabelVocab("event", conf.mconf.hl_evt, evt_type_builder)
        hl_ef = HLabelVocab("entity_filler", conf.mconf.hl_ef, ef_type_builder)
        hl_arg = HLabelVocab("arg", conf.mconf.hl_arg, arg_role_builder)
        # deal with pre-trained word embeddings
        w2vec = None
        if dconf.init_from_pretrain:
            # todo(warn): for convenience, extra vocab (usually dev&test) is only used for collecting pre-train vecs
            # collect extra words and lemmas
            extra_word_set = set()
            extra_lemma_set = set()
            for inst in extra_stream:
                for sent in inst.sents:
                    for w in sent.words.vals:
                        extra_word_set.add(w)
                    for w in sent.lemmas.vals:
                        extra_lemma_set.add(w)
            # must provide dconf.pretrain_file
            w2vec = WordVectors.load(dconf.pretrain_file)
            # first filter according to thresholds
            word_builder.filter(lambda ww, rank, val: (val >= dconf.word_fthres and rank <= dconf.word_rthres) or w2vec.has_key(ww))
            lemma_builder.filter(lambda ww, rank, val: (val >= dconf.word_fthres and rank <= dconf.word_rthres) or w2vec.has_key(ww))
            # then add extra ones
            for w in extra_word_set:
                if w2vec.has_key(w) and (not word_builder.has_key_currently(w)):
                    word_builder.feed_one(w)
            for w in extra_lemma_set:
                if w2vec.has_key(w) and (not lemma_builder.has_key_currently(w)):
                    lemma_builder.feed_one(w)
            # finially build the vocab and embeds
            word_vocab = word_builder.finish()
            word_embed1 = word_vocab.filter_embed(w2vec, init_nohit=dconf.pretrain_init_nohit, scale=dconf.pretrain_scale)
            lemma_vocab = lemma_builder.finish()
            lemma_embed1 = lemma_vocab.filter_embed(w2vec, init_nohit=dconf.pretrain_init_nohit, scale=dconf.pretrain_scale)
            # first build pool-embeds, the final decision will depend on each of the flags
            # todo(WARN): assert all hit?
            hl_evt_pembed = hl_evt.filter_pembed(w2vec, init_nohit=dconf.pretrain_init_nohit, assert_all_hit=False)
            hl_ef_pembed = hl_ef.filter_pembed(w2vec, init_nohit=dconf.pretrain_init_nohit, assert_all_hit=False)
            hl_arg_pembed = hl_arg.filter_pembed(w2vec, init_nohit=dconf.pretrain_init_nohit, assert_all_hit=False)
            # by-product of filtered output pre-trained embeddings for later faster processing
            if dconf.output_pretrain_filter:
                w2vec.save_hits(dconf.output_pretrain_filter)
        else:
            word_vocab = word_builder.finish_thresh(rthres=dconf.word_rthres, fthres=dconf.word_fthres)
            lemma_vocab = lemma_builder.finish_thresh(rthres=dconf.word_rthres, fthres=dconf.word_fthres)
            word_embed1 = lemma_embed1 = None
            #
            for one_cc in [conf.mconf.hl_evt, conf.mconf.hl_ef, conf.mconf.hl_arg]:
                if hasattr(one_cc, "pool_init_hint"):
                    assert not one_cc.pool_init_hint, "cannot init pool because the overall pre-train-init flag is not set"
            hl_evt_pembed = hl_ef_pembed = hl_arg_pembed = None
        char_vocab = char_builder.finish()
        upos_vocab = upos_builder.finish()
        ulabel_vocab = ulabel_builder.finish()
        # =====
        # finally assign things
        ret.put_voc("word", word_vocab)
        ret.put_voc("lemma", lemma_vocab)
        ret.put_voc("char", char_vocab)
        ret.put_voc("upos", upos_vocab)
        ret.put_voc("ulabel", ulabel_vocab)
        ret.put_emb("word", word_embed1)
        ret.put_emb("lemma", lemma_embed1)
        # don't need to be jsonable since we are using pickle all at once
        # todo(WARN): the conf in vocab is also stored!!
        ret.put_voc("hl_evt", hl_evt)
        ret.put_voc("hl_ef", hl_ef)
        ret.put_voc("hl_arg", hl_arg)
        ret.put_emb("hl_evt", hl_evt_pembed)
        ret.put_emb("hl_ef", hl_ef_pembed)
        ret.put_emb("hl_arg", hl_arg_pembed)
        return ret
