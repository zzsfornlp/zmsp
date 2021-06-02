#

# test vocab + input + embedder + encoder

import numpy as np

from msp.data import VocabBuilder, VocabPackage
from msp.data import TextReader
from msp.utils import Helper, zlog
from msp.nn import layers, BK
from msp.nn.layers import BiAffineScorer
from msp.nn.modules import EmbedConf, MyEmbedder, EncConf, MyEncoder
from msp.zext.seq_helper import DataPadder

def main():
    np.random.seed(1234)
    NUM_POS = 10
    # build vocabs
    reader = TextReader("./test_utils.py")
    vb_word = VocabBuilder("w")
    vb_char = VocabBuilder("c")
    for one in reader:
        vb_word.feed_stream(one.tokens)
        vb_char.feed_stream((c for w in one.tokens for c in w))
    voc_word = vb_word.finish()
    voc_char = vb_char.finish()
    voc_pos = VocabBuilder.build_from_stream(range(NUM_POS), name="pos")
    vpack = VocabPackage({"word": voc_word, "char": voc_char, "pos": voc_pos}, {"word": None})
    # build model
    pc = BK.ParamCollection()
    conf_emb = EmbedConf().init_from_kwargs(init_words_from_pretrain=False, dim_char=10, dim_posi=10, emb_proj_dim=400, dim_extras="50", extra_names="pos")
    conf_emb.do_validate()
    mod_emb = MyEmbedder(pc, conf_emb, vpack)
    conf_enc = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_cnn_layer=1, enc_att_layer=1)
    conf_enc._input_dim = mod_emb.get_output_dims()[0]
    mod_enc = MyEncoder(pc, conf_enc)
    enc_output_dim = mod_enc.get_output_dims()[0]
    mod_scorer = BiAffineScorer(pc, enc_output_dim, enc_output_dim, 10)
    # build data
    word_padder = DataPadder(2, pad_lens=(0, 50), mask_range=2)
    char_padder = DataPadder(3, pad_lens=(0, 50, 20))
    word_idxes = []
    char_idxes = []
    pos_idxes = []
    for toks in reader:
        one_words = []
        one_chars = []
        for w in toks.tokens:
            one_words.append(voc_word.get_else_unk(w))
            one_chars.append([voc_char.get_else_unk(c) for c in w])
        word_idxes.append(one_words)
        char_idxes.append(one_chars)
        pos_idxes.append(np.random.randint(voc_pos.trg_len(), size=len(one_words))+1)   # pred->trg
    word_arr, word_mask_arr = word_padder.pad(word_idxes)
    pos_arr, _ = word_padder.pad(pos_idxes)
    char_arr, _ = char_padder.pad(char_idxes)
    #
    # run
    rop = layers.RefreshOptions(hdrop=0.2, gdrop=0.2, fix_drop=True)
    for _ in range(5):
        mod_emb.refresh(rop)
        mod_enc.refresh(rop)
        mod_scorer.refresh(rop)
        #
        expr_emb = mod_emb(word_arr, char_arr, [pos_arr])
        zlog(BK.get_shape(expr_emb))
        expr_enc = mod_enc(expr_emb, word_mask_arr)
        zlog(BK.get_shape(expr_enc))
        #
        mask_expr = BK.input_real(word_mask_arr)
        score0 = mod_scorer.paired_score(expr_enc, expr_enc, mask_expr, mask_expr)
        score1 = mod_scorer.plain_score(expr_enc.unsqueeze(-2), expr_enc.unsqueeze(-3), mask_expr.unsqueeze(-1), mask_expr.unsqueeze(-2))
        #
        zmiss = float(BK.avg(score0-score1))
        assert zmiss < 0.0001
    zlog("OK")
    pass

if __name__ == '__main__':
    main()
