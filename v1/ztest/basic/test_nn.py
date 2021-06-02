#

import numpy as np
from msp.nn import layers, BK

def main():
    pc = BK.ParamCollection()
    N_BATCH, N_SEQ = 8, 4
    N_HIDDEN, N_LAYER = 5, 3
    N_INPUT = N_HIDDEN
    N_FF = 10
    # encoders
    rnn_encoder = layers.RnnLayerBatchFirstWrapper(pc, layers.RnnLayer(pc, N_INPUT, N_HIDDEN, N_LAYER, bidirection=True))
    cnn_encoder = layers.Sequential(pc, [layers.CnnLayer(pc, N_INPUT, N_HIDDEN, 3, act="relu") for _ in range(N_LAYER)])
    att_encoder = layers.Sequential(pc, [layers.TransformerEncoderLayer(pc, N_INPUT, N_FF) for _ in range(N_LAYER)])
    dropout_md = layers.DropoutLastN(pc)
    #
    rop = layers.RefreshOptions(hdrop=0.2, gdrop=0.2, dropmd=0.2, fix_drop=True)
    rnn_encoder.refresh(rop)
    cnn_encoder.refresh(rop)
    att_encoder.refresh(rop)
    dropout_md.refresh(rop)
    #
    x = BK.input_real(np.random.randn(N_BATCH, N_SEQ, N_INPUT))
    x_mask = np.asarray([[1.]*z+[0.]*(N_SEQ-z) for z in np.random.randint(N_SEQ//2, N_SEQ, N_BATCH)])
    y_rnn = rnn_encoder(x, x_mask)
    y_cnn = cnn_encoder(x, x_mask)
    y_att = att_encoder(x, x_mask)
    zz = dropout_md(y_att)
    print("The end.")
    pass

if __name__ == '__main__':
    main()
