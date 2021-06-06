#

# the new model with the mtl modules

from typing import List, Dict
from msp2.nn import BK
from msp2.nn.layers import node_reg
from msp2.nn.modules import BaseModelConf, BaseModel
from msp2.nn.modules import PlainEncoderConf, PlainEncoder, BertEncoderConf, BertEncoder
from msp2.tasks.common.models.embedder import MyEmbdder, MyEmbedderConf
from msp2.data.vocab import VocabPackage
from msp2.data.inst import yield_sents, Sent, DataPadder
from msp2.data.stream import BatchHelper
from msp2.tasks.common.models.iter import *
from .mtl.srl import *

# =====

class MtlModelConf(BaseModelConf):
    def __init__(self):
        super().__init__()
        # components
        # -- input + enc
        self.input_choice = "emb"  # emb/bert
        self.bert_conf = BertEncoderConf()  # bert
        self.emb_conf = MyEmbedderConf()  # embedding
        self.enc_conf = PlainEncoderConf()  # plain encoder
        # -- dec
        self.srl_conf = MySRLConf()  # srl
        # --
        # decode
        self.decode_sent_thresh_diff = 20  # sent diff thresh in decoding
        self.decode_sent_thresh_batch = 8  # num sent one time

@node_reg(MtlModelConf)
class MtlModel(BaseModel):
    def __init__(self, conf: MtlModelConf, vpack: VocabPackage):
        super().__init__(conf)
        conf: MtlModelConf = self.conf
        self.vpack = vpack
        # =====
        # components
        # -- input
        self.input_choice_emb, self.input_choice_bert = [conf.input_choice==z for z in ["emb", "bert"]]
        _need_berter = self.input_choice_bert or (self.input_choice_emb and conf.emb_conf.ec_bert.dim>0)
        self.berter = BertEncoder(conf.bert_conf) if _need_berter else None
        self.emb = MyEmbdder(conf.emb_conf, vpack, berter=self.berter) if self.input_choice_emb else None
        # inputter's dim -> [bsize, slen, **D**]
        if self.input_choice_emb:
            self.input_dim = self.emb.get_output_dims()[0]
        elif self.input_choice_bert:
            self.input_dim = self.berter.get_output_dims()[0]
        else:
            raise NotImplementedError(f"Error: UNK input choice of {conf.input_choice}")
        # -- encoder -> [bsize, slen, D']
        self.enc = PlainEncoder(conf.enc_conf, input_dim=self.input_dim)
        self.enc_dim = self.enc.get_output_dims()[0]
        # -- decoder
        self.srl = MySRLNode(conf.srl_conf, vpack.get_voc('evt'), vpack.get_voc('arg'), isize=self.enc_dim)
        # --
        self.idec_manager = IdecManager()
        self.idec_manager.add_nodes(self.srl.get_idec_nodes())
        # =====
        # --
        zzz = self.optims  # finally build optim!

    # helper: embed and encode
    def _input_emb(self, insts: List[Sent]):
        input_map = self.emb.run_inputs(insts)
        mask_expr, emb_expr = self.emb.run_embeds(input_map)  # [bs, slen, ?]
        vstate = self.idec_manager.new_vstate(emb_expr, mask_expr)
        vstate.update(emb_expr, force_lidx=0)  # especially the L0, but no output!
        return mask_expr, emb_expr, vstate

    def _input_bert(self, insts: List[Sent]):
        bi = self.berter.create_input_batch_from_sents(insts)
        mask_expr = BK.input_real(DataPadder.lengths2mask([len(z) for z in insts]))  # [bs, slen, *]
        vstate = self.idec_manager.new_vstate(None, mask_expr)  # todo(+N): currently ignore emb layer!
        bert_expr = self.berter.forward(bi, vstate=vstate)
        return mask_expr, bert_expr, vstate

    def _emb_and_enc(self, insts: List[Sent]):
        # input
        if self.input_choice_emb:  # use emb
            mask_expr, emb_expr, vstate = self._input_emb(insts)
        elif self.input_choice_bert:  # use bert
            mask_expr, emb_expr, vstate = self._input_bert(insts)
        else:
            raise NotImplementedError()
        # encode
        enc_expr = self.enc.forward(emb_expr, mask_expr=mask_expr, vstate=vstate)
        return mask_expr, emb_expr, enc_expr, vstate  # [bs, slen, *]

    # =====
    def loss_on_batch(self, annotated_insts: List, loss_factor=1., training=True, **kwargs):
        self.refresh_batch(training)
        # --
        sents: List[Sent] = list(yield_sents(annotated_insts))
        # emb and enc
        mask_expr, emb_expr, enc_expr, vstate = self._emb_and_enc(sents)
        # dec
        loss_srl = self.srl.loss(sents, enc_expr, mask_expr)
        # --
        # final loss and backward
        info = {"inst": len(annotated_insts), "sent": len(sents), "fb": 1}
        final_loss, loss_info = self.collect_loss([loss_srl])
        info.update(loss_info)
        if training:
            assert final_loss.requires_grad
            BK.backward(final_loss, loss_factor)
        return info

    def predict_on_batch(self, insts: List, **kwargs):
        conf: MtlModelConf = self.conf
        self.refresh_batch(False)
        # --
        sents: List[Sent] = list(yield_sents(insts))
        with BK.no_grad_env():
            # batch run inside if input is doc
            sent_buckets = BatchHelper.group_buckets(
                sents, thresh_diff=conf.decode_sent_thresh_diff, thresh_all=conf.decode_sent_thresh_batch,
                size_f=lambda x: 1, sort_key=lambda x: len(x))
            for one_sents in sent_buckets:
                # emb and enc
                mask_expr, emb_expr, enc_expr, vstate = self._emb_and_enc(one_sents)
                # dec
                self.srl.predict(one_sents, enc_expr, mask_expr)
        # --
        info = {"inst": len(insts), "sent": len(sents)}
        return info
