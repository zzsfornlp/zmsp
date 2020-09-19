#

# train a simple linear prober on the attentions

from msp import utils
from msp.data import MultiCatStreamer, InstCacher

from ..run.confs import OverallConf, init_everything, build_model
from ..run.run import get_data_reader, PreprocessStreamer, index_stream, batch_stream, MltTrainingRunner
from ..run.vocab import MLMVocabPackage

from typing import List
from copy import deepcopy
from msp.nn import BK
from ..model.mtl import BaseModel, BaseModelConf, MtlMlmModel, GeneralSentence
from ..model.mods.dpar import DparG1DecoderConf, DparG1Decoder, PairScorerConf

#
class LinearProbeConf(BaseModelConf):
    def __init__(self):
        super().__init__()

class LinearProbeModel(BaseModel):
    def __init__(self, model: MtlMlmModel):
        super().__init__(BaseModelConf())
        # -----
        self.model = model
        model.refresh_batch(False)
        # simple linear layer
        dpar_conf = deepcopy(model.conf.dpar_conf)
        dpar_conf.pre_dp_space = 0
        dpar_conf.dps_conf = PairScorerConf().init_from_kwargs(use_input0=False, use_input1=False, use_input_pair=True,
                                                               use_biaffine=False, use_ff1=False, use_ff2=True,
                                                               ff2_hid_layer=0, use_bias=False)
        dpar_conf.optim.optim = model.conf.main_optim.optim
        inputp_dim = len(model.encoder.layers) * model.enc_attn_count  # concat all attns
        self.dpar = self.add_component("dpar", DparG1Decoder(self.pc, None, inputp_dim, dpar_conf, model.inputter))

    def fb_on_batch(self, insts: List[GeneralSentence], training=True, loss_factor=1.,
                    rand_gen=None, assign_attns=False, **kwargs):
        self.refresh_batch(training)
        # get inputs with models
        with BK.no_grad_env():
            input_map = self.model.inputter(insts)
            emb_t, mask_t, enc_t, cache, enc_loss = self.model._emb_and_enc(input_map, collect_loss=True)
            input_t = BK.concat(cache.list_attn, -1)  # [bs, slen, slen, L*H]
        losses = [self.dpar.loss(insts, BK.zeros([1,1]), input_t, mask_t)]
        # -----
        info = self.collect_loss_and_backward(losses, training, loss_factor)
        info.update({"fb": 1, "sent": len(insts), "tok": sum(len(z) for z in insts)})
        return info

    def inference_on_batch(self, insts: List[GeneralSentence], **kwargs):
        conf = self.conf
        self.refresh_batch(False)
        # print(f"{len(insts)}: {insts[0].sid}")
        with BK.no_grad_env():
            # decode for dpar
            input_map = self.model.inputter(insts)
            emb_t, mask_t, enc_t, cache, _ = self.model._emb_and_enc(input_map, collect_loss=False)
            input_t = BK.concat(cache.list_attn, -1)  # [bs, slen, slen, L*H]
            self.dpar.predict(insts, BK.zeros([1,1]), input_t, mask_t)
        return {}

#
def main(args):
    conf = init_everything(args)
    dconf, mconf = conf.dconf, conf.mconf
    # dev/test can be non-existing!
    if not dconf.dev and dconf.test:
        utils.zwarn("No dev but give test, actually use test as dev (for early stopping)!!")
    dt_golds, dt_cuts = [], []
    for file, one_cut in [(dconf.dev, dconf.cut_dev), (dconf.test, "")]:  # no cut for test!
        if len(file)>0:
            utils.zlog(f"Add file `{file}(cut={one_cut})' as dt-file #{len(dt_golds)}.")
            dt_golds.append(file)
            dt_cuts.append(one_cut)
    if len(dt_golds) == 0:
        utils.zwarn("No dev set, then please specify static lrate schedule!!")
    # data
    train_streamer = PreprocessStreamer(get_data_reader(dconf.train, dconf.input_format, cut=dconf.cut_train),
                                        lower_case=dconf.lower_case, norm_digit=dconf.norm_digit)
    dt_streamers = [PreprocessStreamer(get_data_reader(f, dconf.dev_input_format, cut=one_cut),
                                       lower_case=dconf.lower_case, norm_digit=dconf.norm_digit)
                    for f, one_cut in zip(dt_golds, dt_cuts)]
    # vocab
    if mconf.no_build_dict:
        vpack = MLMVocabPackage.build_by_reading(dconf.dict_dir)
    else:
        # include dev/test only for convenience of including words hit in pre-trained embeddings
        vpack = MLMVocabPackage.build_from_stream(dconf.vconf, train_streamer, MultiCatStreamer(dt_streamers))
        vpack.save(dconf.dict_dir)
    # model
    model = build_model(conf, vpack)
    # index the data
    train_inst_preparer = model.get_inst_preper(True)
    test_inst_preparer = model.get_inst_preper(False)
    to_cache = dconf.cache_data
    to_cache_shuffle = dconf.to_cache_shuffle
    # todo(note): make sure to cache both train and dev to save time for cached computation
    backoff_pos_idx = dconf.backoff_pos_idx
    train_iter = batch_stream(index_stream(train_streamer, vpack, to_cache, to_cache_shuffle, train_inst_preparer, backoff_pos_idx), mconf.train_batch_size, mconf, True)
    dt_iters = [batch_stream(index_stream(z, vpack, to_cache, to_cache_shuffle, test_inst_preparer, backoff_pos_idx), mconf.test_batch_size, mconf, False) for z in dt_streamers]
    # training runner
    tr = MltTrainingRunner(mconf.rconf, model, vpack, dev_outfs=dconf.output_file, dev_goldfs=dt_golds, dev_out_format=dconf.output_format)
    if mconf.train_preload_model:
        tr.load(dconf.model_load_name, mconf.train_preload_process)
    # =====
    # switch with the linear model
    linear_model = LinearProbeModel(model)
    tr.model = linear_model
    # =====
    # go
    tr.run(train_iter, dt_iters)
    utils.zlog("The end of Training.")

def lookat_weights():
    import torch
    m = torch.load("zmodel2.best")
    assert len(m) == 1
    map_w = list(m.values())[0]  # [output, input]
    map_w = map_w[1:]
    # -----
    num_step, num_head = 6, 8
    input_labels = [f"seeS{s}H{h}" for s in range(num_step) for h in range(num_head)]
    output_labels = ["punct", "case", "nsubj", "det", "root", "nmod", "advmod", "obj", "obl", "amod", "compound", "aux", "conj", "mark", "cc", "cop", "advcl", "acl", "xcomp", "nummod", "ccomp", "appos", "flat", "parataxis", "discourse", "expl", "fixed", "list", "iobj", "csubj", "goeswith", "vocative", "reparandum", "orphan", "dep", "dislocated", "clf"]
    assert tuple(map_w.shape) == (len(output_labels), len(input_labels))
    #
    labels = [input_labels, output_labels]
    other_labels = [output_labels, input_labels]
    weights = [map_w.T, map_w]
    for one_label, one_other_label, one_weight in zip(labels, other_labels, weights):
        topk_vals, topk_idxes = one_weight.topk(5, dim=-1)  # [?, 5]
        assert len(one_label) == len(topk_vals)
        for i, lab in enumerate(one_label):
            print(f"{lab}: " + ", ".join([f"{one_other_label[o_idx]}({o_val})" for o_val, o_idx in zip(topk_vals[i], topk_idxes[i])]))

# run
"""
SRC_DIR=../../src/
MODEL_DIR=../run_trans_enu10k/
PYTHONPATH=${SRC_DIR} CUDA_VISIBLE_DEVICES=2 python3 -m pdb ${SRC_DIR}/tasks/cmd.py zmlm.main.train_linear ${MODEL_DIR}/_conf2 train_preload_model:1 model_load_name:${MODEL_DIR}/zmodel2.best lrate.val:0.5 main_optim.optim:sgd lrate_warmup:0 max_epochs:20 lrate.start_bias:1 lrate.scale:5 lrate.m:0.5 dpar_conf.label_neg_rate:0.
# b tasks/zmlm/main/train_linear:50
"""
