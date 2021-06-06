#

# adapter for XMU's model

from typing import List, Dict
from msp2.nn import BK
from msp2.nn.layers import node_reg
from msp2.nn.layers import ModuleWrapper
from msp2.nn.modules import BaseModelConf, BaseModel
from msp2.data.inst import yield_sents, Sent, DataPadder
from msp2.data.vocab import SeqSchemeHelperStr
from msp2.utils import zlog

# --
try:
    import tagger.data as data
    import torch.distributed as dist
    import tagger.models as models
    import tagger.optimizers as optimizers
    import tagger.utils as utils
    import tagger.utils.summary as summary
    from tagger.utils.validation import ValidationWorker
    import torch
except:
    pass
# --

def default_params():
    params = utils.HParams(
        input="",
        output="",
        model="transformer",
        vocab=["", ""],
        pad="<pad>",
        bos="<eos>",
        eos="<eos>",
        unk="<unk>",
        # Dataset
        batch_size=4096,
        fixed_batch_size=False,
        min_length=1,
        max_length=256,
        buffer_size=10000,
        # Initialization
        initializer_gain=1.0,
        initializer="uniform_unit_scaling",
        # Regularization
        scale_l1=0.0,
        scale_l2=0.0,
        # Training
        script="",
        warmup_steps=4000,
        train_steps=100000,
        update_cycle=1,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        adadelta_rho=0.95,
        adadelta_epsilon=1e-6,
        clipping="global_norm",
        clip_grad_norm=5.0,
        learning_rate=1.0,
        learning_rate_schedule="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        device_list=[0],
        embedding="",
        # Validation
        keep_top_k=50,
        frequency=10,
        # Checkpoint Saving
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        save_summary=True,
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
    )
    return params

def merge_params(params1, params2):
    import six
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params

def override_params(params, args):
    params.model = args.model or params.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.parse(args.parameters)

    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(params.vocab[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(params.vocab[1])

    params.vocabulary = {
        "source": src_vocab, "target": tgt_vocab
    }
    params.lookup = {
        "source": src_w2idx, "target": tgt_w2idx
    }
    params.mapping = {
        "source": src_idx2w, "target": tgt_idx2w
    }

    return params

def convert_to_string(inputs, tensor, params):
    import torch
    # inputs = torch.squeeze(inputs)
    inputs = torch.squeeze(inputs, dim=1)
    inputs = inputs.tolist()
    # tensor = torch.squeeze(tensor, dim=1)
    tensor = torch.squeeze(tensor, dim=1)
    tensor = tensor.tolist()
    decoded = []

    for wids, lids in zip(inputs, tensor):
        output = []
        for wid, lid in zip(wids, lids):
            if wid == 0:
                break
            output.append(params.mapping["target"][lid])
        decoded.append(b" ".join(output))

    return decoded

# =====
class XmuModelConf(BaseModelConf):
    def __init__(self):
        super().__init__()
        # --
        self.model = "deepatt"
        self.input = ""
        self.output = ""
        self.vocabulary = []  # narg==2
        self.parameters = "save_summary=false,feature_size=100,hidden_size=200,filter_size=800,residual_dropout=0.2,num_hidden_layers=10,attention_dropout=0.1,relu_dropout=0.1,batch_size=4096,optimizer=adadelta,initializer=orthogonal,initializer_gain=1.0,train_steps=600000,learning_rate_schedule=piecewise_constant_decay,learning_rate_values=[1.0,0.5,0.25,],learning_rate_boundaries=[400000,50000],device_list=[0],clip_grad_norm=1.0,embedding=glove.6B.100d.txt,script=run.sh"
        # todo(note): make sure embedding/vocab is at current running dir

class NpWarapper:
    def __init__(self, arr):
        self.arr = arr

    def numpy(self):
        return self.arr

@node_reg(XmuModelConf)
class XmuModel(BaseModel):
    def __init__(self, conf: XmuModelConf, vpack):
        super().__init__(conf)
        conf: XmuModelConf = self.conf
        self.vpack = vpack
        # =====
        # --
        # init their model
        model_cls = models.get_model("deepatt")
        # --
        params = default_params()
        params = merge_params(params, model_cls.default_params(None))
        # params = import_params(args.output, args.model, params)
        params = override_params(params, conf)
        # --
        self.params = params
        model = model_cls(params).cuda()
        model.load_embedding(params.embedding)
        # --
        self.embedding = data.load_glove_embedding(params.embedding)
        # =====
        # wrap their model
        self.M = ModuleWrapper(model, None)
        self.bio_helper = SeqSchemeHelperStr("BIO")
        # --
        zzz = self.optims  # finally build optim!

    def loss_on_batch(self, annotated_insts: List, loss_factor=1., training=True, **kwargs):
        self.refresh_batch(training)
        # --
        sents: List[Sent] = list(yield_sents(annotated_insts))
        # ==
        # extend to events
        import numpy as np
        bsize = sum(len(z.events) for z in sents)
        mlen = max(len(z) for z in sents)
        arr_preds = np.full([bsize, mlen], 0., dtype=np.int32)
        arr_inputs = np.full([bsize, mlen], b'<pad>', dtype=object)
        arr_labels = np.full([bsize, mlen], b'<pad>', dtype=object)
        ii = 0
        for sent in sents:
            for evt in sent.events:
                widx, wlen = evt.mention.get_span()
                assert wlen == 1
                # --
                arr_preds[ii, widx] = 1
                arr_inputs[ii, :len(sent)] = [s.lower().encode() for s in sent.seq_word.vals]
                # --
                tmp_labels = ["O"] * len(sent)
                for arg in evt.args:
                    role = arg.role
                    a_widx, a_wlen = arg.arg.mention.get_span()
                    a_labs = ["B-" + role] + ["I-" + role] * (a_wlen - 1)
                    assert all(z=="O" for z in tmp_labels[a_widx:a_widx+a_wlen])
                    tmp_labels[a_widx:a_widx+a_wlen] = a_labs
                # --
                arr_labels[ii, :len(sent)] = [z.encode() for z in tmp_labels]
                # --
                ii += 1
        assert ii == bsize
        features, labels = data.lookup(({"preds": NpWarapper(arr_preds), "inputs": NpWarapper(arr_inputs)}, NpWarapper(arr_labels)), "train", self.params)
        # ==
        final_loss = self.M(features, labels)
        info = {"inst": len(annotated_insts), "sent": len(sents), "fb": 1, "loss": final_loss.item()}
        if training:
            assert final_loss.requires_grad
            BK.backward(final_loss, loss_factor)
        zlog(f"batch shape = {len(annotated_insts)} {bsize} {mlen} {bsize*mlen}")
        return info

    def predict_on_batch(self, insts: List, **kwargs):
        self.refresh_batch(False)
        # --
        # self.M.node.eval()
        # --
        sents: List[Sent] = list(yield_sents(insts))
        with BK.no_grad_env():
            # collect input
            # ==
            # extend to events
            import numpy as np
            bsize = sum(len(z.events) for z in sents)
            # --
            if bsize == 0:
                return {"inst": len(insts), "sent": len(sents)}
            # --
            mlen = max(len(z) for z in sents)
            arr_preds = np.full([bsize, mlen], 0., dtype=np.int32)
            arr_inputs = np.full([bsize, mlen], b'<pad>', dtype=object)
            arr_labels = np.full([bsize, mlen], b'<pad>', dtype=object)
            ii = 0
            for sent in sents:
                for evt in sent.events:
                    widx, wlen = evt.mention.get_span()
                    assert wlen == 1
                    # --
                    arr_preds[ii, widx] = 1
                    arr_inputs[ii, :len(sent)] = [s.lower().encode() for s in sent.seq_word.vals]
                    # --
                    ii += 1
            assert ii == bsize
            features = data.lookup(({"preds": NpWarapper(arr_preds), "inputs": NpWarapper(arr_inputs)}, NpWarapper(arr_labels)), "infer", self.params, self.embedding)
            # ==
            labels = self.M.node.argmax_decode(features)
            batch = convert_to_string(features["inputs"], labels, self.params)
            # write output
            ii = 0
            for sent in sents:
                sent.delete_frames("ef")
                for evt in sent.events:
                    evt.clear_args()
                    # --
                    a_labs = batch[ii].decode().split()
                    evt.info["slab"] = a_labs
                    # print(" ".join(a_labs) + "\n")
                    assert len(a_labs) == len(sent)
                    spans = self.bio_helper.tags2spans(a_labs)
                    # --
                    for widx, wlen, role in spans:
                        ef = sent.make_entity_filler(widx, wlen)
                        evt.add_arg(ef, role)
                    # --
                    ii += 1
            assert ii == bsize
        # --
        info = {"inst": len(insts), "sent": len(sents)}
        return info

# =====
# "train_extras:no_build_dict:1 mconf:xmu vocabulary:deep_srl/vocab.txt,deep_srl/label.txt"
# "train_use_cache:0 dev_use_cache:0 valid_first:1"
