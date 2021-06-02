#

# some basic components

from typing import List
from msp.utils import Conf
from msp.nn import BK
from msp.nn.layers import BasicNode, ActivationHelper, Affine, NoDropRop, Dropout

# -----
# Affine/MLP layer for convenient use
class AffineHelperNode(BasicNode):
    def __init__(self, pc, input_dim, hid_dim: int, hid_act='linear', hid_drop=0., hid_piece4init=1,
                 out_dim=0, out_fbias=0., out_fact="linear", out_piece4init=1, init_scale=1.):
        super().__init__(pc, None, None)
        # -----
        # hidden layer
        self.hid_aff = self.add_sub_node("hidden", Affine(
            pc, input_dim, hid_dim, n_piece4init=hid_piece4init, init_scale=init_scale, init_rop=NoDropRop()))
        self.hid_act_f = ActivationHelper.get_act(hid_act)
        self.hid_drop = self.add_sub_node("drop", Dropout(pc, (hid_dim, ), fix_rate=hid_drop))
        # -----
        # output layer (optional)
        self.final_output_dim = hid_dim
        # todo(+N): how about split hidden layers for each specific output
        self.out_fbias = out_fbias  # fixed extra bias
        self.out_act_f = ActivationHelper.get_act(out_fact)
        # no output dropouts
        if out_dim > 0:
            assert hid_act != "linear", "Please use non-linear activation for mlp!"
            assert hid_piece4init == 1, "Strange hid_piece4init for hidden layer with out_dim>0"
            self.final_aff = self.add_sub_node("final", Affine(
                pc, hid_dim, out_dim, n_piece4init=out_piece4init, init_scale=init_scale, init_rop=NoDropRop()))
            self.final_output_dim = out_dim
        else:
            self.final_aff = None

    def get_output_dims(self, *input_dims):
        return (self.final_output_dim, )

    def __call__(self, input_t):
        # hidden
        hid_t = self.hid_aff(input_t)
        # act and drop
        act_hid_t = self.hid_act_f(hid_t)
        drop_hid_t = self.hid_drop(act_hid_t)
        # optional extra layer
        if self.final_aff:
            out0 = self.final_aff(drop_hid_t)
            out1 = self.out_act_f(out0 + self.out_fbias)  # fixed bias and final activation
            return out1
        else:
            return drop_hid_t

# -----
# gumbel-softmax / concrete-node
class ConcreteNodeConf(Conf):
    def __init__(self):
        self.use_gumbel = False
        self.use_argmax = False
        self.prune_val = 0.  # hard prune for probs

# discrete / gumbel softmax
class ConcreteNode(BasicNode):
    def __init__(self, pc, conf: ConcreteNodeConf):
        super().__init__(pc, None, None)
        self.use_gumbel = conf.use_gumbel
        self.use_argmax = conf.use_argmax
        self.prune_val = conf.prune_val
        self.gumbel_eps = 1e-10

    # [*, S] -> normalized [*, S]
    def __call__(self, scores, temperature=1., dim=-1):
        is_training = self.rop.training
        # only use stochastic at training
        if is_training:
            if self.use_gumbel:
                gumbel_eps = self.gumbel_eps
                G = (BK.rand(BK.get_shape(scores)) + gumbel_eps).clamp(max=1.)  # [0,1)
                scores = scores - (gumbel_eps - G.log()).log()
        # normalize
        probs = BK.softmax(scores / temperature, dim=dim)  # [*, S]
        # prune and re-normalize?
        if self.prune_val > 0.:
            probs = probs * (probs > self.prune_val).float()
            # todo(note): currently no re-normalize
            # probs = probs / probs.sum(dim=dim, keepdim=True)  # [*, S]
        # argmax and ste
        if self.use_argmax:  # use the hard argmax
            max_probs, _ = probs.max(dim, keepdim=True)  # [*, 1]
            # todo(+N): currently we do not re-normalize here, should it be done here?
            st_probs = (probs>=max_probs).float() * probs  # [*, S]
            if is_training:  # (hard-soft).detach() + soft
                st_probs = (st_probs - probs).detach() + probs  # [*, S]
            return st_probs
        else:
            return probs

# -----
# affine combiner
class AffineCombiner(BasicNode):
    def __init__(self, pc, input_dims: List[int], use_affs: List[bool], out_dim: int,
                 out_act='linear', out_drop=0., param_init_scale=1.):
        super().__init__(pc, None, None)
        # -----
        self.input_dims = input_dims
        self.use_affs = use_affs
        self.out_dim = out_dim
        # =====
        assert len(input_dims) == len(use_affs)
        self.aff_nodes = []
        for d, use in zip(input_dims, use_affs):
            if use:
                one_aff = self.add_sub_node("aff", Affine(pc, d, out_dim, init_scale=param_init_scale, init_rop=NoDropRop()))
            else:
                assert d == out_dim, f"Dimension mismatch for skipping affine: {d} vs {out_dim}!"
                one_aff = None
            self.aff_nodes.append(one_aff)
        self.out_act_f = ActivationHelper.get_act(out_act)
        self.out_drop = self.add_sub_node("drop", Dropout(pc, (out_dim, ), fix_rate=out_drop))

    def get_output_dims(self, *input_dims):
        return (self.out_dim, )

    # input is List[*, len, dim]
    def __call__(self, input_ts: List):
        comb_t = 0.
        assert len(input_ts) == len(self.aff_nodes), "Input number mismatch"
        for one_t, one_aff in zip(input_ts, self.aff_nodes):
            if one_aff:
                add_t = one_aff(one_t)
            else:
                add_t = one_t
            comb_t += add_t
        comb_t_act = self.out_act_f(comb_t)
        comb_t_drop = self.out_drop(comb_t_act)
        return comb_t_drop
