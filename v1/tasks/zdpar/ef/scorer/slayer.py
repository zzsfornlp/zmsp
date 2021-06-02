#

# middle-part structured layer

from typing import List
import numpy as np

from msp.utils import Conf, zcheck
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, Embedding, MultiHeadAttention, AttConf
from msp.zext.seq_helper import DataPadder

# =====
# first the one for ef and g2

# the one that calculates the feature-enhanced repr
class SL0Conf(Conf):
    def __init__(self):
        self._input_dim = -1  # enc's (input) last dimension
        self._num_label = -1  # number of labels
        # model dimensions
        self.dim_label = 30
        #
        self.use_label_feat = True  # whether use feats
        self.use_chs = True  # whether use children feats
        self.chs_num = 0  # how many (recent) ch's siblings to consider, 0 means all: [-num:]
        self.chs_f = "att"  # how to represent the ch set: att=attention, sum=sum, ff=feadfoward (always 0 vector if no ch)
        self.chs_att = AttConf().init_from_kwargs(d_kqv=256, att_dropout=0.1, head_count=2)
        self.use_par = True  # whether use parent feats
        #
        self.zero_extra_output_params = False  # whether makes extra output params zero when init

# another helper class for children set calculation
class ChsReprer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, rconf: SL0Conf):
        super().__init__(pc, None, None)
        # concat enc and label if use-label
        input_dim = rconf._input_dim
        # todo(warn): always add label related params
        self.dim = (input_dim + rconf.dim_label)
        self.is_att, self.is_sum, self.is_ff = [rconf.chs_f == z for z in ["att", "sum", "ff"]]
        self.ff_reshape = [-1, self.dim * rconf.chs_num]  # only for ff
        # todo(+N): not elegant, but add the params to make most things compatible when reloading
        self.mha = self.add_sub_node("fn", MultiHeadAttention(pc, self.dim, input_dim, self.dim, rconf.chs_att))
        if self.is_att:
            # todo(+N): the only possibly inconsistent drop is the one for attention, may consider disabling it.
            self.fnode = self.mha
        elif self.is_sum:
            self.fnode = None
        elif self.is_ff:
            zcheck(rconf.chs_num > 0, "Err: Cannot ff with 0 child")
            self.fnode = self.add_sub_node("fn", Affine(pc, self.dim * rconf.chs_num, self.dim, act="elu"))
        else:
            raise NotImplementedError(f"UNK chs method: {rconf.chs_f}")

    def get_output_dims(self, *input_dims):
        return (self.dim, )

    def zeros(self, batch):
        return BK.zeros((batch, self.dim))

    # [*, D(q)], [*, max-ch, D(kv)], [*, max-ch], [*]
    def __call__(self, chs_input_state_t, chs_input_mem_t, chs_mask_t, chs_valid_t):
        if self.is_att:
            # [*, max-ch, size], ~, [*, 1, size], [*, max-ch] -> [*, size]
            ret = self.fnode(chs_input_mem_t, chs_input_mem_t, chs_input_state_t.unsqueeze(-2), chs_mask_t).squeeze(-2)
        # ignore head for the rest
        elif self.is_sum:
            # ignore head
            if chs_mask_t is None:
                ret = chs_input_mem_t.sum(-2)
            else:
                ret = (chs_input_mem_t*chs_mask_t.unsqueeze(-1)).sum(-2)
        elif self.is_ff:
            if chs_mask_t is None:
                reshaped_input_state_t = chs_input_mem_t.view(self.ff_reshape)
            else:
                reshaped_input_state_t = (chs_input_mem_t*chs_mask_t.unsqueeze(-1)).view(self.ff_reshape)
            ret = self.fnode(reshaped_input_state_t)
        else:
            ret = None
        if chs_valid_t is None:
            return ret
        else:
            # out-most mask: [*, D_DL] * [*, 1]
            return ret * (chs_valid_t.unsqueeze(-1))

# a helper class to calculate structured repr for one node
# todo(note): currently a simple architecture: cur+par+children_summary
class SL0Layer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, rconf: SL0Conf):
        super().__init__(pc, None, None)
        self.dim = rconf._input_dim  # both input/output dim
        # padders for child nodes
        self.chs_start_posi = -rconf.chs_num
        self.ch_idx_padder = DataPadder(2, pad_vals=0, mask_range=2)  # [*, num-ch]
        self.ch_label_padder = DataPadder(2, pad_vals=0)
        #
        self.label_embeddings = self.add_sub_node("label", Embedding(pc, rconf._num_label, rconf.dim_label, fix_row0=False))
        self.dim_label = rconf.dim_label
        # todo(note): now adopting flatten groupings for basic, and then that is all, no more recurrent features
        # group 1: [cur, chs, par] -> head_pre_size
        self.use_chs = rconf.use_chs
        self.use_par = rconf.use_par
        self.use_label_feat = rconf.use_label_feat
        # components (add the parameters anyway)
        # todo(note): children features: children + (label of mod->children)
        self.chs_reprer = self.add_sub_node("chs", ChsReprer(pc, rconf))
        self.chs_ff = self.add_sub_node("chs_ff", Affine(pc, self.chs_reprer.get_output_dims()[0], self.dim, act="tanh"))
        # todo(note): parent features: parent + (label of parent->mod)
        # todo(warn): always add label related params
        par_ff_inputs = [self.dim, rconf.dim_label]
        self.par_ff = self.add_sub_node("par_ff", Affine(pc, par_ff_inputs, self.dim, act="tanh"))
        # no other groups anymore!
        if rconf.zero_extra_output_params:
            self.par_ff.zero_params()
            self.chs_ff.zero_params()

    # calculating the structured representations, giving raw repr-tensor
    # 1) [*, D]; 2) [*, D], [*], [*]; 3) [*, chs-len, D], [*, chs-len], [*, chs-len], [*]
    def calculate_repr(self, cur_t, par_t, label_t, par_mask_t, chs_t, chs_label_t, chs_mask_t, chs_valid_mask_t):
        ret_t = cur_t  # [*, D]
        # padding 0 if not using labels
        dim_label = self.dim_label
        # child features
        if self.use_chs and chs_t is not None:
            if self.use_label_feat:
                chs_label_rt = self.label_embeddings(chs_label_t)  # [*, max-chs, dlab]
            else:
                labels_shape = BK.get_shape(chs_t)
                labels_shape[-1] = dim_label
                chs_label_rt = BK.zeros(labels_shape)
            chs_input_t = BK.concat([chs_t, chs_label_rt], -1)
            chs_feat0 = self.chs_reprer(cur_t, chs_input_t, chs_mask_t, chs_valid_mask_t)
            chs_feat = self.chs_ff(chs_feat0)
            ret_t += chs_feat
        # parent features
        if self.use_par and par_t is not None:
            if self.use_label_feat:
                cur_label_t = self.label_embeddings(label_t)  # [*, dlab]
            else:
                labels_shape = BK.get_shape(par_t)
                labels_shape[-1] = dim_label
                cur_label_t = BK.zeros(labels_shape)
            par_feat = self.par_ff([par_t, cur_label_t])
            if par_mask_t is not None:
                par_feat *= par_mask_t.unsqueeze(-1)
            ret_t += par_feat
        return ret_t

    # todo(note): if no other features, then no change for the repr!
    def forward_repr(self, cur_t):
        return cur_t

    # preparation: padding for chs/par
    def pad_chs(self, idxes_list: List[List], labels_list: List[List]):
        start_posi = self.chs_start_posi
        if start_posi < 0:  # truncate
            idxes_list = [x[start_posi:] for x in idxes_list]
        # overall valid mask
        chs_valid = [(0. if len(z)==0 else 1.) for z in idxes_list]
        # if any valid children in the batch
        if all(x>0 for x in chs_valid):
            padded_chs_idxes, padded_chs_mask = self.ch_idx_padder.pad(idxes_list)  # [*, max-ch], [*, max-ch]
            if self.use_label_feat:
                if start_posi < 0:  # truncate
                    labels_list = [x[start_posi:] for x in labels_list]
                padded_chs_labels, _ = self.ch_label_padder.pad(labels_list)  # [*, max-ch]
                chs_label_t = BK.input_idx(padded_chs_labels)
            else:
                chs_label_t = None
            chs_idxes_t, chs_mask_t, chs_valid_mask_t = \
                BK.input_idx(padded_chs_idxes), BK.input_real(padded_chs_mask), BK.input_real(chs_valid)
            return chs_idxes_t, chs_label_t, chs_mask_t, chs_valid_mask_t
        else:
            return None, None, None, None

    def pad_par(self, idxes: List, labels: List):
        par_idxes_t = BK.input_idx(idxes)
        labels_t = BK.input_idx(labels)
        # todo(note): specifically, <0 means non-exist
        # todo(note): an interesting bug, the bug is ">=" was wrongly written as "<", in this way, 0 will act as the parent of those who actually do not have parents and are to be attached, therefore maybe patterns of "parent=0" will get much positive scores
        # todo(note): ACTUALLY, mainly because of the difference in search and forward-backward!!
        par_mask_t = (par_idxes_t>=0).float()
        par_idxes_t.clamp_(0)  # since -1 will be illegal idx
        labels_t.clamp_(0)
        return par_idxes_t, labels_t, par_mask_t

# =====
# second the specific one for s2

class SL1Conf(Conf):
    def __init__(self):
        self._input_dim = -1  # enc's (input) last dimension
        self._num_label = -1  # number of labels
        #
        self.use_par = True
        self.use_chs = True
        self.sl_par_att = AttConf().init_from_kwargs(d_kqv=256, att_dropout=0., head_count=2)
        self.sl_chs_att = AttConf().init_from_kwargs(d_kqv=256, att_dropout=0., head_count=2)
        self.mix_marginals_head_count = 0  # how many heads to mix
        self.mix_marginals_rate = 1.  # mix into the self attentions
        #
        self.zero_extra_output_params = False  # whether makes extra output params zero when init

# additional structured-masked layer
class SL1Layer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, slconf: SL1Conf):
        super().__init__(pc, None, None)
        self.dim = slconf._input_dim
        self.use_par = slconf.use_par
        self.use_chs = slconf.use_chs
        # parent and children attentional senc
        self.node_par = self.add_sub_node("npar", MultiHeadAttention(pc, self.dim, self.dim, self.dim, slconf.sl_par_att))
        self.node_chs = self.add_sub_node("nchs", MultiHeadAttention(pc, self.dim, self.dim, self.dim, slconf.sl_chs_att))
        self.ff_par = self.add_sub_node("par_ff", Affine(pc, self.dim, self.dim, act="tanh"))
        self.ff_chs = self.add_sub_node("chs_ff", Affine(pc, self.dim, self.dim, act="tanh"))
        # todo(note): currently simply sum them!
        self.mix_marginals_head_count = slconf.mix_marginals_head_count
        self.mix_marginals_rate = slconf.mix_marginals_rate
        if slconf.zero_extra_output_params:
            self.ff_par.zero_params()
            self.ff_chs.zero_params()

    def get_output_dims(self, *input_dims):
        return (self.dim, )

    # calculation for one type of node
    def _calc_one_node(self, node_att: BasicNode, node_ff: BasicNode, enc_expr, qk_mask, qk_marginals):
        # [*, len, D]
        hidden = node_att(enc_expr, enc_expr, enc_expr, mask_qk=qk_mask, eprob_qk=qk_marginals,
                      eprob_mix_rate=self.mix_marginals_rate, eprob_head_count=self.mix_marginals_head_count)
        q_mask = (qk_mask.float().sum(-1)>0.).float()  # [*, len], at least one for query
        hidden = hidden * (q_mask.unsqueeze(-1))  # [*, len, D], zeros for emtpy ones
        output = node_ff(hidden)
        return output

    # [*, slen, D], [*, len-m, len-h]
    def __call__(self, enc_expr, valid_expr, arc_marg_expr):
        # ===== avoid NAN
        def _sum_marg(m, dim):
            s = m.sum(dim).unsqueeze(dim)
            s += (s < 1e-5).float() * 1e-5
            return s
        # =====
        output = enc_expr
        arc_marg_expr = arc_marg_expr * valid_expr  # only keep the after-pruning ones
        if self.use_par:
            m_mask = valid_expr  # [*, lem-m, len-h]
            m_marg = arc_marg_expr / _sum_marg(arc_marg_expr, -1)
            senc_par_expr = self._calc_one_node(self.node_par, self.ff_par, enc_expr, m_mask, m_marg)
            output = output + senc_par_expr
        if self.use_chs:
            h_mask = BK.copy(valid_expr.transpose(-1, -2))  # [*, len-h, len-m]
            h_marg = (arc_marg_expr / _sum_marg(arc_marg_expr, -2)).transpose(-1, -2)
            senc_chs_expr = self._calc_one_node(self.node_chs, self.ff_chs, enc_expr, h_mask, h_marg)
            output = output + senc_chs_expr
        return output
