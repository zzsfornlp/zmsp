#

# Basic Span Node:
# -- modeling a span of text

__all__ = [
    "BaseSpanConf", "BaseSpanNode",
]

import numpy as np
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants

class BaseSpanConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1
        # --
        self.use_starts = True  # use starts: [start,
        self.use_ends = True  # use ends: end]
        self.use_softhead = True  # softhead with att
        self.softhead_topk = -1  # <=0 means nope!
        self.use_width = True  # use span width
        self.width_pe = False  # use posi embeddings
        self.width_dim = 50  # length embedding size
        self.width_num = 100  # max length for len embedding, this should be enough
        # --
        self.use_proj = True  # whether further project to a fixed dim
        self.proj_dim = 512  # proj output dim if using proj
        self.proj_conf = AffineConf().direct_update(out_act='elu')

@node_reg(BaseSpanConf)
class BaseSpanNode(BasicNode):
    def __init__(self, conf: BaseSpanConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: BaseSpanConf = self.conf
        # --
        input_dim = conf.isize
        output_dims = []
        if conf.use_starts:  # use start ones
            output_dims.append(input_dim)
        if conf.use_ends:  # use end ones
            output_dims.append(input_dim)
        if conf.use_softhead:  # use softhead
            self.softhead_scorer = AffineNode(None, isize=input_dim, osize=1, no_drop=True)  # no dropout for att-score
            output_dims.append(input_dim)
        if conf.use_width:  # use span-len embedding
            if conf.width_pe:
                # especially make it trainable
                self.width_embed = PosiEmbeddingNode(None, osize=conf.width_dim, max_val=conf.width_num, freeze=False)
            else:  # otherwise
                self.width_embed = EmbeddingNode(None, osize=conf.width_dim, n_words=conf.width_num)
            output_dims.append(conf.width_dim)
        assert len(output_dims)>0, "Must output sth for SpanNode!"
        self.repr_dims = output_dims
        self.output_dim = sum(output_dims)
        if conf.use_proj:
            self.final_proj = AffineNode(conf.proj_conf, isize=self.output_dim, osize=conf.proj_dim)
            self.output_dim = conf.proj_dim

    def extra_repr(self) -> str:
        conf: BaseSpanConf = self.conf
        return f"BaseSpanNode({self.repr_dims}->{conf.proj_dim if conf.use_proj else ''})"

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # =====

    # [bsize, slen, D], 2x[bsize, ?], -- (currently no need for these) [bsize, slen], [bsize, ?]
    # -- input_mask: BK.Expr = None, span_mask: BK.Expr = None):
    # note: must make sure inputs are all valid, here we do not check or mask!!
    def forward(self, input_expr: BK.Expr, widx_expr: BK.Expr, wlen_expr: BK.Expr):
        conf: BaseSpanConf = self.conf
        # --
        # note: check empty, otherwise error
        input_item_shape = BK.get_shape(widx_expr)
        if np.prod(input_item_shape) == 0:
            return BK.zeros(input_item_shape + [self.output_dim])  # return an empty but shaped tensor
        # --
        start_idxes, end_idxes = widx_expr, widx_expr+wlen_expr  # make [start, end)
        # get sizes
        bsize, slen = BK.get_shape(input_expr)[:2]
        # num_span = BK.get_shape(start_idxes, 1)
        arange2_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
        # --
        reprs = []
        if conf.use_starts:  # start [start,
            reprs.append(input_expr[arange2_t, start_idxes])  # [bsize, ?, D]
        if conf.use_ends:  # simply ,end-1]
            reprs.append(input_expr[arange2_t, end_idxes-1])
        if conf.use_softhead:
            # expand range
            all_span_idxes, all_span_mask = expand_ranged_idxes(widx_expr, wlen_expr, 0, None)  # [bsize, ?, MW]
            # flatten
            flatten_all_span_idxes = all_span_idxes.view(bsize, -1)  # [bsize, ?*MW]
            flatten_all_span_mask = all_span_mask.view(bsize, -1)  # [bsize, ?*MW]
            # get softhead score (consider mask here)
            softhead_scores = self.softhead_scorer(input_expr).squeeze(-1)  # [bsize, slen]
            flatten_all_span_scores = softhead_scores[arange2_t, flatten_all_span_idxes]  # [bsize, ?*MW]
            flatten_all_span_scores += (1.-flatten_all_span_mask) * Constants.REAL_PRAC_MIN
            all_span_scores = flatten_all_span_scores.view(all_span_idxes.shape)  # [bsize, ?, MW]
            # reshape and (optionally topk) and softmax
            softhead_topk = conf.softhead_topk
            if softhead_topk>0 and BK.get_shape(all_span_scores,-1)>softhead_topk:  # further select topk; note: this may save mem
                final_span_score, _tmp_idxes = all_span_scores.topk(softhead_topk, dim=-1, sorted=False)  # [bsize, ?, K]
                final_span_idxes = all_span_idxes.gather(-1, _tmp_idxes)  # [bsize, ?, K]
            else:
                final_span_score, final_span_idxes = all_span_scores, all_span_idxes  # [bsize, ?, MW]
            final_prob = final_span_score.softmax(-1)  # [bsize, ?, ??]
            # [bsize, ?, ??, D]
            final_repr = input_expr[arange2_t, final_span_idxes.view(bsize, -1)].view(BK.get_shape(final_span_idxes)+[-1])
            weighted_repr = (final_repr * final_prob.unsqueeze(-1)).sum(-2)  # [bsize, ?, D]
            reprs.append(weighted_repr)
        if conf.use_width:
            cur_width_embed = self.width_embed(wlen_expr)  # [bsize, ?, DE]
            reprs.append(cur_width_embed)
        # concat
        concat_repr = BK.concat(reprs, -1)  # [bsize, ?, SUM]
        if conf.use_proj:
            ret = self.final_proj(concat_repr)  # [bsize, ?, DR]
        else:
            ret = concat_repr
        return ret
