#

# Model and Component

__all__ = [
    "NodeGroupConf", "NodeGroup", "LossHelper", "BaseModelConf", "BaseModel",
]

from typing import List, Dict
from collections import OrderedDict
from itertools import chain
from ..layers.base import *
from msp2.proc import ZModel, SVConf, ScheduledValue
from msp2.nn import OptimConf, BK
from msp2.nn import refresh as nn_refresh
from msp2.proc import TRConf
from msp2.utils import zlog, Conf, DictHelper, WithWrapper, ConfEntryChoices

# =====
# NodeGroup (not a BasicNode, simply a grouper!)

class NodeGroupConf(Conf):
    def __init__(self):
        # node names
        self.g_names = []
        # optimizer
        self.g_optim = ConfEntryChoices({"yes": OptimConf(), "no": None}, "no")
        # lrate factor (by default 1.0)
        self.g_lrf = SVConf().direct_update(val=1., which_idx="eidx", mode="none", min_val=0.)
        # margin (by default 0.0)
        self.g_margin = SVConf().direct_update(val=0.)
        # node: not here!
        # # loss lambda (for the whole module)
        # self.loss_lambda = SVConf().direct_update(val=1.)

class NodeGroup:
    def __init__(self, conf: NodeGroupConf, group_name: str, nodes: List[BasicNode]):
        self.conf = conf
        self.group_name = group_name
        self.nodes = nodes
        # scheduled values
        self.lrf = ScheduledValue(f"lrf", conf.g_lrf)
        self.margin = ScheduledValue(f"margin", conf.g_margin)
        # self.loss_lambda = ScheduledValue(f"{self.group_name}:lambda", conf.loss_lambda)

    def get_scheduled_values(self) -> Dict:
        # return [self.lrf, self.margin, self.loss_lambda]
        return OrderedDict([("lrf", self.lrf), ("margin", self.margin)])

# =====
# loss helper

# common format of loss_info is: {'name1': {'sum', 'count', ...}, 'name2': ...}
class LossHelper:
    # compile one loss for one component
    @staticmethod
    def compile_leaf_loss(name: str, loss_sum: BK.Expr, loss_count: BK.Expr, loss_lambda=1., **other_values):
        if loss_lambda <= 0.:
            return OrderedDict()
        local_dict = {"sum0": loss_sum, "sum": loss_sum*loss_lambda, "count": loss_count, "run": 1}
        local_dict.update(other_values)
        return OrderedDict({name: local_dict})

    # collect all losses for one component
    @staticmethod
    def compile_component_loss(name: str, sub_losses: List[Dict], loss_lambda=1.):
        ret_dict = OrderedDict()
        if loss_lambda <= 0.:
            return ret_dict
        name_prefix = name+"." if name else name
        for one_sub_loss in sub_losses:
            for k, v in one_sub_loss.items():
                ret_dict[f"{name_prefix}{k}"] = v
        for one_item in ret_dict.values():
            one_item["sum"] = one_item["sum"] * loss_lambda
        return ret_dict

    # collect all losses for (possibly) multiple runs
    @staticmethod
    def combine_multiple_losses(inputs: List[Dict]):
        # each input is a flattened loss Dict
        ret = OrderedDict()
        for one_input in inputs:
            if one_input is None:  # skip None
                continue
            for name, leaf_info in one_input.items():
                if name in ret:
                    # adding together
                    target_info = ret[name]
                    for k, v in leaf_info.items():
                        target_info[k] = target_info.get(k, 0.) + v
                else:
                    # direct set
                    ret[name] = leaf_info
        return ret

# =====
# Model

# todo(note): now training and testing confs are mixed here!
class BaseModelConf(BasicConf):
    def __init__(self):
        super().__init__()
        # overall conf (for non-component nodes)
        # optimizer
        self.main_optim = OptimConf()
        # lrate factor
        self.main_lrf = SVConf().direct_update(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        # different groups (useful for different lrates)
        for ii in range(10):  # this should be enough!
            setattr(self, f"mpg{ii}", NodeGroupConf())  # model-param-group
        # dropouts (general default settings)
        self.df_hdrop = 0.1  # hdrop
        self.df_gdrop = 0.33  # gdrop (always fixed for recurrent connections)
        self.df_idrop = 0.33  # idrop for rnn
        self.df_fix_drop = False  # fix drop for one run for each dropout
        # ema
        self.ema_conf = EMAConf()

    def get_active_mpgs(self):  # get active ones!
        rets = [getattr(self, f"mpg{ii}") for ii in range(10)]
        return [z for z in rets if len(z.g_names)>0]

@node_reg(BaseModelConf)
class BaseModel(BasicNode, ZModel):  # implements ZModel's interface!
    def __init__(self, conf: BaseModelConf):
        super().__init__(conf)
        conf: BaseModelConf = self.conf
        # --
        self.main_lrf = ScheduledValue(f"_lrf", conf.main_lrf)
        # --
        self._groups: Dict[str, NodeGroup] = {}  # str -> NodeGroup
        self._optims = None
        # for refreshing (previous is training)
        self.previous_refresh_training = True
        # ema
        self.ema = None

    # ======
    # optims: lazy building

    @property
    def optims(self) -> OrderedDict:
        if self._optims is None:
            self._optims = self._build_optims()
        return self._optims

    def _build_optims(self):
        conf: BaseModelConf = self.conf
        optims = OrderedDict({"__main__": BK.Optim(conf.main_optim, self.main_lrf)})
        assert len(list(self.parameters(recurse=False))) == 0, "Parameters should be within sub-modules!"
        # --
        # build groups
        for ii, cc in enumerate(self.conf.get_active_mpgs()):
            g_nodes = [getattr(self, n) for n in cc.g_names]
            self.add_node_group(f"mpg{ii}", cc, g_nodes)
        # first add groups
        all_children_ids = set(id(n) for n in self.children())
        all_grouped_ids = set()
        for g in self._groups.values():
            assert g.group_name not in optims
            new_optim = BK.Optim(conf.main_optim if g.conf.g_optim is None else g.conf.g_optim, g.lrf)
            optims[g.group_name] = new_optim
            # --
            for n in g.nodes:
                assert id(n) in all_children_ids and id(n) not in all_grouped_ids
                new_optim.add_params(n.parameters())
                all_grouped_ids.add(id(n))
        # then add the rest
        for n in self.children():
            assert isinstance(n, BasicNode)
            if id(n) not in all_grouped_ids:  # add to default one!
                optims["__main__"].add_params(n.parameters())
        # =====
        # check all params
        BK.get_current_param_manager().check_params(self.parameters())
        zlog(f"Build optims for Model {self} ok!")
        # --
        self.ema = EMA(self.parameters(), conf.ema_conf)
        return optims

    # =====
    def add_node_group(self, group_name: str, gconf: NodeGroupConf, nodes: List[BasicNode]):
        assert group_name not in self._groups
        # first check that nodes must be direct children
        all_children_ids = set(id(n) for n in self.children())
        all_grouped_ids = set(id(n) for g in self._groups.values() for n in g.nodes)
        assert all((id(n) in all_children_ids and id(n) not in all_grouped_ids) for n in nodes)
        # then add group
        self._groups[group_name] = NodeGroup(gconf, group_name, nodes)

    def get_scheduled_values(self):
        ret = super().get_scheduled_values()  # from Model
        for g in self._groups.values():  # from NodeGroup
            DictHelper.update_dict(ret, g.get_scheduled_values(), key_prefix=(g.group_name+"."))
        return ret

    def _get_scheduled_values(self):
        return OrderedDict([("_lrf", self.main_lrf)])

    def refresh(self, rop: RefreshOptions = None):
        # todo(note): only reading training field!!
        self.refresh_batch(rop.training)

    # called before each mini-batch
    def refresh_batch(self, training: bool):
        # refresh graph
        # todo(warn): make sure to remember clear this one
        nn_refresh()
        # refresh nodes
        if not training:
            # if not self.previous_refresh_training:
            #     # todo(+1): currently no need to refresh testing mode multiple times
            #     return
            rop = RefreshOptions(training=False)  # default no dropout
            self.previous_refresh_training = False
        else:
            conf: BaseModelConf = self.conf
            # make a default version
            rop = RefreshOptions(training=True, hdrop=conf.df_hdrop, idrop=conf.df_idrop,
                                 gdrop=conf.df_gdrop, fix_drop=conf.df_fix_drop)
            self.previous_refresh_training = True
        # then common refresh as BasicNode
        BasicNode.refresh(self, rop)

    def update(self, lrate: float, grad_factor: float):
        for optim in self.optims.values():
            optim.update(lrate, grad_factor)
        self.ema.update(self.parameters())

    # with model.ema_wrap_dev(): ...
    def ema_wrap_dev(self):
        return WithWrapper(lambda: self.ema.copy_to(self.parameters()), lambda: self.ema.copy_back(self.parameters()))

    # =====
    # collect all loss
    def collect_loss(self, losses: List[Dict], ret_dict=False):
        final_loss_dict = LossHelper.combine_multiple_losses(losses)  # loss_name -> {}
        if len(final_loss_dict) <= 0:
            return BK.zeros([]), {}  # no loss!
        final_losses = []
        final_losses_dict = OrderedDict()
        ret_info = OrderedDict()
        for loss_name, loss_info in final_loss_dict.items():
            one_real_loss = loss_info['sum']/(loss_info['count']+1e-5)
            # --
            final_losses.append(one_real_loss)  # add scalar-tensor
            final_losses_dict[loss_name] = one_real_loss
            # --
            for k, v in loss_info.items():
                ret_info[f"loss:{loss_name}_{k}"] = float(v.item()) if hasattr(v, "item") else float(v)
            ret_info[f"loss:{loss_name}_div"] = float(one_real_loss.item()) if hasattr(one_real_loss, "item") else float(v)
        if ret_dict:
            return final_losses_dict, ret_info
        else:
            return BK.stack(final_losses).sum() if len(final_losses)>0 else BK.zeros([]), ret_info

    # == load and save models
    # todo(+n): load and save optim states to allow continue training?
    def load(self, path, strict=None):
        if strict is not None:
            BK.load_model(self, path, strict=strict)
        else:  # otherwise, first try strict, then relax if there are errors
            try:
                BK.load_model(self, path, strict=True)
            except:
                import traceback
                zlog(f"#== Error in strict loading:\n{traceback.format_exc()}\n#==")
                BK.load_model(self, path, strict=False)
        zlog(f"Load {self} from {path}.", func="io")

    def save(self, path):
        BK.save_model(self, path)
        zlog(f"Save {self} to {path}.", func="io")

    def __repr__(self):
        return f"{self.__class__}(NumParam={self.count_param_number()})"

    def str_details(self):
        return super().__repr__()

# =====
# EMA (also storing params)

class EMAConf(Conf):
    def __init__(self):
        self.ema_decay = 0.  # 0. as off
        self.ema_copy_back = True
        self.ema_update_step = 1

class EMA:
    def __init__(self, parameters, conf: EMAConf, **kwargs):
        conf = EMAConf.direct_conf(conf, **kwargs)
        self.conf = conf
        self.step = 0
        if conf.ema_decay > 0.:
            self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        else:  # no need!
            self.shadow_params = None
        conf.ema_decay = (conf.ema_decay ** conf.ema_update_step)  # with step
        self.tmp_params = None
        # --

    def update(self, parameters):
        # --
        conf: EMAConf = self.conf
        self.step += 1
        decay = conf.ema_decay
        if decay <= 0.: return
        if self.step % conf.ema_update_step != 0: return
        # --
        with BK.no_grad_env():
            one_minus_decay = 1.0 - decay
            parameters = [p for p in parameters if p.requires_grad]
            assert len(parameters) == len(self.shadow_params)
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))
        # --

    def copy_to(self, parameters):
        # --
        decay = self.conf.ema_decay
        if decay <= 0.: return
        # --
        assert self.tmp_params is None
        self.tmp_params = []
        with BK.no_grad_env():
            parameters = [p for p in parameters if p.requires_grad]
            assert len(parameters) == len(self.shadow_params)
            for s_param, param in zip(self.shadow_params, parameters):
                self.tmp_params.append(param.clone().detach())
                param.data.copy_(s_param.data)
        # --

    def copy_back(self, parameters):
        # --
        decay = self.conf.ema_decay
        if decay <= 0.: return
        # --
        if self.conf.ema_copy_back:  # whether we copy back?
            with BK.no_grad_env():
                parameters = [p for p in parameters if p.requires_grad]
                assert len(parameters) == len(self.tmp_params)
                for t_param, param in zip(self.tmp_params, parameters):
                    param.data.copy_(t_param.data)
        self.tmp_params = None
        # --

# --
# b msp2/nn/modules/base:228
