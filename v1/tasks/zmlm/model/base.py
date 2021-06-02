#

# base models with msp.nn
from typing import List, Dict, Union
from collections import OrderedDict
from msp.model import Model
from msp.nn import BK
from msp.nn import refresh as nn_refresh
from msp.nn.layers import RefreshOptions, BasicNode
from msp.utils import Conf, zlog, JsonRW, Helper
from msp.zext.process_train import RConf, SVConf, ScheduledValue, OptimConf

# =====
# conf
class BaseModelConf(Conf):
    def __init__(self):
        self.rconf = RConf()
        # overall conf
        # optimizer
        self.main_optim = OptimConf()
        # lrate factor (<0 means not activated)
        self.main_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        # dropouts
        # todo(+N): should have distribute dropouts into each Node, but ...
        self.drop_hidden = 0.1  # hdrop
        self.gdrop_rnn = 0.33  # gdrop (always fixed for recurrent connections)
        self.idrop_rnn = 0.33  # idrop for rnn
        self.fix_drop = False  # fix drop for one run for each dropout

# detailize common routines, with sub_module components
class BaseModel(Model):
    def __init__(self, conf: BaseModelConf):
        self.conf = conf
        # ===== Model =====
        self.pc = BK.ParamCollection()
        self.main_lrf = ScheduledValue(f"main:lrf", conf.main_lrf)
        self._scheduled_values = [self.main_lrf]
        # -----
        self.nodes: Dict[str, BasicNode] = OrderedDict()
        self.components: Dict[str, BaseModule] = OrderedDict()
        # for refreshing dropouts
        self.previous_refresh_training = True

    # called before each mini-batch
    def refresh_batch(self, training: bool):
        # refresh graph
        # todo(warn): make sure to remember clear this one
        nn_refresh()
        # refresh nodes
        if not training:
            if not self.previous_refresh_training:
                # todo(+1): currently no need to refresh testing mode multiple times
                return
            self.previous_refresh_training = False
            rop = RefreshOptions(training=False)  # default no dropout
        else:
            conf = self.conf
            rop = RefreshOptions(training=True, hdrop=conf.drop_hidden, idrop=conf.idrop_rnn,
                                 gdrop=conf.gdrop_rnn, fix_drop=conf.fix_drop)
            self.previous_refresh_training = True
        for node in self.nodes.values():
            node.refresh(rop)
        for node in self.components.values():
            node.refresh(rop)

    def update(self, lrate, grad_factor):
        self.pc.optimizer_update(lrate, grad_factor)

    def add_scheduled_value(self, sv):
        self._scheduled_values.append(sv)
        return sv

    def get_scheduled_values(self):
        return self._scheduled_values + Helper.join_list(z.get_scheduled_values() for z in self.components.values())

    # == load and save models
    # todo(warn): no need to load confs here
    def load(self, path, strict=True):
        self.pc.load(path, strict)
        # self.conf = JsonRW.load_from_file(path+".json")
        zlog(f"Load {self.__class__.__name__} model from {path}.", func="io")

    def save(self, path):
        self.pc.save(path)
        JsonRW.to_file(self.conf, path + ".json")
        zlog(f"Save {self.__class__.__name__} model to {path}.", func="io")

    # =====
    def add_component(self, name: str, node: 'BaseModule'):
        assert name not in self.components
        self.components[name] = node
        # set up optim
        self.pc.optimizer_set(node.conf.optim.optim, node.lrf, node.conf.optim, params=node.get_parameters(),
                              check_repeat=True, check_full=True)
        # pop off
        self.pc.nnc_pop(node.name)
        zlog(f"Add component module, {name}: {node}")
        return node

    def add_node(self, name: str, node: BasicNode):
        assert name not in self.nodes
        self.nodes[name] = node
        # set up optim
        self.pc.optimizer_set(self.conf.main_optim.optim, self.main_lrf, self.conf.main_optim,
                              params=node.get_parameters(), check_repeat=True, check_full=True)
        # pop off
        self.pc.nnc_pop(node.name)
        zlog(f"Add node, {name}: {node}")
        return node

    # collect all loss
    def collect_loss_and_backward(self, loss_info_cols: List[Dict], training: bool, loss_factor: float):
        final_loss_dict = LossHelper.combine_multiple(loss_info_cols)  # loss_name -> {}
        if len(final_loss_dict) <= 0:
            return {}  # no loss!
        final_losses = []
        ret_info_vals = OrderedDict()
        for loss_name, loss_info in final_loss_dict.items():
            final_losses.append(loss_info['sum']/(loss_info['count']+1e-5))
            for k in loss_info.keys():
                one_item = loss_info[k]
                ret_info_vals[f"loss:{loss_name}_{k}"] = one_item.item() if hasattr(one_item, "item") else float(one_item)
        final_loss = BK.stack(final_losses).sum()
        if training and final_loss.requires_grad:
            BK.backward(final_loss, loss_factor)
        return ret_info_vals

# =====
class BaseModuleConf(Conf):
    def __init__(self):
        # optimizer
        self.optim = OptimConf()
        # lrate factor (<0 means not activated)
        self.lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        # margin (<0 means not activated)
        self.margin = SVConf().init_from_kwargs(val=0.)
        # loss lambda (for the whole module)
        self.loss_lambda = SVConf().init_from_kwargs(val=1.)
        # my hdrop (<0 means not activated)
        self.hdrop = -1.

class BaseModule(BasicNode):
    def __init__(self, pc: BK.ParamCollection, conf: BaseModuleConf, name=None, init_rop=None, output_dims=None):
        super().__init__(pc, name, init_rop)
        self.conf = conf
        self.output_dims = output_dims
        # scheduled values
        self.lrf = ScheduledValue(f"{self.name}:lrf", conf.lrf)
        self.margin = ScheduledValue(f"{self.name}:margin", conf.margin)
        self.loss_lambda = ScheduledValue(f"{self.name}:lambda", conf.loss_lambda)
        self.hdrop = conf.hdrop
        if self.hdrop >= 0.:
            zlog(f"Forcing different hdrop at {self.name}: {self.hdrop}")

    def refresh(self, rop=None):
        local_changed = (self.hdrop>=0. and rop is not None)
        if local_changed:  # todo(note): modified inplace
            old_hdrop = rop.hdrop
            rop.hdrop = self.hdrop
        super().refresh(rop)
        if local_changed:
            rop.hdrop = old_hdrop

    def get_scheduled_values(self):
        return [self.lrf, self.margin, self.loss_lambda]

    def get_output_dims(self, *input_dims):
        return self.output_dims

    # input list of leafs
    def _compile_component_loss(self, comp_name, losses: List):
        cur_loss_lambda = self.loss_lambda.value
        if cur_loss_lambda <= 0.:
            losses = []  # no loss since closed by component lambda
        ret = LossHelper.compile_component_info(comp_name, losses, loss_lambda=cur_loss_lambda)
        return ret

    # =====
    def loss(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        raise NotImplementedError()

# =====
# loss helper

# common format of loss_info is: {'name1': {'sum', 'count', ...}, 'name2': ...}
class LossHelper:
    @staticmethod
    def compile_leaf_info(name: str, loss_sum: BK.Expr, loss_count: BK.Expr, loss_lambda=1., **other_values):
        local_dict = {"sum0": loss_sum, "sum": loss_sum*loss_lambda, "count": loss_count, "run": 1}
        local_dict.update(other_values)
        return OrderedDict({name: local_dict})

    @staticmethod
    def compile_component_info(name: str, sub_losses: List[Dict], loss_lambda=1.):
        ret_dict = OrderedDict()
        for one_sub_loss in sub_losses:
            for k, v in one_sub_loss.items():
                ret_dict[f"{name}.{k}"] = v
        for one_item in ret_dict.values():
            one_item["sum"] = one_item["sum"] * loss_lambda
        return ret_dict

    @staticmethod
    def combine_multiple(inputs: List[Dict]):
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
