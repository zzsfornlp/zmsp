#                                                   #
# This code is based on https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py,
# as well as https://github.com/allenai/sledgehammer/blob/master/scripts/temperature_scaling/temperature_scaling.py
# Under the MIT license:
#
# MIT License
#
# Copyright (c) 2017 Geoff Pleiss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F

from collections import Counter
from mspx.utils import Conf, init_everything, zwarn, zlog, default_json_serializer
from mspx.data.rw import ReaderGetterConf
from mspx.data.inst import yield_sents

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

# This function probably should live outside of this class, but whatever
def set_temperature(arr_logits, arr_labels, optimizer, cuda_device: int):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    cuda_func = lambda x: (x if cuda_device<0 else x.cuda(cuda_device))
    nll_criterion = cuda_func(nn.CrossEntropyLoss())
    ece_criterion = cuda_func(_ECELoss())

    # --
    # First: collect all the logits and labels for the validation set
    # arr_logits, arr_labels = read_data(input_file)
    t_logits = cuda_func(torch.FloatTensor(arr_logits))  # [*, V]
    labels = cuda_func(torch.LongTensor(arr_labels))  # [*]
    # --
    logits = t_logits
    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    zlog(f'#====')
    zlog('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
    ece_criterion.print_bins(logits, labels)
    best_nll = before_temperature_nll
    best_temp = 1.
    # for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.005, 0.0001, 0.00005, 0.00001, 0.000005]:
    for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.005, 0.0001]:
        temp_model = cuda_func(ModelWithTemperature(None))
        optimizer.optimize(temp_model, lr, nll_criterion, logits, labels, before_temperature_nll)
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(temp_model.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(temp_model.temperature_scale(logits), labels).item()
        if after_temperature_nll < best_nll:
            best_nll = after_temperature_nll
            best_temp = temp_model.temperature.item()
            zlog(f'lr: {lr}: temp: {best_temp:.3f} After temperature - NLL: {after_temperature_ece:.3f},'
                 f' ECE: {after_temperature_ece:.3f}')
    ece_criterion.print_bins(logits/best_temp, labels)
    return best_temp

class Optimizer():
    def __init__(self, num_epochs: int):
        self._num_epochs = num_epochs

    def optimize(self, temp_model: ModelWithTemperature, lr: float,
                 nll_criterion,
                 logits: torch.FloatTensor, labels: torch.FloatTensor,
                 before_temperature_nll: float):
        raise NotImplementedError


class LBFGS_optimizer(Optimizer):
    def __init__(self, num_epochs: int):
        super(LBFGS_optimizer, self).__init__(num_epochs)

    def optimize(self, temp_model: ModelWithTemperature, lr: float,
                 nll_criterion,
                 logits: torch.FloatTensor, labels: torch.FloatTensor,
                 before_temperature_nll: float):
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([temp_model.temperature], lr=lr, max_iter=self._num_epochs)

        def cal_eval():
            loss = nll_criterion(temp_model.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(cal_eval)


class Adam_optimizer(Optimizer):
    def __init__(self, num_epochs: int):
        super(Adam_optimizer, self).__init__(num_epochs)

    def optimize(self, temp_model: ModelWithTemperature, lr: float,
                 nll_criterion,
                 logits: torch.FloatTensor, labels: torch.FloatTensor,
                 before_temperature_nll: float):

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.Adam([temp_model.temperature], lr=lr)
        best_loss = before_temperature_nll
        best_epoch = 0

        for i in range(self._num_epochs):
            temp_model.zero_grad()

            loss = nll_criterion(temp_model.temperature_scale(logits), labels)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = 1
            elif i - best_epoch > 50:
                print("Stopped at {} with value {}".format(best_epoch, best_loss))
                break


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def print_bins(self, logits, labels):
        import pandas as pd
        # --
        softmaxes = F.softmax(logits, dim=1)  # [*, L]
        confidences, predictions = torch.max(softmaxes, 1)  # [*]
        accuracies = predictions.eq(labels)  # [*]
        # --
        _data = {"L": [], "U": [], "Prop": [], "Acc": [], "Conf": [], "Diff": []}
        # --
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            _data["L"].append(bin_lower)
            _data["U"].append(bin_upper)
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            _data["Prop"].append(prop_in_bin.item())
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                diff = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                _data["Acc"].append(accuracy_in_bin.item())
                _data["Conf"].append(avg_confidence_in_bin.item())
                _data["Diff"].append(diff.item())
            else:
                _data["Acc"].append(0.)
                _data["Conf"].append(0.)
                _data["Diff"].append(0.)
        # --
        df = pd.DataFrame(_data)
        zlog(df.to_string())
        # --

def evaluate(t_gold, t_pred, t_mask=None, nil=0):
    if t_mask is not None:
        t_gold = t_gold[t_mask]
        t_pred = t_pred[t_mask]
    t_corr = (t_gold == t_pred)  # correct
    t_gold_v = (t_gold != nil)  # not nil
    t_pred_v = (t_pred != nil)  # not nil
    A = t_corr.sum() / torch.ones_like(t_gold).sum().clamp(min=1)
    P = (t_corr & t_pred_v).sum() / t_pred_v.sum().clamp(min=1)
    R = (t_corr & t_gold_v).sum() / t_gold_v.sum().clamp(min=1)
    F = 2 * P * R / (P+R).clamp(min=1e-5)  # note: once a BUG (min=1)!!
    M = torch.tensor(1.) if t_mask is None else t_mask.float().mean()
    ret = {a: round(b.item(), 4) for a,b in zip("APRFM", [A,P,R,F,M])}
    return ret

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.key_cali = ""  # arr name!
        self.mode = 'ts'  # ts=temperature-scaling, th=deciding threshold
        # --
        # for temperature scaling
        self.cuda_device = 0
        self.optimizer = 'adam'
        self.n_epochs = 10000
        # --
        # for thresholding
        self.th_nil = 0  # idx for NIL? note: 0 is a safe one, since dpar would not use it!
        self.th_metric = 'F'  # which eval metric?
        self.th_utility = '1-m'  # al-utility; 1-m=1-margin
        self.th_ranges = []  # specific checking thresholds: empty for auto!
        self.th_auto_start = 0.1  # percentage to start
        self.th_trg = [0.5]  # metric target: curr+(max-curr)*THR; note: take the last one for compatibility
        # debugging purpose
        self.th_res_save = ""  # output file!
        self.th_plot = False
        # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # read data
    cc = Counter()
    all_logits, all_labels = [], []
    for sent in yield_sents(conf.R.get_reader()):
        cc['sent'] += 1
        arr = sent.arrs[conf.key_cali]
        all_logits.append(arr[..., 1:])
        all_labels.append(arr[..., 0])
        cc['item'] += arr.shape[0]
    maxV = max(z.shape[-1] for z in all_logits)
    pad_all_logits = []
    for arrs in all_logits:
        for one_arr in arrs:
            _pad = maxV - len(one_arr)
            _pad_arr = np.pad(one_arr, [(0,_pad)], constant_values=one_arr.min()-10.)  # not too much!
            pad_all_logits.append(_pad_arr)
    arr_logits, arr_labels = np.stack(pad_all_logits, 0), np.concatenate(all_labels, 0)
    zlog(f"Read {cc} {arr_logits.shape} {arr_labels.shape}")
    # --
    # calculate ACC & F1
    t_gold = torch.as_tensor(arr_labels).long()  # [*]
    t_logits = torch.as_tensor(arr_logits)  # [*, V]
    t_pred = t_logits.argmax(-1)  # [*]
    overall_res = evaluate(t_gold, t_pred, nil=conf.th_nil)
    overall_metric = overall_res[conf.th_metric]
    zlog(f"Overall-result = {overall_res}")
    zlog(f"Overall-metric = {overall_metric}")
    # --
    # main jobs
    if conf.mode == 'ts':
        optimizer = conf.optimizer
        n_epochs = conf.n_epochs
        if optimizer == 'adam':
            optimizer = Adam_optimizer(n_epochs)
        elif optimizer == 'lbfgs':
            optimizer = LBFGS_optimizer(n_epochs)
        else:
            raise NotImplementedError(f"UNK optimizer of {optimizer}")
        best_temp = set_temperature(arr_logits, arr_labels, optimizer, conf.cuda_device)
        zlog(f"BEST_TEMP = {best_temp}")
    elif conf.mode == 'th':
        if conf.th_utility == '1-m':  # 1-margin
            _top2, _ = t_logits.topk(2, -1)  # [*, 2]
            t_utility = 1. - (_top2[:,0]-_top2[:,1]).abs()  # [*]
        else:
            raise NotImplementedError()
        # check utility-threshold-result
        all_results = []
        # --
        t_sorted, _ = t_utility.sort()
        if len(conf.th_ranges) > 0:  # specified!
            _ths = conf.th_ranges
        else:  # auto decide!
            _start = int(len(t_sorted) * conf.th_auto_start)
            _ths = t_sorted[_start:].unique_consecutive()  # only care about the thresholds!
        # --
        best_metric = overall_metric  # starting with the base one!
        best_line0 = None
        for one_th in _ths:
            _mask = (t_utility <= one_th)  # trust these if using this threshold
            _res = evaluate(t_gold, t_pred, t_mask=_mask, nil=conf.th_nil)
            _res_metric = _res[conf.th_metric]
            all_results.append({'th': float(one_th), 'perc': _res['M'], 'metric': _res_metric, 'res': _res})
            if _res_metric >= best_metric:
                best_metric = _res_metric
                best_line0 = all_results[-1]
        # --
        # search for target: note: currently only take one hyper-parameter!
        trg_metric = overall_metric + (best_metric - overall_metric) * conf.th_trg[-1]
        print_results = []
        best_th, best_line = None, None
        for one_result in all_results:
            _metric = one_result['metric']
            if _metric >= trg_metric:
                best_line = one_result
                best_th = one_result['th']
            if _metric == best_metric or len(print_results)==0 or abs(_metric - print_results[-1]['metric']) >= 0.01:
                print_results.append(one_result)  # add for printing!
        df = pd.DataFrame.from_records(print_results)
        zlog(f"{df.to_string()}")
        zlog(f"TH-res (all={overall_metric},best={best_metric},trg={trg_metric:.4f})")
        zlog(f"BEST_line0 = {best_line0}")
        zlog(f"BEST_line = {best_line}")
        zlog(f"BEST_TH = {best_th}")
        # --
        if conf.th_res_save:
            default_json_serializer.to_file(all_results, conf.th_res_save)
        if conf.th_plot:
            import matplotlib.pyplot as plt
            plt.plot([z['perc'] for z in all_results], [z['metric'] for z in all_results])
            plt.show()
        # --
    # --

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

"""
python3 -m mspx.scripts.tools.calibrate input_path: key_cali:ext0_cali
"""
