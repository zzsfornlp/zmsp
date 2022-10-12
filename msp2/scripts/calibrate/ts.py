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
import torch
from torch import nn, optim
from torch.nn import functional as F


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

# =====
# note: specific data reader!!
def read_data(file: str):
    import pickle
    import numpy as np
    from collections import Counter
    cc = Counter()
    # --
    all_logits, all_labels = [], []
    with open(file, 'rb') as fd:
        while True:
            try:
                obj = pickle.load(fd)  # load one
                arr_logits, arr_labels = obj["logits"], obj["labels"]  # [slen, L, NL], [slen]
                all_logits.append(arr_logits)
                all_labels.append(arr_labels)
                cc["obj"] += 1
                cc["item"] += len(arr_labels)
            except EOFError:
                break
    # --
    # finally concat all
    ret_logits = np.concatenate(all_logits, axis=0)
    ret_labels = np.concatenate(all_labels, axis=0)
    print(f"Load from {file} ({ret_logits.shape}, {ret_labels.shape}): {cc}")
    return ret_logits, ret_labels
# =====

# This function probably should live outside of this class, but whatever
def set_temperature(input_data, n_layers, optimizer, cuda_device: int):
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
    arr_logits, arr_labels = input_data
    assert arr_logits.shape[-1] == n_layers
    labels = cuda_func(torch.LongTensor(arr_labels))  # [*]
    all_logits = [z.squeeze(-1) for z in cuda_func(torch.FloatTensor(arr_logits)).chunk(n_layers, dim=-1)]  # *[*, L]
    # --

    temps = [1 for i in range(n_layers)]

    for layer_index in range(n_layers):
        logits = all_logits[layer_index]

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'#====\nLayer {layer_index}')
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        ece_criterion.print_bins(logits, labels)

        best_nll = before_temperature_nll
        best_temp = 1
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
                temps[layer_index] = best_temp
            #                print('Optimal temperature: %.3f' % temp_model.temperature.item())
                print('lr: %f: temp: %.3f After temperature - NLL: %.3f, ECE: %.3f' % (
            lr, best_temp, after_temperature_nll, after_temperature_ece))

        ece_criterion.print_bins(logits/best_temp, labels)

    return temps


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
        print(df.to_string())
        # --

# ==
def main():
    parser = arg_parser()

    args = parser.parse_args()

    optimizer = args.optimizer
    n_epochs = 10000

    if optimizer == 'adam':
        optimizer = Adam_optimizer(n_epochs)
    elif optimizer == 'lbfgs':
        optimizer = LBFGS_optimizer(n_epochs)
    else:
        print("Illegal optimizer {}, must be one of (adam, lbfgs)")
        return -2
    # --
    rets = set_temperature(read_data(args.file), args.layer, optimizer, args.cuda_device)
    print(f"Final values are: {rets}")
    # --

def arg_parser():
    from argparse import ArgumentParser
    """Extracting CLI arguments"""
    p = ArgumentParser(add_help=False)
    p.add_argument("-u", "--cuda_device", help="CUDA device (or -1 for CPU)", type=int, default=0)
    p.add_argument('-o', '--optimizer', help="Optimizer (adam or lbfgs)", type=str, default="adam")
    p.add_argument('-f', '--file', help="data file", type=str, required=True)
    p.add_argument('-l', '--layer', help="num of layers", type=int, default=4)
    return p

if __name__ == '__main__':
    import sys
    sys.exit(main())

# --
# steps for calibration
"""
# step 1: get prediction logits and correct labels
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../src/ python3 -m pdb -m msp2.tasks.zmtl.main.test ./_conf test:../../../pb/conll05/dev.conll.ud.json arg_conf.pred_weights:0,0,0,1 pred_logits_output_file:out.pkl
# step 2: run ts.py
python3 ts.py --file out.pkl
# => [1.4102871417999268, 1.8438681364059448, 2.4931490421295166, 2.6398212909698486] 
# step 3: re-predict with cali-temps
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../src/ python3 -m msp2.tasks.zmtl.main.test ./_conf test:../../../pb/conll05/dev.conll.ud.json arg_conf.pred_weights:0,0,0,1 srl_conf.pred_exit_by_arg:yes srl_conf.pred_exit_by_arg.exit_crit:prob srl_conf.pred_exit_by_arg.exit_min_k:1 srl_conf.pred_exit_by_arg.exit_thresh:0.9 "arg_conf.app_ts:[1.4102871417999268, 1.8438681364059448, 2.4931490421295166, 2.6398212909698486]"
"""
