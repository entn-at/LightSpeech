import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import hparams as hp
import utils


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='relu'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x .contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor()

    def LR(self, x, duration_predictor_output, alpha=1.0, mel_max_length=None):
        output = list()

        for batch, expand_target in zip(x, duration_predictor_output):
            output.append(self.expand(batch, expand_target, alpha))

        if mel_max_length:
            output = utils.pad(output, mel_max_length)
        else:
            output = utils.pad(output)

        return output

    def expand(self, batch, predicted, alpha):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size*alpha), -1))
        out = torch.cat(out, 0)

        return out

    def rounding(self, num):
        if num - int(num) >= 0.5:
            return int(num) + 1
        else:
            return int(num)

    # Deprecated
    def cal_D_target(self, duration_predictor_output, length_c, length_mel):
        targets = list()
        for i, l in enumerate(length_c):
            origin_batch = duration_predictor_output[i][0:l]
            sum_batch = torch.sum(origin_batch, 0).data

            ratio = length_mel[i] / sum_batch
            processed_batch = origin_batch * ratio

            target_batch = [self.rounding(ele) for ele in processed_batch]
            temp_sum = sum(target_batch)
            diff = (length_mel[i] - temp_sum).item()
            bias = [ele.item()-(int(ele)+0.5) for ele in processed_batch]

            if diff != 0:
                if diff > 0:
                    index_list = [[-1000, -1] for _ in range(diff)]
                    for i, ele in enumerate(bias):
                        if ele < 0:
                            for ind, (value, _) in enumerate(index_list):
                                if ele > value:
                                    index_list[ind][0] = ele
                                    index_list[ind][1] = i
                                    break
                    for (_, index) in index_list:
                        target_batch[index] += 1
                else:
                    index_list = [[1000, -1] for _ in range(-diff)]
                    for i, ele in enumerate(bias):
                        if ele > 0:
                            for ind, (value, _) in enumerate(index_list):
                                if ele < value:
                                    index_list[ind][0] = ele
                                    index_list[ind][1] = i
                                    break
                    for (_, index) in index_list:
                        target_batch[index] -= 1
            targets.append(target_batch)

        for i, target in enumerate(targets):
            targets[i] = np.array(target)
        D = torch.from_numpy(utils.pad_1D(targets)).cuda()

        return D

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if self.training:
            output = self.LR(x, target, mel_max_length=mel_max_length)
            return output, duration_predictor_output
        else:
            for idx, ele in enumerate(duration_predictor_output[0]):
                duration_predictor_output[0][idx] = self.rounding(ele)
            output = self.LR(x, duration_predictor_output, alpha)

            return output


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.linear_layer_1 = Linear(512, 512, w_init='relu')
        self.relu_1 = nn.ReLU()
        self.linear_layer_2 = Linear(512, 1, w_init='relu')
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        out = self.linear_layer_1(x)
        out = self.relu_1(out)
        out = self.linear_layer_2(out)
        out = self.relu_2(out)

        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)

        return out


class PostNet(nn.Module):
    """ Post Net (Not Add Dropout) """

    def __init__(self):
        super(PostNet, self).__init__()
        self.gru_1 = nn.GRU(hp.n_mel_channels,
                            hp.n_mel_channels,
                            num_layers=1,
                            batch_first=True)

        # self.linear = Linear(hp.n_mel_channels,
        #                      hp.n_mel_channels)

        self.gru_2 = nn.GRU(hp.n_mel_channels,
                            hp.n_mel_channels,
                            num_layers=1,
                            batch_first=True)

        self.dropout = nn.Dropout(0.1)

    def mask(self, mel_1, mel_2, length_mel, max_mel_len):
        x_mask = ~utils.get_mask_from_lengths(length_mel, max_mel_len)
        x_mask = x_mask.expand(hp.n_mel_channels,
                               x_mask.size(0), x_mask.size(1))
        x_mask = x_mask.permute(1, 2, 0)
        mel_1.data.masked_fill_(x_mask, 0.0)
        mel_2.data.masked_fill_(x_mask, 0.0)

        return mel_1, mel_2

    def forward(self, mels, length_mel, max_mel_len):
        self.gru_1.flatten_parameters()
        x, _ = self.gru_1(mels)

        mel_postnet_1 = mels + x
        self.gru_2.flatten_parameters()
        y, _ = self.gru_2(mel_postnet_1)

        mel_postnet_2 = mel_postnet_1 + x + y
        mel_postnet_1, mel_postnet_2 = self.mask(
            mel_postnet_1, mel_postnet_2, length_mel, max_mel_len)

        return mel_postnet_1, mel_postnet_2

    def inference(self, mels):
        x, _ = self.gru_1(mels)
        mel_postnet_1 = mels + x
        y, _ = self.gru_2(mel_postnet_1)
        mel_postnet_2 = mel_postnet_1 + x + y

        return mel_postnet_1, mel_postnet_2


if __name__ == "__main__":
    # Test
    test_dp = DurationPredictor()
    print(utils.get_param_num(test_dp))
