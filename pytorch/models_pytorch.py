import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """
    
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



class EmbeddingLayers_pooling(nn.Module):
    def __init__(self):
        super(EmbeddingLayers_pooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=1,
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=2,
                               padding=(4, 4), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=4,
                               padding=(8, 8), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=8,
                               padding=(16, 16), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        return x

class CnnPooling_Max(nn.Module):
    def __init__(self, classes_num):
        super(CnnPooling_Max, self).__init__()

        self.emb = EmbeddingLayers_pooling()
        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
	x = self.emb(input)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        x = F.log_softmax(self.fc_final(x), dim=-1)

        return x

class CnnPooling_Avg(nn.Module):
    def __init__(self, classes_num):
        super(CnnPooling_Avg, self).__init__()

        self.emb = EmbeddingLayers_pooling()
        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
	x = self.emb(input)

        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output

class CnnPooling_Attention(nn.Module):
    def __init__(self, classes_num):
        super(CnnPooling_Attention, self).__init__()

        self.emb = EmbeddingLayers_pooling()
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

    def init_weights(self):
        pass

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        x = self.emb(input)

        output = self.attention(x)

        return output


class Attention2d(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention2d, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        self.att.weight.data.fill_(0.)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return F.sigmoid(x)+0.1

        elif activation == 'log_softmax':
            return F.log_softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        # (samples_num, channel, time_steps * freq_bins)
        att = att.view(att.size(0), att.size(1), att.size(2) * att.size(3))
        cla = cla.view(cla.size(0), cla.size(1), cla.size(2) * cla.size(3))

        epsilon = 0.1 # 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        Return_heatmap = False
        if Return_heatmap:
            return x, norm_att
        else:
            return x


class EmbeddingLayers(nn.Module):
    def __init__(self):
        super(EmbeddingLayers, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        a1 = F.relu(self.bn1(self.conv1(x)))
        a1 = F.max_pool2d(a1, kernel_size=(2, 2))
        a2 = F.relu(self.bn2(self.conv2(a1)))
        a2 = F.max_pool2d(a2, kernel_size=(2, 2))
        a3 = F.relu(self.bn3(self.conv3(a2)))
        a3 = F.max_pool2d(a3, kernel_size=(2, 2))
        emb = F.relu(self.bn4(self.conv4(a3)))
        emb = F.max_pool2d(emb, kernel_size=(2, 2))

        if return_layers is False:
            return emb
        else:
            return [a1, a2, a3, emb]

class DecisionLevelMaxPooling(nn.Module):
    def __init__(self, classes_num):

        super(DecisionLevelMaxPooling, self).__init__()

        self.emb = EmbeddingLayers()
        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        x = self.emb(input)

        # (samples_num, 512, hidden_units)
        output = F.max_pool2d(x, kernel_size=x.shape[2:])
        output = output.view(output.shape[0:2])

        output = F.log_softmax(self.fc_final(output), dim=-1)

        return output

class DecisionLevelAvgPooling(nn.Module):
    def __init__(self, classes_num):
        super(DecisionLevelAvgPooling, self).__init__()

        self.emb = EmbeddingLayers()
        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        x = self.emb(input)

        # (samples_num, 512, hidden_units)
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output

class DecisionLevelFlatten(nn.Module):
    def __init__(self, classes_num):
        super(DecisionLevelFlatten, self).__init__()

        self.emb = EmbeddingLayers()
        self.fc_final = nn.Linear(40960, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        x = self.emb(input)

        # (samples_num, 512, hidden_units)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output

class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, classes_num):

        super(DecisionLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers()
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

    def init_weights(self):
        pass

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return output
