import torch
from torch import nn
import torch.nn.functional as F


class ilm_in(nn.Module):
    def __init__(self, channel, num_groups=32, key_group_size=16, reduction=16, eps=1e-5):
        super(ilm_in, self).__init__()
        assert channel % num_groups == 0
        assert channel % key_group_size == 0

        self.num_groups = channel
        self.feat_per_group = channel // self.num_groups

        self.key_groups = channel // key_group_size
        self.key_feat_per_group = key_group_size

        if self.key_groups > reduction:
            self.embed_size = self.key_groups // reduction
        else:
            self.embed_size = 2

        self.fc_embed = nn.Sequential(
            nn.Linear(self.key_groups, self.embed_size, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_weight = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.key_groups, bias=False),
            nn.Sigmoid()
        )
        self.weight_bias = nn.Parameter(torch.ones(1, channel, 1, 1))

        self.fc_bias = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.key_groups, bias=False),
            nn.Tanh()
        )
        self.bias_bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.key_groups

        x = x.view(b, g, 1, -1)
        key_mean = x.mean(-1, keepdim=True)
        key_var = x.var(-1, keepdim=True)

        weight = self.fc_weight(key_var.view(b, g)).view(b, g, 1).repeat(1, 1, self.key_feat_per_group).view(b, c, 1, 1)
        weight = weight + self.weight_bias
        bias = self.fc_bias(key_mean.view(b, g)).view(b, g, 1).repeat(1, 1, self.key_feat_per_group).view(b, c, 1, 1)
        bias = bias + self.bias_bias

        g = self.num_groups
        x = x.view(b, g, 1, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(b, c, h, w)

        return x * weight + bias

    # def forward(self, x):
    #     b, c, h, w = x.size()
    #     g = self.num_groups

    #     x_ = x.view(b, g, 1, -1)
    #     mean = x_.mean(-1, keepdim=True)
    #     var = x_.var(-1, keepdim=True)

    #     weight = self.fc_weight(var.view(b, g)).view(b, g, 1).repeat(1, 1, self.feat_per_group).view(b, c, 1, 1)
    #     weight = weight + self.weight_bias
    #     bias = self.fc_bias(mean.view(b, g)).view(b, g, 1).repeat(1, 1, self.feat_per_group).view(b, c, 1, 1)
    #     bias = bias + self.bias_bias

    #     outs = []
    #     for f, w, b in zip(x.split(1, 0), weight.split(1, 0), bias.split(1, 0)):
    #         outs.append(F.group_norm(f, self.num_groups, w.squeeze(), b.squeeze(), self.eps))

    #     return torch.cat(outs, 0)