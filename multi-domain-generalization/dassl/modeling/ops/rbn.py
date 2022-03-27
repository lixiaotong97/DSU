import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F


class RepresentativeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(RepresentativeBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        ### weights for affine transformation in BatchNorm ###
        if self.affine:
            # self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            # self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight.data.fill_(1)
            self.bias.data.fill_(0)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        ### weights for centering calibration ###
        self.center_weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.center_weight.data.fill_(0)
        ### weights for scaling calibration ###
        self.scale_weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.scale_bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.scale_weight.data.fill_(0)
        self.scale_bias.data.fill_(1)
        ### calculate statistics ###
        self.stas = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        self._check_input_dim(input)

        ####### centering calibration begin #######
        input += self.center_weight.view(1, self.num_features, 1, 1) * self.stas(input)
        ####### centering calibration end #######

        ####### BatchNorm begin #######
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        output = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        ####### BatchNorm end #######

        ####### scaling calibration begin #######
        scale_factor = torch.sigmoid(self.scale_weight * self.stas(output) + self.scale_bias)
        ####### scaling calibration end #######
        if self.affine:
            return self.weight.reshape(1, -1, 1, 1) * scale_factor * output + self.bias.reshape(1, -1, 1, 1)
        else:
            return scale_factor * output


if __name__ == '__main__':
    images = torch.rand(1, 256, 224, 224).cuda(0)
    rbn_layer = RepresentativeBatchNorm2d(256).cuda(0)
    total = sum([param.nelement() for param in rbn_layer.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))
    print(rbn_layer(images).size())
    print('Memory useage: %.4fM' % (torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
