import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from .layers import FrozenBatchNorm2d
from . import vgg, resnet

class vgg_feature_extractor(nn.Module):
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(vgg_feature_extractor, self).__init__()
        backbone = vgg.__dict__[backbone_name](
                pretrained=pretrained_backbone, pretrained_weights=pretrained_weights)
            
        features, _ = list(backbone.features.children()), list(backbone.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))
        for i in [23,25,27]:
            features[i].dilation = (2, 2)
            features[i].padding = (2, 2)
        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        backbone = nn.Sequential(*([features[i] for i in range(len(features))] + [fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        return_layers = {'4': 'low_fea', '32': 'out'}
        
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        feas = self.backbone(x)
        out = feas['out']
        return out
        
class resnet_feature_extractor(nn.Module):
    def __init__(self, backbone_name, crossnorm=False, uncertainty=0.0, pos=[], pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(resnet_feature_extractor, self).__init__()
        bn_layer = nn.BatchNorm2d
        if freeze_bn:
            bn_layer = FrozenBatchNorm2d
        backbone = resnet.__dict__[backbone_name](
                pretrained=pretrained_backbone, uncertainty=uncertainty, pos=pos,
                replace_stride_with_dilation=[False, True, True], pretrained_weights=pretrained_weights, norm_layer=bn_layer)
        return_layers = {'layer4': 'out'}
        if aux:
            return_layers['layer3'] = 'aux'
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    
    def forward(self, x):
        out = self.backbone(x)['out']
        return out