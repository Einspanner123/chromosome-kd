# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet

__all__ = [
    'SSDVGG',
    'CSPDarknet',
    'CSPNeXt',
    'Darknet',
    'DetectoRS_ResNeXt',
    'DetectoRS_ResNet',
    'EfficientNet',
    'HRNet',
    'HourglassNet',
    'MobileNetV2',
    'PyramidVisionTransformer',
    'PyramidVisionTransformerV2',
    'RegNet',
    'Res2Net',
    'ResNeSt',
    'ResNeXt',
    'ResNet',
    'ResNetV1d',
    'SwinTransformer',
    'TridentResNet',
]
