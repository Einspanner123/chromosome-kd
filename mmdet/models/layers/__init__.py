# Copyright (c) OpenMMLab. All rights reserved.
from .activations import SiLU
from .bbox_nms import fast_nms, multiclass_nms
from .brick_wrappers import (
    AdaptiveAvgPool2d,
    FrozenBatchNorm2d,
    adaptive_avg_pool2d,
)
from .conv_upsample import ConvUpsample
from .csp_layer import CSPLayer
from .dropblock import DropBlock
from .ema import ExpMomentumEMA
from .inverted_residual import InvertedResidual
from .matrix_nms import mask_matrix_nms
from .msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from .normed_predictor import NormedConv2d, NormedLinear
from .pixel_decoder import PixelDecoder, TransformerEncoderPixelDecoder
from .positional_encoding import (
    LearnedPositionalEncoding,
    SinePositionalEncoding,
    SinePositionalEncoding3D,
)
from .res_layer import ResLayer, SimplifiedBasicBlock
from .se_layer import ChannelAttention, DyReLU, SELayer

# yapf: disable
from .transformer import (
                             MLP,
                             AdaptivePadding,
                             CdnQueryGenerator,
                             ConditionalAttention,
                             ConditionalDetrTransformerDecoder,
                             ConditionalDetrTransformerDecoderLayer,
                             DABDetrTransformerDecoder,
                             DABDetrTransformerDecoderLayer,
                             DABDetrTransformerEncoder,
                             DDQTransformerDecoder,
                             DeformableDetrTransformerDecoder,
                             DeformableDetrTransformerDecoderLayer,
                             DeformableDetrTransformerEncoder,
                             DeformableDetrTransformerEncoderLayer,
                             DetrTransformerDecoder,
                             DetrTransformerDecoderLayer,
                             DetrTransformerEncoder,
                             DetrTransformerEncoderLayer,
                             DinoTransformerDecoder,
                             DynamicConv,
                             Mask2FormerTransformerDecoder,
                             Mask2FormerTransformerDecoderLayer,
                             Mask2FormerTransformerEncoder,
                             PatchEmbed,
                             PatchMerging,
                             coordinate_to_encoding,
                             inverse_sigmoid,
                             nchw_to_nlc,
                             nlc_to_nchw,
)

# yapf: enable

__all__ = [
    'MLP',
    'AdaptiveAvgPool2d',
    'AdaptivePadding',
    'CSPLayer',
    'CdnQueryGenerator',
    'ChannelAttention',
    'ConditionalAttention',
    'ConditionalDetrTransformerDecoder',
    'ConditionalDetrTransformerDecoderLayer',
    'ConvUpsample',
    'DABDetrTransformerDecoder',
    'DABDetrTransformerDecoderLayer',
    'DABDetrTransformerEncoder',
    'DDQTransformerDecoder',
    'DeformableDetrTransformerDecoder',
    'DeformableDetrTransformerDecoderLayer',
    'DeformableDetrTransformerEncoder',
    'DeformableDetrTransformerEncoderLayer',
    'DetrTransformerDecoder',
    'DetrTransformerDecoderLayer',
    'DetrTransformerEncoder',
    'DetrTransformerEncoderLayer',
    'DinoTransformerDecoder',
    'DropBlock',
    'DyReLU',
    'DynamicConv',
    'ExpMomentumEMA',
    'FrozenBatchNorm2d',
    'InvertedResidual',
    'LearnedPositionalEncoding',
    'MSDeformAttnPixelDecoder',
    'Mask2FormerTransformerDecoder',
    'Mask2FormerTransformerDecoderLayer',
    'Mask2FormerTransformerEncoder',
    'NormedConv2d',
    'NormedLinear',
    'PatchEmbed',
    'PatchMerging',
    'PixelDecoder',
    'ResLayer',
    'SELayer',
    'SiLU',
    'SimplifiedBasicBlock',
    'SinePositionalEncoding',
    'SinePositionalEncoding3D',
    'TransformerEncoderPixelDecoder',
    'adaptive_avg_pool2d',
    'coordinate_to_encoding',
    'fast_nms',
    'inverse_sigmoid',
    'mask_matrix_nms',
    'multiclass_nms',
    'nchw_to_nlc',
    'nlc_to_nchw',
]
