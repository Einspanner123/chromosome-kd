"""DiffuDETR models — 扩散检测器核心模块

注册 DiffuDETRDetector 到 mmdet MODELS 注册表.
导入后可通过 custom_imports 使用 'DiffuDETR' 类型.
"""

from .criterion import (
    HungarianMatcher,
    SetCriterion,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
    sigmoid_focal_loss,
)
from .diffudetr_detector import DiffuDETRDetector
from .diffudetr_head import DiffuDETRHead
from .diffusion_scheduler import (
    DiffusionScheduler,
    cosine_beta_schedule,
    extract,
    timestep_embedding,
)
from .timestep_block import BBoxEmbed, ClassEmbed, TimeStepBlock
from .transformer import (
    FFN,
    MLP,
    DiffuDETRDecoderLayer,
    DiffuDETRTransformer,
    MultiheadAttention,
    get_sine_pos_embed,
)

__all__ = [
    'FFN',
    'MLP',
    'BBoxEmbed',
    'ClassEmbed',
    'DiffuDETRDecoderLayer',
    'DiffuDETRDetector',
    'DiffuDETRHead',
    'DiffuDETRTransformer',
    'DiffusionScheduler',
    'HungarianMatcher',
    'MultiheadAttention',
    'SetCriterion',
    'TimeStepBlock',
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'cosine_beta_schedule',
    'extract',
    'generalized_box_iou',
    'get_sine_pos_embed',
    'sigmoid_focal_loss',
    'timestep_embedding',
]
