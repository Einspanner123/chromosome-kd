"""LDMDetDetector — mmdet BaseDetector 桥接

将 ldmdet 纯 PyTorch 核心包装为 mmdet 兼容的检测器。
"""

import copy
import inspect
import logging
from typing import Dict, List, Optional, Tuple

import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.structures import InstanceData as MMInstanceData

from ldmdet.core import DiffusionDetHead, SingleDiffusionDetHead, SingleRoIExtractor
from ldmdet.coupling import build_coupling
from ldmdet.criterion import (
    BBoxL1Cost,
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    FocalLossCost,
    GIoULoss,
    IoUCost,
    L1Loss,
    SeesawLoss,
)
from ldmdet.data.structures import ImageMeta

logger = logging.getLogger(__name__)


@MODELS.register_module(name='LDMDetV2', force=True)
@MODELS.register_module(name='LDMDet', force=True)
class LDMDetDetector(BaseDetector):
    """LDMDet 检测器 — mmdet BaseDetector 兼容包装。

    backbone/neck 通过 mmdet 构建，bbox_head 通过 ldmdet 纯 PyTorch 构建。

    Head Distillation v2: 可选 teacher_config/teacher_checkpoint 参数,
    构建 Teacher (H=6, A4 冻结) 并注入 bbox_head, 同时冻结 backbone/neck。
    """

    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        # Head Distillation v2 (详见 REFLOW_HEAD_DISTILL_IMPL_PLAN.md §2.5)
        teacher_config: OptConfigType = None,
        teacher_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bbox_head = self._build_head(bbox_head)

        # 方案A: 保存 teacher_checkpoint 路径, 供 init_weights 加载 backbone/neck
        # (不用 load_from — 会覆盖 init_student_from_teacher 的 head 映射权重)
        self._teacher_checkpoint = teacher_checkpoint

        # Head Distillation v2: 冻结 backbone + neck
        # v2 修正 #1: 蒸馏聚焦 head, 减少 param 量, 避免 backbone 漂移
        # 方案A: freeze_backbone=False 时不冻结, backbone 从 A4 加载并微调
        if self.bbox_head.freeze_backbone:
            self._freeze_backbone()

        # Head Distillation v2: 构建 + 注入 Teacher
        if self.bbox_head.use_distillation:
            teacher_head = self._build_teacher(
                bbox_head, teacher_config, teacher_checkpoint
            )
            self.bbox_head.set_teacher(teacher_head)
            # 从 Teacher 初始化 Student (headwise 映射, 仅当 checkpoint 已加载)
            if teacher_checkpoint is not None:
                self.bbox_head.init_student_from_teacher()

    def _build_head(self, cfg: ConfigType) -> DiffusionDetHead:
        """从配置构建 ldmdet DiffusionDetHead"""
        # deepcopy 避免修改嵌套 dict (coupling/single_head 等含 type 键)
        cfg = copy.deepcopy(cfg)

        # 1. 构建耦合策略
        coupling_cfg = cfg.pop('coupling', None)
        # 兼容旧配置的 ot_coupling + ot_* 参数
        if coupling_cfg is None and cfg.pop('ot_coupling', False):
            coupling_cfg = {
                'type': cfg.pop('ot_coupling_type', 'sinkhorn_stochastic'),
                'epsilon': cfg.pop('ot_epsilon', 5.0),
                'num_iters': cfg.pop('ot_num_iters', 20),
            }
        if coupling_cfg is not None:
            name = coupling_cfg.pop('type')
            coupling = build_coupling(name, **coupling_cfg)
        else:
            coupling = build_coupling('random')

        # 2. 构建 single_head (支持 ShapeAttention 局部形状注意力)
        sh_cfg = cfg.pop('single_head')
        sh_cfg.pop('type', None)

        # 方向 C1 / M1: 可选的局部形状注意力 / 形态感知 RoI 编码器
        # 通过 MODELS 注册表构建, 支持任何已注册模块 (如 MorphologyAwareRoIEncoder)
        shape_attention = None
        if 'shape_attention' in sh_cfg:
            sa_cfg = sh_cfg.pop('shape_attention')
            shape_attention = MODELS.build(sa_cfg)

        single_head = SingleDiffusionDetHead(
            shape_attention=shape_attention, **sh_cfg
        )

        # 3. 构建 roi_extractor
        re_cfg = cfg.pop('roi_extractor', {})
        re_cfg.pop('type', None)
        roi_extractor = SingleRoIExtractor(**re_cfg)

        # 4. 构建 criterion
        criterion_cfg = cfg.pop('criterion', None)
        criterion = None
        if criterion_cfg is not None:
            criterion = self._build_criterion(criterion_cfg)

        # 5. 构建 head — 只传 DiffusionDetHead 接受的参数
        valid_params = set(inspect.signature(DiffusionDetHead.__init__).parameters.keys())
        cfg = {k: v for k, v in cfg.items() if k in valid_params}
        head = DiffusionDetHead(
            **cfg,
            single_head=single_head,
            roi_extractor=roi_extractor,
            criterion=criterion,
            coupling=coupling,
        )
        return head

    def _build_criterion(self, cfg: Dict) -> DiffusionDetCriterion:
        """从配置构建 criterion"""
        cfg = cfg.copy()
        cfg.pop('type', None)
        assigner_cfg = cfg.pop('assigner', cfg.pop('matcher', {}))
        assigner_cfg.pop('type', None)

        # 构建 match costs
        match_costs = []
        for cost_cfg in assigner_cfg.pop('match_costs', []):
            cost_type = cost_cfg.pop('type')
            cost_type = cost_type.replace('PurePyTorch', '')
            cost_map = {
                'FocalLossCost': FocalLossCost,
                'BBoxL1Cost': BBoxL1Cost,
                'IoUCost': IoUCost,
            }
            match_costs.append(cost_map[cost_type](**cost_cfg))

        matcher = DiffusionDetMatcher(match_costs=match_costs, **assigner_cfg)

        loss_cls_cfg = cfg.pop('loss_cls')
        loss_cls_type = loss_cls_cfg.pop('type').replace('PurePyTorch', '')
        loss_cls_map = {
            'FocalLoss': FocalLoss,
            'SeesawLoss': SeesawLoss,
        }
        loss_cls = loss_cls_map[loss_cls_type](**loss_cls_cfg)

        loss_bbox_cfg = cfg.pop('loss_bbox')
        loss_bbox_cfg.pop('type', None)
        loss_bbox = L1Loss(**loss_bbox_cfg)

        loss_giou_cfg = cfg.pop('loss_giou')
        loss_giou_cfg.pop('type', None)
        loss_giou = GIoULoss(**loss_giou_cfg)

        return DiffusionDetCriterion(
            **cfg, matcher=matcher, loss_cls=loss_cls, loss_bbox=loss_bbox, loss_giou=loss_giou
        )

    # ================================================================
    # Head Distillation v2: Teacher 构建 + 注入
    # ================================================================

    def _freeze_backbone(self):
        """冻结 backbone + neck 参数 (Head Distillation v2 修正 #1)

        仅 requires_grad_(False) 不够 — BN running stats 仍会在 train 模式下更新。
        需配合 train() 重写将 backbone/neck 置于 eval 模式。
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if self.neck is not None:
            for p in self.neck.parameters():
                p.requires_grad_(False)

    def _build_teacher(
        self,
        student_cfg: ConfigType,
        teacher_config: OptConfigType,
        teacher_checkpoint: Optional[str],
    ) -> DiffusionDetHead:
        """构建 Teacher head 并加载 checkpoint (Head Distillation v2)

        Args:
            student_cfg: Student bbox_head 配置 (用于 auto-construct)
            teacher_config: Teacher bbox_head 配置; None 时自动从 student 配置构建
            teacher_checkpoint: Teacher checkpoint 路径 (A4 完整 detector checkpoint)

        Returns:
            Teacher DiffusionDetHead 实例 (已加载权重, eval 模式)
        """
        if teacher_config is None:
            # Auto-construct: Teacher = A4 架构 (H=6, 无蒸馏)
            # Student 继承自 A4, deepcopy 后恢复 num_heads=6 即得 A4 配置
            teacher_config = copy.deepcopy(student_cfg)
            teacher_config['num_heads'] = 6
            teacher_config['use_distillation'] = False
            teacher_config['freeze_backbone'] = False
            # 清理蒸馏专用参数 (Teacher 不使用)
            teacher_config.pop('distill_lambda', None)
            teacher_config.pop('distill_head_map', None)
            teacher_config.pop('deep_supervision_aux_weight', None)
            # Teacher 仅 forward (不计算 loss), 无需构建 criterion
            teacher_config.pop('criterion', None)
            logger.info(
                'Head Distillation: teacher_config 未提供, '
                '自动从 student 配置构建 (num_heads=6)'
            )

        teacher_head = self._build_head(teacher_config)

        if teacher_checkpoint is not None:
            self._load_teacher_checkpoint(teacher_head, teacher_checkpoint)
        else:
            logger.warning(
                'Head Distillation: teacher_checkpoint 未提供, '
                'Teacher 使用随机权重 (仅适用于单元测试)'
            )

        return teacher_head

    def _load_teacher_checkpoint(
        self,
        teacher_head: DiffusionDetHead,
        checkpoint_path: str,
    ):
        """从完整 detector checkpoint 加载 bbox_head 权重到 teacher_head

        Checkpoint 是 LDMDetDetector 完整状态 (backbone.* + neck.* + bbox_head.*),
        仅提取 bbox_head.* 前缀的权重并去除前缀后加载。
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        prefix = 'bbox_head.'
        head_state = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                head_state[k[len(prefix):]] = v

        if not head_state:
            logger.warning(
                f'Checkpoint {checkpoint_path} 中未找到 "{prefix}" 前缀, '
                '尝试直接加载 (可能不匹配)'
            )
            head_state = state_dict

        missing, unexpected = teacher_head.load_state_dict(
            head_state, strict=False
        )
        if missing:
            logger.warning(
                f'Teacher head missing keys ({len(missing)}): {missing[:5]}'
            )
        if unexpected:
            logger.warning(
                f'Teacher head unexpected keys ({len(unexpected)}): '
                f'{unexpected[:5]}'
            )
        logger.info(
            f'Head Distillation: Teacher checkpoint 已加载: {checkpoint_path}'
        )

    def train(self, mode: bool = True):
        """重写 train(): 冻结的 backbone/neck 保持 eval 模式 (BN 不更新)

        Head Distillation v2 的 Teacher 通过 object.__setattr__ 存储,
        不是 nn.Module 子模块, 不受 super().train() 影响, 始终保持 eval。

        方案A (freeze_backbone=False): backbone 参与训练, 但 backbone 配置中
        frozen_stages=1 仍冻结 stem, norm_eval=True 仍使 BN 处于 eval — 标准
        fine-tuning 实践, 无需在此特殊处理。
        """
        super().train(mode)
        if self.bbox_head.freeze_backbone:
            self.backbone.eval()
            if self.neck is not None:
                self.neck.eval()
        return self

    def init_weights(self):
        """重写 init_weights: 方案A — 从 A4 checkpoint 加载 backbone/neck 权重

        时序 (mmengine Runner):
          1. __init__: 构建 backbone/neck/head + 注入 Teacher + init_student_from_teacher
             (Student head ← Teacher head 映射, 此时 backbone 仍为默认初始化)
          2. init_weights (本方法):
             a. super().init_weights(): 加载 ImageNet 预训练 backbone (init_cfg),
                不触及 bbox_head (无 init_cfg) → head 映射权重保留
             b. _load_backbone_from_checkpoint: 用 A4 的 backbone/neck 覆盖 ImageNet
                权重, 使 Student 特征空间与 Teacher head 对齐 (修复 root cause)
          3. load_from (若设置): 会覆盖全部 state_dict — 方案A **不使用 load_from**,
             避免破坏 head 映射 (A4 head 1/2 会错误覆盖 Student head 1/2)

        root cause: v2 freeze_backbone=True 使 Student backbone 停在 ImageNet,
        而 Teacher head 在 A4 (染色体训练) 特征上学习 → 特征分布不匹配, mAP 0.711。
        方案A: 加载 A4 backbone + 解冻, Student/Teacher 共享 A4 特征空间。
        """
        super().init_weights()
        # 方案A: 蒸馏模式 + 未冻结 backbone + 有 teacher_checkpoint 时, 加载 A4 backbone/neck
        if (self.bbox_head.use_distillation
                and not self.bbox_head.freeze_backbone
                and self._teacher_checkpoint is not None):
            self._load_backbone_from_checkpoint(self._teacher_checkpoint)

    def _load_backbone_from_checkpoint(self, checkpoint_path: str):
        """从完整 detector checkpoint 加载 backbone/neck 权重 (方案A)

        与 _load_teacher_checkpoint 不同: 这里加载到 Student 自身的 backbone/neck,
        而非 Teacher head。仅提取 backbone.*/neck.* 前缀, 不触及 bbox_head.*
        (head 映射权重由 __init__ 的 init_student_from_teacher 设置, 必须保留)。

        Args:
            checkpoint_path: A4 完整 detector checkpoint 路径
                (含 backbone.* + neck.* + bbox_head.* 全部状态)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # 提取 backbone 权重 (去前缀)
        backbone_sd = {
            k[len('backbone.'):]: v for k, v in state_dict.items()
            if k.startswith('backbone.')
        }
        # 提取 neck 权重 (去前缀)
        neck_sd = {
            k[len('neck.'):]: v for k, v in state_dict.items()
            if k.startswith('neck.')
        }

        if not backbone_sd:
            logger.warning(
                f'方案A: checkpoint {checkpoint_path} 中未找到 "backbone." 前缀, '
                'backbone 保持 ImageNet 预训练权重 (特征不匹配风险!)'
            )
        else:
            missing, unexpected = self.backbone.load_state_dict(
                backbone_sd, strict=False
            )
            if missing:
                logger.warning(
                    f'方案A: backbone missing keys ({len(missing)}): {missing[:5]}'
                )
            if unexpected:
                logger.warning(
                    f'方案A: backbone unexpected keys ({len(unexpected)}): '
                    f'{unexpected[:5]}'
                )
            logger.info(
                f'方案A: backbone 权重已从 {checkpoint_path} 加载 '
                f'({len(backbone_sd)} keys)'
            )

        if neck_sd and self.neck is not None:
            missing, unexpected = self.neck.load_state_dict(
                neck_sd, strict=False
            )
            if missing:
                logger.warning(
                    f'方案A: neck missing keys ({len(missing)}): {missing[:5]}'
                )
            if unexpected:
                logger.warning(
                    f'方案A: neck unexpected keys ({len(unexpected)}): '
                    f'{unexpected[:5]}'
                )
            logger.info(
                f'方案A: neck 权重已从 {checkpoint_path} 加载 ({len(neck_sd)} keys)'
            )
        elif neck_sd and self.neck is None:
            logger.warning('方案A: checkpoint 含 neck 权重但模型无 neck, 跳过')

    def extract_feat(self, batch_inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.backbone(batch_inputs)
        if self.neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples: List[DetDataSample]) -> dict:
        x = self.extract_feat(batch_inputs)
        img_metas = []
        gt_bboxes = []
        gt_labels = []
        for ds in batch_data_samples:
            img_metas.append(ImageMeta(
                img_shape=ds.metainfo['img_shape'],
                pad_shape=ds.metainfo.get('pad_shape'),
                ori_shape=ds.metainfo.get('ori_shape'),
                scale_factor=ds.metainfo.get('scale_factor'),
                img_id=ds.metainfo.get('img_id'),
            ))
            gt_bboxes.append(ds.gt_instances.bboxes)
            gt_labels.append(ds.gt_instances.labels)
        return self.bbox_head.loss(x, img_metas, gt_bboxes, gt_labels)

    def predict(
        self,
        batch_inputs: torch.Tensor,
        batch_data_samples: List[DetDataSample],
        rescale: bool = True,
        return_trajectory: bool = False,
    ) -> List[DetDataSample]:
        x = self.extract_feat(batch_inputs)
        img_metas = []
        for ds in batch_data_samples:
            img_metas.append(ImageMeta(
                img_shape=ds.metainfo['img_shape'],
                pad_shape=ds.metainfo.get('pad_shape'),
                ori_shape=ds.metainfo.get('ori_shape'),
                scale_factor=ds.metainfo.get('scale_factor'),
                img_id=ds.metainfo.get('img_id'),
            ))

        results_list = self.bbox_head.predict(x, img_metas, rescale=rescale)

        for i, ds in enumerate(batch_data_samples):
            res = results_list[i]
            pred = MMInstanceData()
            pred.bboxes = res.bboxes
            pred.scores = res.scores
            pred.labels = res.labels
            ds.pred_instances = pred
        return batch_data_samples

    def _forward(self, batch_inputs, batch_data_samples):
        x = self.extract_feat(batch_inputs)
        bs = len(batch_data_samples)
        t = x[0].new_zeros((bs,), dtype=torch.float32)
        noise = torch.randn(bs, self.bbox_head.num_proposals, 4, device=x[0].device)
        cls_logits, pred_bboxes, _ = self.bbox_head(x, noise, t)
        return cls_logits, pred_bboxes
