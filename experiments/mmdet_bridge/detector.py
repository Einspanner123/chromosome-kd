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

from ldmdet.core import (
    DiffusionDetHead,
    SingleDiffusionDetHead,
    SingleRoIExtractor,
)
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
)
from ldmdet.data.structures import ImageMeta

logger = logging.getLogger(__name__)


@MODELS.register_module(name='LDMDetV2', force=True)
@MODELS.register_module(name='LDMDet', force=True)
class LDMDetDetector(BaseDetector):
    """LDMDet 检测器 — mmdet BaseDetector 兼容包装。

    backbone/neck 通过 mmdet 构建，bbox_head 通过 ldmdet 纯 PyTorch 构建。
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
        teacher_config: OptConfigType = None,
        teacher_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg
        )

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bbox_head = self._build_head(bbox_head)
        self._teacher_checkpoint = teacher_checkpoint
        if self.bbox_head.use_distillation:
            teacher_head = self._build_teacher(
                bbox_head, teacher_config, teacher_checkpoint
            )
            self.bbox_head.set_teacher(teacher_head)
            if teacher_checkpoint is not None:
                self.bbox_head.init_student_from_teacher()
        if self.bbox_head.freeze_backbone:
            self._freeze_backbone_and_neck()
        if self.bbox_head.quality_only_training:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            if self.neck is not None:
                for parameter in self.neck.parameters():
                    parameter.requires_grad_(False)

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

        # 2. 构建 single_head
        sh_cfg = cfg.pop('single_head')
        sh_cfg.pop('type', None)

        single_head = SingleDiffusionDetHead(**sh_cfg)

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
        valid_params = set(
            inspect.signature(DiffusionDetHead.__init__).parameters.keys()
        )
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
        }
        loss_cls = loss_cls_map[loss_cls_type](**loss_cls_cfg)

        loss_bbox_cfg = cfg.pop('loss_bbox')
        loss_bbox_cfg.pop('type', None)
        loss_bbox = L1Loss(**loss_bbox_cfg)

        loss_giou_cfg = cfg.pop('loss_giou')
        loss_giou_cfg.pop('type', None)
        loss_giou = GIoULoss(**loss_giou_cfg)

        return DiffusionDetCriterion(
            **cfg,
            matcher=matcher,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_giou=loss_giou,
        )

    # ================================================================
    # Parent-matched cascade-head distillation
    # ================================================================

    def _build_teacher(
        self,
        student_cfg: ConfigType,
        teacher_config: OptConfigType,
        teacher_checkpoint: Optional[str],
    ) -> DiffusionDetHead:
        """Build the full-cascade teacher and load only its head tensors."""
        if teacher_config is None:
            teacher_config = copy.deepcopy(student_cfg)
            teacher_config['num_heads'] = 6
            for key in (
                'use_distillation',
                'distill_lambda',
                'distill_head_map',
                'deep_supervision_aux_weight',
                'freeze_backbone',
            ):
                teacher_config.pop(key, None)
            teacher_config.pop('criterion', None)
        teacher = self._build_head(teacher_config)
        if teacher_checkpoint is not None:
            self._load_component_checkpoint(
                teacher,
                teacher_checkpoint,
                prefix='bbox_head.',
                component_name='teacher bbox_head',
            )
        else:
            logger.warning(
                'Head-distillation teacher has random weights; this is '
                'permitted only for configuration/unit tests'
            )
        return teacher

    @staticmethod
    def _checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state, dict):
            raise TypeError(
                f'{checkpoint_path} does not contain a state dictionary'
            )
        return state

    def _load_component_checkpoint(
        self,
        module: torch.nn.Module,
        checkpoint_path: str,
        prefix: str,
        component_name: str,
    ) -> None:
        state = self._checkpoint_state(checkpoint_path)
        component = {
            name[len(prefix) :]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        if not component:
            raise RuntimeError(
                f'{checkpoint_path} contains no {prefix!r} tensors'
            )
        missing, unexpected = module.load_state_dict(component, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f'{component_name} checkpoint mismatch: '
                f'missing={missing[:8]}, unexpected={unexpected[:8]}'
            )
        logger.info(
            'Loaded %d %s tensors from %s',
            len(component),
            component_name,
            checkpoint_path,
        )

    def _freeze_backbone_and_neck(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if self.neck is not None:
            for parameter in self.neck.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.bbox_head.freeze_backbone:
            self.backbone.eval()
            if self.neck is not None:
                self.neck.eval()
        if self.bbox_head._teacher is not None:
            self.bbox_head._teacher.eval()
        return self

    def init_weights(self):
        """Initialize student features from its parent without overwriting heads."""
        super().init_weights()
        if (
            self.bbox_head.use_distillation
            and self._teacher_checkpoint is not None
        ):
            self._load_component_checkpoint(
                self.backbone,
                self._teacher_checkpoint,
                prefix='backbone.',
                component_name='student backbone',
            )
            if self.neck is not None:
                self._load_component_checkpoint(
                    self.neck,
                    self._teacher_checkpoint,
                    prefix='neck.',
                    component_name='student neck',
                )

    def extract_feat(
        self, batch_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        x = self.backbone(batch_inputs)
        if self.neck:
            x = self.neck(x)
        return x

    def loss(
        self,
        batch_inputs: torch.Tensor,
        batch_data_samples: List[DetDataSample],
    ) -> dict:
        x = self.extract_feat(batch_inputs)
        img_metas = []
        gt_bboxes = []
        gt_labels = []
        for ds in batch_data_samples:
            img_metas.append(
                ImageMeta(
                    img_shape=ds.metainfo['img_shape'],
                    pad_shape=ds.metainfo.get('pad_shape'),
                    ori_shape=ds.metainfo.get('ori_shape'),
                    scale_factor=ds.metainfo.get('scale_factor'),
                    img_id=ds.metainfo.get('img_id'),
                )
            )
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
            img_metas.append(
                ImageMeta(
                    img_shape=ds.metainfo['img_shape'],
                    pad_shape=ds.metainfo.get('pad_shape'),
                    ori_shape=ds.metainfo.get('ori_shape'),
                    scale_factor=ds.metainfo.get('scale_factor'),
                    img_id=ds.metainfo.get('img_id'),
                )
            )

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
        noise = torch.randn(
            bs, self.bbox_head.num_proposals, 4, device=x[0].device
        )
        cls_logits, pred_bboxes, _ = self.bbox_head(x, noise, t)
        return cls_logits, pred_bboxes
