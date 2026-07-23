"""LDMDetDetector — mmdet BaseDetector 桥接

将 ldmdet 纯 PyTorch 核心包装为 mmdet 兼容的检测器。
"""

import copy
import inspect
from typing import Dict, List, Tuple

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
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bbox_head = self._build_head(bbox_head)

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
