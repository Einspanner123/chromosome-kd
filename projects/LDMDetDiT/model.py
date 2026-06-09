import inspect
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

# torch.set_float32_matmul_precision('high')
# if torch.backends.cudnn.is_available():
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.allow_tf32 = True
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .mods.dit_head import DiTDiffusionDetHead
from .mods.dit_single_head import DiTSingleHead
from .mods.modules import SpatialTuningAdapter
from .mods.loss import (
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    FocalLossCost,
    GIoULoss,
    IoUCost,
    L1Loss,
    RelativeL1Cost,
)
from .mods.structures import ImageMeta

# 注册所有组件到 MODELS 注册表，以便可以通过配置文件构建
# force=True 避免与 LDMDet 同时 import 时的注册冲突
MODELS.register_module(
    name='PurePyTorchDiffusionDetCriterion', module=DiffusionDetCriterion,
    force=True,
)
MODELS.register_module(
    name='PurePyTorchDiffusionDetMatcher', module=DiffusionDetMatcher,
    force=True,
)
MODELS.register_module(name='PurePyTorchFocalLoss', module=FocalLoss, force=True)
MODELS.register_module(name='PurePyTorchL1Loss', module=L1Loss, force=True)
MODELS.register_module(name='PurePyTorchGIoULoss', module=GIoULoss, force=True)
MODELS.register_module(name='PurePyTorchFocalLossCost', module=FocalLossCost, force=True)
MODELS.register_module(name='PurePyTorchIoUCost', module=IoUCost, force=True)
MODELS.register_module(name='PurePyTorchRelativeL1Cost', module=RelativeL1Cost, force=True)

MODELS.register_module(name='DiTDiffusionDetHead', module=DiTDiffusionDetHead, force=True)
MODELS.register_module(name='DiTSingleHead', module=DiTSingleHead, force=True)


@MODELS.register_module(force=True)
@MODELS.register_module(name='LDMDetDiT', force=True)
class PurePyTorchDiffusionDet(BaseDetector):
    """
    使用纯 PyTorch 实现的 DiffusionDet 包装类，兼容 MMDetection 3.x 框架。
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
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg
        )

        # 构建 backbone
        if (
            isinstance(backbone, dict) and backbone.get('type') == 'timm_model'
        ) or (isinstance(backbone, dict) and backbone.get('type') == 'timm'):
            import timm

            backbone_cfg = backbone.copy()
            model_name = backbone_cfg.pop('model_name')
            backbone_cfg.pop('type', None)
            self.backbone = timm.create_model(model_name, **backbone_cfg)
        elif isinstance(backbone, dict) and backbone.get('type') in [
            'ConvNeXtV2',
            'projects.LDMDetDiT.mods.convnextv2.ConvNeXtV2',
        ]:
            from .mods.convnextv2 import ConvNeXtV2

            backbone_cfg = backbone.copy()
            backbone_cfg.pop('type')
            self.backbone = ConvNeXtV2(
                **self._filter_kwargs(ConvNeXtV2, backbone_cfg)
            )
        else:
            self.backbone = MODELS.build(backbone)

        if neck is not None:
            # 自动处理模块路径
            if (
                isinstance(neck, dict)
                and neck.get('type') == 'PurePyTorchSimpleFeatureFusion'
            ):
                from .mods.modules import PurePyTorchSimpleFeatureFusion

                neck_cfg = neck.copy()
                neck_cfg.pop('type')
                self.neck = PurePyTorchSimpleFeatureFusion(**neck_cfg)
            elif (
                isinstance(neck, dict)
                and neck.get('type') == 'SpatialTuningAdapter'
            ):
                from .mods.modules import SpatialTuningAdapter

                neck_cfg = neck.copy()
                neck_cfg.pop('type')
                self.neck = SpatialTuningAdapter(**neck_cfg)
            else:
                self.neck = MODELS.build(neck)
        else:
            self.neck = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 构建 bbox_head
        # 注意：这里的 bbox_head 配置需要适配纯 PyTorch 版本的 DiffusionDetHead
        self.bbox_head = self._build_bbox_head(bbox_head)

    def _filter_kwargs(self, obj_cls: Any, cfg: Dict) -> Dict:
        """根据类构造函数签名过滤配置字典，避免传入不支持的参数"""
        if not inspect.isclass(obj_cls):
            return cfg

        sig = inspect.signature(obj_cls.__init__)
        params = sig.parameters
        # 如果类定义了 **kwargs，则不进行过滤
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            return cfg

        valid_cfg = {
            k: v for k, v in cfg.items() if k in params or k == 'type'
        }
        return valid_cfg

    def _build_bbox_head(self, cfg: ConfigType) -> nn.Module:
        """构建检测头及其子组件 (仅 DiT 版本)"""
        cfg_copy = cfg.copy()

        head_type = cfg_copy.get('type', 'DiTDiffusionDetHead')

        # 1. 构建 single_head
        single_head_cfg = cfg_copy.pop('single_head')
        if isinstance(single_head_cfg, dict):
            if 'type' not in single_head_cfg:
                single_head_cfg['type'] = 'DiTSingleHead'

            obj_cls = MODELS.get(single_head_cfg['type'])
            single_head = MODELS.build(
                self._filter_kwargs(obj_cls, single_head_cfg)
            )
        else:
            single_head = single_head_cfg

        # 2. 构建 criterion
        criterion_cfg = cfg_copy.pop('criterion', None)
        criterion = None
        if criterion_cfg is not None:
            criterion = self._build_criterion(criterion_cfg)

        # 3. 构建 Head
        if 'type' not in cfg_copy:
            cfg_copy['type'] = head_type

        torch_compile = cfg_copy.pop('torch_compile', False)

        cfg_copy.update(dict(
            single_head=single_head,
            criterion=criterion,
        ))

        obj_cls = MODELS.get(cfg_copy['type'])
        head = MODELS.build(self._filter_kwargs(obj_cls, cfg_copy))

        if torch_compile and hasattr(torch, 'compile'):
            head.forward = torch.compile(head.forward, dynamic=True)

        return head

    def _build_criterion(self, cfg: ConfigType) -> nn.Module:
        """构建损失函数组件"""
        cfg_copy = cfg.copy()

        # 构建 Matcher
        assigner_cfg = cfg_copy.pop('assigner')
        if 'type' not in assigner_cfg:
            assigner_cfg['type'] = 'PurePyTorchDiffusionDetMatcher'

        # 处理 match_costs
        if 'match_costs' in assigner_cfg:
            costs = []
            for cost_cfg in assigner_cfg['match_costs']:
                if isinstance(cost_cfg, dict):
                    # 映射旧名称到 PurePyTorch 前缀名称
                    type_map = {
                        'FocalLossCost': 'PurePyTorchFocalLossCost',
                        'BBoxL1Cost': 'PurePyTorchBBoxL1Cost',
                        'IoUCost': 'PurePyTorchIoUCost',
                        'RelativeL1Cost': 'PurePyTorchRelativeL1Cost',
                    }
                    if 'type' in cost_cfg:
                        cost_cfg['type'] = type_map.get(
                            cost_cfg['type'], cost_cfg['type']
                        )

                    obj_cls = MODELS.get(cost_cfg['type'])
                    costs.append(
                        MODELS.build(self._filter_kwargs(obj_cls, cost_cfg))
                    )
                else:
                    costs.append(cost_cfg)
            assigner_cfg['match_costs'] = costs

        obj_cls = MODELS.get(assigner_cfg['type'])
        matcher = MODELS.build(self._filter_kwargs(obj_cls, assigner_cfg))

        # 构建各类 Loss
        loss_cls_cfg = cfg_copy.pop('loss_cls')
        obj_cls = MODELS.get(loss_cls_cfg['type'])
        loss_cls = MODELS.build(self._filter_kwargs(obj_cls, loss_cls_cfg))

        loss_giou_cfg = cfg_copy.pop('loss_giou')
        obj_cls = MODELS.get(loss_giou_cfg['type'])
        loss_giou = MODELS.build(self._filter_kwargs(obj_cls, loss_giou_cfg))

        # 实例化 Criterion
        if 'type' not in cfg_copy:
            cfg_copy['type'] = 'PurePyTorchDiffusionDetCriterion'

        cfg_copy.update(
            dict(
                matcher=matcher,
                loss_cls=loss_cls,
                loss_giou=loss_giou,
            )
        )

        obj_cls = MODELS.get(cfg_copy['type'])
        return MODELS.build(self._filter_kwargs(obj_cls, cfg_copy))

    def extract_feat(self, batch_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """提取特征"""
        x = self.backbone(batch_inputs)
        if self.neck:
            if isinstance(self.neck, SpatialTuningAdapter):
                x = self.neck(x, batch_inputs)
            else:
                x = self.neck(x)
        return x

    def loss(
        self,
        batch_inputs: torch.Tensor,
        batch_data_samples: List[DetDataSample],
    ) -> dict:
        """训练模式的前向传播"""
        # 1. 提取特征
        x = self.extract_feat(batch_inputs)

        # 2. 准备数据格式
        img_metas = []
        gt_bboxes = []
        gt_labels = []

        for data_sample in batch_data_samples:
            # 转换为纯 PyTorch 结构的数据类
            meta = ImageMeta(
                img_shape=data_sample.metainfo['img_shape'],
                pad_shape=data_sample.metainfo.get('pad_shape'),
                ori_shape=data_sample.metainfo.get('ori_shape'),
                scale_factor=data_sample.metainfo.get('scale_factor'),
            )
            img_metas.append(meta)
            gt_bboxes.append(data_sample.gt_instances.bboxes)
            gt_labels.append(data_sample.gt_instances.labels)

        # 3. 计算损失
        losses = self.bbox_head.loss(x, img_metas, gt_bboxes, gt_labels)
        return losses

    def predict(
        self,
        batch_inputs: torch.Tensor,
        batch_data_samples: List[DetDataSample],
        rescale: bool = True,
        return_trajectory: bool = False,
    ) -> List[DetDataSample]:
        """预测模式的前向传播"""
        # 1. 提取特征
        x = self.extract_feat(batch_inputs)

        # 2. 准备 img_metas
        img_metas = []
        for data_sample in batch_data_samples:
            meta = ImageMeta(
                img_shape=data_sample.metainfo['img_shape'],
                pad_shape=data_sample.metainfo.get('pad_shape'),
                ori_shape=data_sample.metainfo.get('ori_shape'),
                scale_factor=data_sample.metainfo.get('scale_factor'),
            )
            img_metas.append(meta)

        # 3. 运行 Head 的 predict
        if return_trajectory:
            results_list, trajectory = self.bbox_head.predict(
                x, img_metas, rescale=rescale, return_trajectory=True
            )
        else:
            results_list = self.bbox_head.predict(
                x, img_metas, rescale=rescale
            )

        # 4. 封装回 DetDataSample
        for i in range(len(batch_data_samples)):
            res = results_list[i]
            # res 是 DetectionResult 数据类
            from mmengine.structures import InstanceData

            pred_instances = InstanceData()
            pred_instances.bboxes = res.bboxes
            pred_instances.scores = res.scores
            pred_instances.labels = res.labels

            batch_data_samples[i].pred_instances = pred_instances

            # 如果返回了轨迹，存入 metainfo 供可视化工具使用
            if return_trajectory:
                batch_data_samples[i].metainfo['sampling_trajectory'] = (
                    trajectory
                )

        return batch_data_samples

    def _forward(
        self,
        batch_inputs: torch.Tensor,
        batch_data_samples: List[DetDataSample],
    ) -> Tuple[List[torch.Tensor]]:
        """基础前向传播，通常用于导出模型或简单的特征提取测试"""
        # 这里返回 Head 的原始输出 (logits 和 bboxes)
        x = self.extract_feat(batch_inputs)
        # 模拟一个全 0 的时间步进行测试
        t = x[0].new_zeros((x[0].shape[0],), dtype=torch.float32)

        # 初始噪声框
        img_metas = []
        for data_sample in batch_data_samples:
            meta = ImageMeta(
                img_shape=data_sample.metainfo['img_shape'],
                pad_shape=data_sample.metainfo.get('pad_shape'),
                ori_shape=data_sample.metainfo.get('ori_shape'),
                scale_factor=data_sample.metainfo.get('scale_factor'),
            )
            img_metas.append(meta)

        noise_bboxes = torch.randn(
            len(img_metas), self.bbox_head.num_proposals, 4, device=x[0].device
        )
        curr_bboxes = self.bbox_head._raw_to_xyxy(noise_bboxes, img_metas)

        all_cls_logits, all_pred_bboxes, _ = self.bbox_head(x, curr_bboxes, t)
        return all_cls_logits, all_pred_bboxes
