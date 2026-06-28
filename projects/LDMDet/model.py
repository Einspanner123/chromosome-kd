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
from .mods.diffusiondet_head import DiffusionDetHead
from .mods.loss import (
    BBoxL1Cost,
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    FocalLossCost,
    GIoULoss,
    IoUCost,
    L1Loss,
)
from .mods.roi_extractor import SingleRoIExtractor
from .mods.single_head import SingleDiffusionDetHead
from .mods.structures import ImageMeta

# 方向六: 注册 FeatureBridgeModule 和 ChromoGenFeatureExtractor 到 MODELS 注册表
from ldmdet.feature_bridge import (  # noqa: E402
    ChromoGenFeatureExtractor,
    FeatureBridgeModule,
)

# 注册所有组件到 MODELS 注册表，以便可以通过配置文件构建
MODELS.register_module(
    name='PurePyTorchDiffusionDetHead', module=DiffusionDetHead
)
MODELS.register_module(
    name='PurePyTorchSingleDiffusionDetHead', module=SingleDiffusionDetHead
)
MODELS.register_module(
    name='PurePyTorchSingleRoIExtractor', module=SingleRoIExtractor
)
MODELS.register_module(
    name='PurePyTorchDiffusionDetCriterion', module=DiffusionDetCriterion
)
MODELS.register_module(
    name='PurePyTorchDiffusionDetMatcher', module=DiffusionDetMatcher
)
MODELS.register_module(name='PurePyTorchFocalLoss', module=FocalLoss)
MODELS.register_module(name='PurePyTorchL1Loss', module=L1Loss)
MODELS.register_module(name='PurePyTorchGIoULoss', module=GIoULoss)
MODELS.register_module(name='PurePyTorchFocalLossCost', module=FocalLossCost)
MODELS.register_module(name='PurePyTorchBBoxL1Cost', module=BBoxL1Cost)
MODELS.register_module(name='PurePyTorchIoUCost', module=IoUCost)
# 方向六: 特征桥接组件
MODELS.register_module(name='FeatureBridgeModule', module=FeatureBridgeModule)
MODELS.register_module(
    name='ChromoGenFeatureExtractor', module=ChromoGenFeatureExtractor
)


@MODELS.register_module()
@MODELS.register_module(name='LDMDet')
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
        # 方向六: 生成模型感知迁移 (默认 None, 不影响 baseline)
        feature_bridge: OptConfigType = None,
        chromogen_extractor: OptConfigType = None,
        chromogen_vae: OptConfigType = None,
        chromogen_condition_encoder: OptConfigType = None,
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
        else:
            self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 方向六: 构建 FeatureBridgeModule (可选, 默认 None)
        self.feature_bridge = None
        if feature_bridge is not None:
            self.feature_bridge = MODELS.build(feature_bridge)

        # 方向六: 构建 ChromoGenFeatureExtractor (可选, 默认 None)
        self.chromogen_extractor = None
        if chromogen_extractor is not None:
            self.chromogen_extractor = MODELS.build(chromogen_extractor)

        # 方向六: 构建 ChromoGen VAE encoder (可选, 默认 None)
        self.chromogen_vae = None
        if chromogen_vae is not None:
            self.chromogen_vae = MODELS.build(chromogen_vae)

        # 方向六: 构建 ChromoGen 条件编码器 (可选, 默认 None)
        self.chromogen_condition_encoder = None
        if chromogen_condition_encoder is not None:
            self.chromogen_condition_encoder = MODELS.build(chromogen_condition_encoder)

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
        """构建检测头及其子组件"""
        cfg_copy = cfg.copy()

        head_type = cfg_copy.get('type', 'PurePyTorchDiffusionDetHead')

        # 1. 构建 single_head
        single_head_cfg = cfg_copy.pop('single_head')
        if isinstance(single_head_cfg, dict):
            if 'type' not in single_head_cfg:
                single_head_cfg['type'] = 'PurePyTorchSingleDiffusionDetHead'

            obj_cls = MODELS.get(single_head_cfg['type'])
            single_head = MODELS.build(
                self._filter_kwargs(obj_cls, single_head_cfg)
            )
        else:
            single_head = single_head_cfg

        # 2. 构建 roi_extractor
        roi_extractor = None
        roi_extractor_cfg = cfg_copy.pop('roi_extractor', None)
        if roi_extractor_cfg is not None:
            if isinstance(roi_extractor_cfg, dict):
                if 'type' not in roi_extractor_cfg:
                    roi_extractor_cfg['type'] = 'PurePyTorchSingleRoIExtractor'

                obj_cls = MODELS.get(roi_extractor_cfg['type'])
                roi_extractor = MODELS.build(
                    self._filter_kwargs(obj_cls, roi_extractor_cfg)
                )
            else:
                roi_extractor = roi_extractor_cfg

        # 3. 构建 criterion (仅在训练时需要，或者统一构建)
        criterion_cfg = cfg_copy.pop('criterion', None)
        criterion = None
        if criterion_cfg is not None:
            criterion = self._build_criterion(criterion_cfg)

        # 4. 构建可选的计数分支和一致性损失
        counting_branch_cfg = cfg_copy.pop('counting_branch', None)
        counting_branch = None
        if counting_branch_cfg is not None:
            if isinstance(counting_branch_cfg, dict):
                obj_cls = MODELS.get(counting_branch_cfg['type'])
                counting_branch = MODELS.build(
                    self._filter_kwargs(obj_cls, counting_branch_cfg)
                )
            else:
                counting_branch = counting_branch_cfg

        consistency_loss_cfg = cfg_copy.pop('consistency_loss', None)
        consistency_loss = None
        if consistency_loss_cfg is not None:
            if isinstance(consistency_loss_cfg, dict):
                obj_cls = MODELS.get(consistency_loss_cfg['type'])
                consistency_loss = MODELS.build(
                    self._filter_kwargs(obj_cls, consistency_loss_cfg)
                )
            else:
                consistency_loss = consistency_loss_cfg

        # 5. 构建 Head
        if 'type' not in cfg_copy:
            cfg_copy['type'] = head_type

        torch_compile = cfg_copy.pop('torch_compile', False)

        build_kwargs = dict(
            single_head=single_head,
            criterion=criterion,
            roi_extractor=roi_extractor,
            counting_branch=counting_branch,
            consistency_loss=consistency_loss,
        )

        cfg_copy.update(build_kwargs)

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
                    # 映射旧的名称到新的纯 PyTorch 名称
                    type_map = {
                        'FocalLossCost': 'PurePyTorchFocalLossCost',
                        'BBoxL1Cost': 'PurePyTorchBBoxL1Cost',
                        'IoUCost': 'PurePyTorchIoUCost',
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

        loss_bbox_cfg = cfg_copy.pop('loss_bbox')
        obj_cls = MODELS.get(loss_bbox_cfg['type'])
        loss_bbox = MODELS.build(self._filter_kwargs(obj_cls, loss_bbox_cfg))

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
                loss_bbox=loss_bbox,
                loss_giou=loss_giou,
            )
        )

        obj_cls = MODELS.get(cfg_copy['type'])
        return MODELS.build(self._filter_kwargs(obj_cls, cfg_copy))

    def extract_feat(self, batch_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """提取特征

        方向六: 若启用 feature_bridge, 在 FPN 输出后融合 ChromoGen 特征。
        """
        x = self.backbone(batch_inputs)
        if self.neck:
            x = self.neck(x)

        # 方向六: ChromoGen 特征融合
        if self.feature_bridge is not None and self.chromogen_extractor is not None:
            x = self._fuse_with_chromogen(x, batch_inputs)

        return x

    def _fuse_with_chromogen(
        self, ldmdet_feats: Tuple[torch.Tensor], batch_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """将 ChromoGen UNet 特征融合到 LDMDet FPN 特征

        Args:
            ldmdet_feats: LDMDet FPN 4 层特征
            batch_inputs: 原始输入图像 (B, 3, H, W), 用于 VAE 编码

        Returns:
            fused: 融合后的 4 层特征
        """
        # 1. VAE 编码: image → latent
        # ChromoGenVAE.forward 内部处理归一化和缩放, 直接调用返回 latent
        if self.chromogen_vae is not None:
            latent = self.chromogen_vae(batch_inputs)
        else:
            # 若无 VAE, 跳过融合 (返回原始特征)
            return ldmdet_feats

        # 2. 条件编码 (可选)
        encoder_hidden_states = None
        if self.chromogen_condition_encoder is not None:
            encoder_hidden_states = self.chromogen_condition_encoder(batch_inputs)

        # 3. UNet 特征提取 (timestep 用固定值 500, 仅用于特征提取)
        # ChromoGen 冻结, 用 no_grad 节省显存 (梯度只流向 FBM alpha)
        bs = batch_inputs.shape[0]
        timestep = torch.full((bs,), 500, device=batch_inputs.device, dtype=torch.long)
        if encoder_hidden_states is None:
            encoder_hidden_states = torch.zeros(
                bs, 1, 768, device=batch_inputs.device, dtype=batch_inputs.dtype
            )

        with torch.no_grad():
            cg_feats = self.chromogen_extractor(latent, timestep, encoder_hidden_states)

        # 4. FBM 融合
        fused = self.feature_bridge(ldmdet_feats, cg_feats)
        return fused

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
        if self.bbox_head.diffusion_type == 'ddpm':
            t = x[0].new_zeros((x[0].shape[0],), dtype=torch.long)
        else:
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
