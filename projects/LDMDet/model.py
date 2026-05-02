import inspect
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.optim import OptimWrapperDict

from .mods.diffusiondet_head import DiffusionDetHead
from .mods.loss import (
    BBoxL1Cost,
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FlowMatchingVelocityLoss,
    FocalLoss,
    FocalLossCost,
    GIoULoss,
    IoUCost,
    L1Loss,
)
from .mods.noise_sampler import StructuredNoiseSampler
from .mods.roi_extractor import SingleRoIExtractor
from .mods.single_head import SingleDiffusionDetHead
from .mods.sinkhorn import SinkhornOTMatcher
from .mods.structures import ImageMeta

# 注册所有组件到 MODELS 注册表，以便可以通过配置文件构建
MODELS.register_module(name="PurePyTorchDiffusionDetHead", module=DiffusionDetHead)
MODELS.register_module(
    name="PurePyTorchSingleDiffusionDetHead", module=SingleDiffusionDetHead
)
MODELS.register_module(name="PurePyTorchSingleRoIExtractor", module=SingleRoIExtractor)
MODELS.register_module(
    name="PurePyTorchDiffusionDetCriterion", module=DiffusionDetCriterion
)
MODELS.register_module(
    name="PurePyTorchDiffusionDetMatcher", module=DiffusionDetMatcher
)
MODELS.register_module(name="PurePyTorchFocalLoss", module=FocalLoss)
MODELS.register_module(name="PurePyTorchL1Loss", module=L1Loss)
MODELS.register_module(name="PurePyTorchGIoULoss", module=GIoULoss)
MODELS.register_module(name="PurePyTorchFocalLossCost", module=FocalLossCost)
MODELS.register_module(name="PurePyTorchBBoxL1Cost", module=BBoxL1Cost)
MODELS.register_module(name="PurePyTorchIoUCost", module=IoUCost)

# === FlowDet 新增模块注册 ===
MODELS.register_module(name="PurePyTorchSinkhornOTMatcher", module=SinkhornOTMatcher)
MODELS.register_module(
    name="PurePyTorchStructuredNoiseSampler", module=StructuredNoiseSampler
)
MODELS.register_module(
    name="PurePyTorchFlowMatchingVelocityLoss", module=FlowMatchingVelocityLoss
)


@MODELS.register_module()
class LDMDet(BaseDetector):
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
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
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
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return cfg

        valid_cfg = {k: v for k, v in cfg.items() if k in params or k == "type"}
        return valid_cfg

    def _build_bbox_head(self, cfg: ConfigType) -> nn.Module:
        """构建检测头及其子组件"""
        cfg_copy = cfg.copy()

        # 1. 构建 single_head
        single_head_cfg = cfg_copy.pop("single_head")
        if isinstance(single_head_cfg, dict):
            # 如果配置中没有 type，默认使用我们注册的纯 PyTorch 版本
            if "type" not in single_head_cfg:
                single_head_cfg["type"] = "PurePyTorchSingleDiffusionDetHead"

            obj_cls = MODELS.get(single_head_cfg["type"])
            single_head = MODELS.build(self._filter_kwargs(obj_cls, single_head_cfg))
        else:
            single_head = single_head_cfg

        # 2. 构建 roi_extractor
        roi_extractor_cfg = cfg_copy.pop("roi_extractor")
        if isinstance(roi_extractor_cfg, dict):
            if "type" not in roi_extractor_cfg:
                roi_extractor_cfg["type"] = "PurePyTorchSingleRoIExtractor"

            obj_cls = MODELS.get(roi_extractor_cfg["type"])
            roi_extractor = MODELS.build(
                self._filter_kwargs(obj_cls, roi_extractor_cfg)
            )
        else:
            roi_extractor = roi_extractor_cfg

        # 3. 构建 criterion (仅在训练时需要，或者统一构建)
        criterion_cfg = cfg_copy.pop("criterion", None)
        criterion = None
        if criterion_cfg is not None:
            criterion = self._build_criterion(criterion_cfg)

        # 4. 构建可选的计数分支和一致性损失
        counting_branch_cfg = cfg_copy.pop("counting_branch", None)
        counting_branch = None
        if counting_branch_cfg is not None:
            if isinstance(counting_branch_cfg, dict):
                obj_cls = MODELS.get(counting_branch_cfg["type"])
                counting_branch = MODELS.build(
                    self._filter_kwargs(obj_cls, counting_branch_cfg)
                )
            else:
                counting_branch = counting_branch_cfg

        consistency_loss_cfg = cfg_copy.pop("consistency_loss", None)
        consistency_loss = None
        if consistency_loss_cfg is not None:
            if isinstance(consistency_loss_cfg, dict):
                obj_cls = MODELS.get(consistency_loss_cfg["type"])
                consistency_loss = MODELS.build(
                    self._filter_kwargs(obj_cls, consistency_loss_cfg)
                )
            else:
                consistency_loss = consistency_loss_cfg

        # 4b. 构建可选的结构化噪声采样器 (FlowDet Phase 3A)
        noise_sampler_cfg = cfg_copy.pop("noise_sampler", None)
        noise_sampler = None
        if noise_sampler_cfg is not None:
            if isinstance(noise_sampler_cfg, dict):
                obj_cls = MODELS.get(noise_sampler_cfg["type"])
                noise_sampler = MODELS.build(
                    self._filter_kwargs(obj_cls, noise_sampler_cfg)
                )
            else:
                noise_sampler = noise_sampler_cfg

        # 5. 构建 DiffusionDetHead
        if "type" not in cfg_copy:
            cfg_copy["type"] = "PurePyTorchDiffusionDetHead"

        # 将实例化的子组件传入
        cfg_copy.update(
            dict(
                single_head=single_head,
                roi_extractor=roi_extractor,
                criterion=criterion,
                counting_branch=counting_branch,
                consistency_loss=consistency_loss,
                noise_sampler=noise_sampler,
            )
        )

        obj_cls = MODELS.get(cfg_copy["type"])
        return MODELS.build(self._filter_kwargs(obj_cls, cfg_copy))

    def _build_criterion(self, cfg: ConfigType) -> nn.Module:
        """构建损失函数组件"""
        cfg_copy = cfg.copy()

        # 构建 Matcher
        assigner_cfg = cfg_copy.pop("assigner")
        if "type" not in assigner_cfg:
            assigner_cfg["type"] = "PurePyTorchDiffusionDetMatcher"

        # 处理 match_costs
        if "match_costs" in assigner_cfg:
            costs = []
            for cost_cfg in assigner_cfg["match_costs"]:
                if isinstance(cost_cfg, dict):
                    # 映射旧的名称到新的纯 PyTorch 名称
                    type_map = {
                        "FocalLossCost": "PurePyTorchFocalLossCost",
                        "BBoxL1Cost": "PurePyTorchBBoxL1Cost",
                        "IoUCost": "PurePyTorchIoUCost",
                    }
                    if "type" in cost_cfg:
                        cost_cfg["type"] = type_map.get(
                            cost_cfg["type"], cost_cfg["type"]
                        )

                    obj_cls = MODELS.get(cost_cfg["type"])
                    costs.append(MODELS.build(self._filter_kwargs(obj_cls, cost_cfg)))
                else:
                    costs.append(cost_cfg)
            assigner_cfg["match_costs"] = costs

        obj_cls = MODELS.get(assigner_cfg["type"])
        matcher = MODELS.build(self._filter_kwargs(obj_cls, assigner_cfg))

        # 构建各类 Loss
        loss_cls_cfg = cfg_copy.pop("loss_cls")
        obj_cls = MODELS.get(loss_cls_cfg["type"])
        loss_cls = MODELS.build(self._filter_kwargs(obj_cls, loss_cls_cfg))

        loss_bbox_cfg = cfg_copy.pop("loss_bbox")
        obj_cls = MODELS.get(loss_bbox_cfg["type"])
        loss_bbox = MODELS.build(self._filter_kwargs(obj_cls, loss_bbox_cfg))

        loss_giou_cfg = cfg_copy.pop("loss_giou")
        obj_cls = MODELS.get(loss_giou_cfg["type"])
        loss_giou = MODELS.build(self._filter_kwargs(obj_cls, loss_giou_cfg))

        # 实例化 Criterion
        if "type" not in cfg_copy:
            cfg_copy["type"] = "PurePyTorchDiffusionDetCriterion"

        cfg_copy.update(
            dict(
                matcher=matcher,
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_giou=loss_giou,
            )
        )

        obj_cls = MODELS.get(cfg_copy["type"])
        return MODELS.build(self._filter_kwargs(obj_cls, cfg_copy))

    def extract_feat(self, batch_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """提取特征"""
        x = self.backbone(batch_inputs)
        if self.neck:
            x = self.neck(x)
        return x

    def loss(
        self, batch_inputs: torch.Tensor, batch_data_samples: List[DetDataSample]
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
                img_shape=data_sample.metainfo["img_shape"],
                ori_shape=data_sample.metainfo.get("ori_shape"),
                scale_factor=data_sample.metainfo.get("scale_factor"),
            )
            img_metas.append(meta)
            gt_bboxes.append(data_sample.gt_instances.bboxes)
            gt_labels.append(data_sample.gt_instances.labels)

        # 3. 计算损失
        losses = self.bbox_head.loss(x, img_metas, gt_bboxes, gt_labels)
        return losses

    def train_step(
        self, data: dict, optim_wrapper
    ) -> Dict[str, torch.Tensor]:
        if not getattr(self.bbox_head, "use_pcgrad", False):
            return super().train_step(data, optim_wrapper)

        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            batch_inputs = data["inputs"]
            batch_data_samples = data["data_samples"]

        img_metas = []
        gt_bboxes = []
        gt_labels = []
        for ds in batch_data_samples:
            meta = ImageMeta(
                img_shape=ds.metainfo["img_shape"],
                ori_shape=ds.metainfo.get("ori_shape"),
                scale_factor=ds.metainfo.get("scale_factor"),
            )
            img_metas.append(meta)
            gt_bboxes.append(ds.gt_instances.bboxes)
            gt_labels.append(ds.gt_instances.labels)

        x = self.extract_feat(batch_inputs)

        det_loss_keys = {
            "loss_cls", "loss_bbox", "loss_giou",
            "aux_0_loss_cls", "aux_0_loss_bbox", "aux_0_loss_giou",
            "aux_1_loss_cls", "aux_1_loss_bbox", "aux_1_loss_giou",
            "aux_2_loss_cls", "aux_2_loss_bbox", "aux_2_loss_giou",
            "aux_3_loss_cls", "aux_3_loss_bbox", "aux_3_loss_giou",
            "aux_4_loss_cls", "aux_4_loss_bbox", "aux_4_loss_giou",
        }
        vel_loss_keys = {"loss_velocity", "loss_velocity_aux"}
        other_loss_keys = {"loss_itd", "loss_consistency"}

        losses = self.bbox_head.loss(x, img_metas, gt_bboxes, gt_labels)

        det_loss = sum(losses[k] for k in det_loss_keys if k in losses)
        vel_loss = sum(losses[k] for k in vel_loss_keys if k in losses)
        other_loss = sum(losses[k] for k in other_loss_keys if k in losses)

        named_params = {
            n: p for n, p in self.bbox_head.named_parameters() if p.requires_grad
        }

        self.bbox_head.zero_grad()
        det_loss.backward(retain_graph=True)
        grad_det = {}
        for n, p in named_params.items():
            if p.grad is not None:
                grad_det[n] = p.grad.clone()

        self.bbox_head.zero_grad()
        vel_loss.backward(retain_graph=True)
        grad_vel = {}
        for n, p in named_params.items():
            if p.grad is not None:
                grad_vel[n] = p.grad.clone()

        if isinstance(other_loss, torch.Tensor) and other_loss.item() > 0:
            self.bbox_head.zero_grad()
            other_loss.backward(retain_graph=True)
            grad_other = {}
            for n, p in named_params.items():
                if p.grad is not None:
                    grad_other[n] = p.grad.clone()
        else:
            grad_other = {}

        main_task = getattr(self.bbox_head, "pcgrad_main_task", "det")
        common_names = sorted(set(grad_det.keys()) & set(grad_vel.keys()))

        num_conflict = 0
        num_aligned = 0
        for n in common_names:
            gd = grad_det[n].flatten()
            gv = grad_vel[n].flatten()

            cos = F.cosine_similarity(gd.unsqueeze(0), gv.unsqueeze(0)).item()

            if cos < 0:
                num_conflict += 1
                if main_task == "det":
                    gv_proj = gv - cos * gd / (gd.norm() ** 2 + 1e-8) * gd
                    merged = gd + gv_proj
                else:
                    gd_proj = gd - cos * gv / (gv.norm() ** 2 + 1e-8) * gv
                    merged = gd_proj + gv
            else:
                num_aligned += 1
                merged = gd + gv

            p = named_params[n]
            if p.grad is None:
                p.grad = merged.reshape(p.shape)
            else:
                p.grad.copy_(merged.reshape(p.shape))

        for n in grad_other:
            if n in named_params:
                p = named_params[n]
                go = grad_other[n]
                if p.grad is None:
                    p.grad = go.reshape(p.shape)
                else:
                    p.grad.add_(go.reshape(p.shape))

        for n in grad_det:
            if n not in common_names and n in named_params:
                p = named_params[n]
                gd = grad_det[n]
                if p.grad is None:
                    p.grad = gd.reshape(p.shape)
                else:
                    p.grad.add_(gd.reshape(p.shape))

        for n in grad_vel:
            if n not in common_names and n in named_params:
                p = named_params[n]
                gv = grad_vel[n]
                if p.grad is None:
                    p.grad = gv.reshape(p.shape)
                else:
                    p.grad.add_(gv.reshape(p.shape))

        if hasattr(self.bbox_head, "_pcgrad_stats"):
            self.bbox_head._pcgrad_stats["conflict"] = num_conflict
            self.bbox_head._pcgrad_stats["aligned"] = num_aligned

        if isinstance(optim_wrapper, OptimWrapperDict):
            for ow in optim_wrapper.values():
                ow.step()
                ow.zero_grad()
        else:
            optim_wrapper.step()
            optim_wrapper.zero_grad()

        _, log_vars = self.parse_losses(losses)
        log_vars["pcgrad_conflict"] = num_conflict
        log_vars["pcgrad_aligned"] = num_aligned
        return log_vars

    def predict(
        self,
        batch_inputs: torch.Tensor,
        batch_data_samples: List[DetDataSample],
        rescale: bool = True,
    ) -> List[DetDataSample]:
        """预测模式的前向传播"""
        # 1. 提取特征
        x = self.extract_feat(batch_inputs)

        # 2. 准备 img_metas
        img_metas = []
        for data_sample in batch_data_samples:
            meta = ImageMeta(
                img_shape=data_sample.metainfo["img_shape"],
                ori_shape=data_sample.metainfo.get("ori_shape"),
                scale_factor=data_sample.metainfo.get("scale_factor"),
            )
            img_metas.append(meta)

        # 3. 运行 Head 的 predict
        results_list = self.bbox_head.predict(x, img_metas, rescale=rescale)

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

        return batch_data_samples

    def _forward(
        self, batch_inputs: torch.Tensor, batch_data_samples: List[DetDataSample]
    ) -> Tuple[List[torch.Tensor]]:
        """基础前向传播，通常用于导出模型或简单的特征提取测试"""
        # 这里返回 Head 的原始输出 (logits 和 bboxes)
        x = self.extract_feat(batch_inputs)
        # 模拟一个全 0 的时间步进行测试
        if self.bbox_head.diffusion_type == "ddpm":
            t = x[0].new_zeros((x[0].shape[0],), dtype=torch.long)
        else:
            t = x[0].new_zeros((x[0].shape[0],), dtype=torch.float32)

        # 初始噪声框
        img_metas = []
        for data_sample in batch_data_samples:
            meta = ImageMeta(
                img_shape=data_sample.metainfo["img_shape"],
                ori_shape=data_sample.metainfo.get("ori_shape"),
                scale_factor=data_sample.metainfo.get("scale_factor"),
            )
            img_metas.append(meta)

        noise_bboxes = torch.randn(
            len(img_metas), self.bbox_head.num_proposals, 4, device=x[0].device
        )
        curr_bboxes = self.bbox_head._raw_to_xyxy(noise_bboxes, img_metas)

        all_cls_logits, all_pred_bboxes, _, _ = self.bbox_head(x, curr_bboxes, t)
        return all_cls_logits, all_pred_bboxes
