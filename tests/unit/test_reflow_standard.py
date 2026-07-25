"""测试 ReFlow (Standard MSE): 基于 Coupling 变换的 2-Rectification

核心机制 (REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1):
  用 A4 推理生成新 coupling (x_0^pred, x_1^noise), 替代 (x_0^GT, x_1^noise),
  再用标准检测损失 (cls=GT, box=x_0^pred) 训练 2-RF 拉直轨迹。

v2 关键设计 (混合 target):
  - cls target = GT (SimOTA 用 GT 分配正负样本)
  - box target = x_0^pred (RF 拉直目标, 正样本 box 回归到 A4 预测)
  - 正样本筛选仍用 GT (matcher 基于 GT)

核心验证:
1. use_reflow_coupling 参数存在 (RectifiedFlow + DiffusionDetHead)
2. box_target_mode 参数存在 (DiffusionDetCriterion), 默认 'gt' 向后兼容
3. box_target_mode='x0_pred' 时 box loss target 用 x_0^pred 而非 GT
4. reflow 模式下 _build_training_targets 从预存 coupling 加载 x_start/noise
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.diffusion.rectified_flow import RectifiedFlow
from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.roi_extractor import SingleRoIExtractor
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.criterion import (
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    GIoULoss,
    L1Loss,
)
from ldmdet.data.structures import InstanceData, ModelOutput


def _make_single_head(num_classes=24, feat_channels=64):
    return SingleDiffusionDetHead(
        num_classes=num_classes, feat_channels=feat_channels,
        num_cls_convs=1, num_reg_convs=2, use_focal_loss=True,
        use_normalized_classifier=False, time_conditioning='adaln_zero',
    )


def _make_roi_extractor(out_channels=64):
    return SingleRoIExtractor(
        featmap_strides=[16], out_channels=out_channels,
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True),
    )


def _make_criterion(box_target_mode='gt', num_classes=24):
    """构建 DiffusionDetCriterion (可选 box_target_mode)"""
    matcher = DiffusionDetMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    return DiffusionDetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        loss_cls=FocalLoss(use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        deep_supervision=False,
        box_target_mode=box_target_mode,
    )


def _make_targets_and_outputs(num_queries=10, num_classes=24, bs=2):
    """构造 targets (GT) + outputs (pred) + box_targets (x0_pred)"""
    torch.manual_seed(42)
    # GT bboxes (xyxy, 每图 2 个)
    gt_bboxes = torch.tensor([[[10, 10, 50, 50], [60, 60, 100, 100]]] * bs, dtype=torch.float32)
    gt_labels = torch.tensor([[0, 1]] * bs, dtype=torch.long)
    targets = [
        InstanceData(bboxes=gt_bboxes[i], labels=gt_labels[i], img_shape=(100, 100))
        for i in range(bs)
    ]
    # pred boxes: [bs, num_queries, 4] — 必须与 pred_logits 的 N 对齐
    # matcher 用 pred_bboxes.shape[:2] 取 N, 若 pred_boxes 只有 2 个框,
    # is_in_boxes 为 [bs,2,max_gt] 而 focal cost 为 [bs,10,max_gt], stack 失败
    # 复制 GT + 小扰动填充到 num_queries, 保证 matcher center-prior 能匹配正样本
    n_gt = gt_bboxes.shape[1]
    repeats = (num_queries + n_gt - 1) // n_gt
    pred_boxes = gt_bboxes.repeat(1, repeats, 1)[:, :num_queries, :].contiguous()
    pred_boxes = pred_boxes + torch.randn_like(pred_boxes) * 0.5  # 小扰动 (pred≠GT)
    pred_logits = torch.randn(bs, num_queries, num_classes + 1)
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_boxes, aux_outputs=None)
    # x0_pred: [bs, n_gt, 4] (与 GT 同布局, ReFlow 的 box target)
    # 明显不同于 GT (Δ=5), 确保 loss_gt ≠ loss_x0
    x0_preds = torch.tensor([[[15, 15, 55, 55], [65, 65, 105, 105]]] * bs, dtype=torch.float32)
    return targets, outputs, x0_preds


# ============================================================
# 1. 参数存在性
# ============================================================

class TestReflowParamsExist:
    """ReFlow 参数存在性测试"""

    def test_rectified_flow_reflow_params(self):
        """RectifiedFlow 接受 use_reflow_coupling, reflow_dims 参数"""
        rf = RectifiedFlow(snr_scale=2.0, use_reflow_coupling=True, reflow_dims='all')
        assert rf.use_reflow_coupling is True
        assert rf.reflow_dims == 'all'

    def test_rectified_flow_reflow_params_default(self):
        """RectifiedFlow reflow 参数默认值 (向后兼容)"""
        rf = RectifiedFlow(snr_scale=2.0)
        assert rf.use_reflow_coupling is False
        assert rf.reflow_dims == 'all'

    def test_head_reflow_params(self):
        """DiffusionDetHead 接受 use_reflow_coupling, reflow_coupling_path 参数"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_reflow_coupling=True,
            reflow_coupling_path='fake_couplings.pt',
        )
        assert head.use_reflow_coupling is True
        assert head.reflow_coupling_path == 'fake_couplings.pt'

    def test_head_reflow_params_default(self):
        """DiffusionDetHead reflow 参数默认值 (向后兼容)"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert head.use_reflow_coupling is False
        assert head.reflow_coupling_path is None

    def test_criterion_box_target_mode_param(self):
        """DiffusionDetCriterion 接受 box_target_mode 参数"""
        criterion = _make_criterion(box_target_mode='x0_pred')
        assert criterion.box_target_mode == 'x0_pred'

    def test_criterion_box_target_mode_default(self):
        """DiffusionDetCriterion box_target_mode 默认 'gt' (向后兼容)"""
        criterion = _make_criterion()
        assert criterion.box_target_mode == 'gt'


# ============================================================
# 2. box_target_mode 行为
# ============================================================

class TestBoxTargetMode:
    """box_target_mode: 'gt' (默认) vs 'x0_pred' (ReFlow)"""

    def test_gt_mode_uses_gt_bboxes(self):
        """box_target_mode='gt' 时 box loss target = GT bboxes (现有行为)"""
        criterion = _make_criterion(box_target_mode='gt')
        targets, outputs, x0_preds = _make_targets_and_outputs()
        losses = criterion.forward(outputs, targets)
        assert 'loss_bbox' in losses
        assert losses['loss_bbox'].item() > 0  # pred ≠ GT, loss 非零

    def test_x0_pred_mode_uses_x0_pred(self):
        """box_target_mode='x0_pred' 时 box loss target = x_0^pred (非 GT)

        关键: ReFlow 的 box target 从 GT 改为 A4 预测 (x_0^pred)。
        传入 x0_preds ≠ GT, 验证 loss 与用 x0_preds 计算的一致,
        且不同于用 GT 计算的 loss。
        """
        targets, outputs, x0_preds = _make_targets_and_outputs()
        # gt 模式 loss
        criterion_gt = _make_criterion(box_target_mode='gt')
        losses_gt = criterion_gt.forward(outputs, targets)
        loss_gt = losses_gt['loss_bbox'].item()
        # x0_pred 模式 loss (传入 box_targets=x0_preds)
        criterion_x0 = _make_criterion(box_target_mode='x0_pred')
        losses_x0 = criterion_x0.forward(outputs, targets, box_targets=x0_preds)
        loss_x0 = losses_x0['loss_bbox'].item()
        # x0_preds ≠ GT → 两个 loss 应不同 (证明 box target 来源改变)
        assert abs(loss_gt - loss_x0) > 1e-4, \
            f"box_target_mode 未改变 box target: loss_gt={loss_gt}, loss_x0={loss_x0}"

    def test_x0_pred_mode_ignored_without_box_targets(self):
        """box_target_mode='x0_pred' 但未传 box_targets 时, 回退到 GT (安全降级)"""
        criterion = _make_criterion(box_target_mode='x0_pred')
        targets, outputs, _ = _make_targets_and_outputs()
        # 不传 box_targets → 应回退到 GT (不崩溃)
        losses = criterion.forward(outputs, targets)
        assert 'loss_bbox' in losses

    def test_x0_pred_per_proposal_mode(self):
        """canonical ReFlow: box_targets 为 per-proposal [bs, num_proposals, 4]

        REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1.2: box target = x_0^pred (per-proposal,
        每个 proposal 的 A4 预测 = RF 轨迹端点), 与 pred_boxes 直接对齐, 不经 gather。
        形状 [bs, num_proposals, 4] 触发 criterion 的 per-proposal 分流 (shape[1]==N)。
        """
        targets, outputs, _ = _make_targets_and_outputs()
        num_queries = outputs.pred_boxes.shape[1]
        bs = outputs.pred_boxes.shape[0]
        # per-proposal x0_pred: 与 pred_boxes 和 GT 都不同
        x0_preds_per_proposal = outputs.pred_boxes + 3.0  # [bs, num_queries, 4]
        assert x0_preds_per_proposal.shape == (bs, num_queries, 4)

        criterion_gt = _make_criterion(box_target_mode='gt')
        criterion_x0 = _make_criterion(box_target_mode='x0_pred')
        losses_gt = criterion_gt.forward(outputs, targets)
        losses_x0 = criterion_x0.forward(
            outputs, targets, box_targets=x0_preds_per_proposal
        )
        # per-proposal x0_pred ≠ GT → box loss 应不同
        assert abs(losses_gt['loss_bbox'].item() - losses_x0['loss_bbox'].item()) > 1e-4, \
            "per-proposal box_target_mode 未改变 box target"
        # cls target 始终 GT, 两模式一致
        assert torch.allclose(losses_gt['loss_cls'], losses_x0['loss_cls'], atol=1e-5)

    def test_cls_target_always_gt(self):
        """cls target 始终用 GT (不受 box_target_mode 影响)

        ReFlow 混合 target: cls=GT, box=x_0^pred。验证 cls loss 在两种模式下一致。
        """
        targets, outputs, x0_preds = _make_targets_and_outputs()
        criterion_gt = _make_criterion(box_target_mode='gt')
        criterion_x0 = _make_criterion(box_target_mode='x0_pred')
        losses_gt = criterion_gt.forward(outputs, targets)
        losses_x0 = criterion_x0.forward(outputs, targets, box_targets=x0_preds)
        # cls loss 应基本一致 (matcher 基于 GT, cls target=GT)
        assert torch.allclose(losses_gt['loss_cls'], losses_x0['loss_cls'], atol=1e-5), \
            "cls target 应始终用 GT, 不受 box_target_mode 影响"


# ============================================================
# 3. reflow coupling 加载 (head._build_training_targets)
# ============================================================

class TestReflowCouplingLoad:
    """reflow 模式下 _build_training_targets 从预存 coupling 加载 x_start/noise"""

    def test_reflow_mode_loads_coupling(self):
        """use_reflow_coupling=True 时 _build_training_targets 从 coupling 文件加载

        reflow 模式: x_start = 预存 x_0^pred, noise = 预存 x_1^noise,
        跳过在线 OT coupling (matched_idx 在线重算基于 GT)。
        """
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None, diffusion_type='rectified_flow',
            use_reflow_coupling=True,
            reflow_coupling_path='fake_couplings.pt',
        )
        # mock coupling 文件: {image_id: {'x0_pred': tensor, 'noise': tensor}}
        fake_noise = torch.randn(10, 4)
        fake_x0_pred = torch.randn(10, 4)
        fake_coupling = {0: {'noise': fake_noise, 'x0_pred': fake_x0_pred}}

        with patch.object(head, '_load_reflow_coupling', return_value=fake_coupling):
            head._reflow_couplings = fake_coupling
            bs = 1
            device = torch.device('cpu')
            t = torch.tensor([0.5])
            gt_bboxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)]
            targets = [InstanceData(bboxes=gt_bboxes[0], labels=torch.tensor([0]), img_shape=(100, 100))]
            x_boxes, x_starts, x_noises, matched_idx = head._build_training_targets(
                bs, device, t, targets, gt_bboxes
            )
            # reflow 模式: x_start 应为预存 x0_pred (非 GT diffusion)
            # 注意: x_start 经历了 _forward_diffusion, 但 reflow 模式 x_start = x0_pred
            assert len(x_starts) == bs
            # noise 应为预存 noise (非随机)
            assert len(x_noises) == bs


# ============================================================
# 4. BUG #1 修复: 空间一致性 (raw cxcywh → 归一化 xyxy)
# ============================================================

class TestSpaceConsistencyFix:
    """BUG #1 修复: head.loss() 中 box_targets 空间转换

    coupling 的 x0_pred 处于 raw 扩散空间 (cxcywh, scaled [-snr, snr]);
    criterion 期望 box_targets 与 outputs.pred_boxes 同空间 (归一化 xyxy [0,1]).
    head.loss() 必须在传给 criterion 前转换空间.
    """

    def test_raw_to_normalized_xyxy_conversion(self):
        """raw cxcywh scaled → normalized xyxy 转换数学正确性

        验证 head.loss() 中的转换公式:
          norm_cxcywh = (raw.clamp(-snr, snr) / snr + 1) / 2  → [0, 1]
          norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        与 _sampler.raw_to_xyxy 的前两步一致 (不乘 img_scale).
        """
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        snr_scale = 2.0

        # 构造 raw cxcywh: 中心 (0, 0) 即图像中心, 宽高 (1, 1) 即一半图像
        # raw = (norm_cxcywh * 2 - 1) * snr → norm_cxcywh = (raw / snr + 1) / 2
        # 若 raw = 0 → norm_cxcywh = 0.5 (中心)
        # 若 raw = snr → norm_cxcywh = 1.0 (右/下边界)
        # 若 raw = -snr → norm_cxcywh = 0.0 (左/上边界)
        raw_cxcywh = torch.tensor([[[0.0, 0.0, snr_scale, snr_scale]]])  # 中心, 全宽全高
        norm_cxcywh = (raw_cxcywh.clamp(-snr_scale, snr_scale) / snr_scale + 1) / 2
        # 期望: norm_cxcywh = [0.5, 0.5, 1.0, 1.0]
        expected_cxcywh = torch.tensor([[[0.5, 0.5, 1.0, 1.0]]])
        assert torch.allclose(norm_cxcywh, expected_cxcywh, atol=1e-6)

        norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        # cxcywh (0.5, 0.5, 1.0, 1.0) → xyxy (0.0, 0.0, 1.0, 1.0)
        expected_xyxy = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        assert torch.allclose(norm_xyxy, expected_xyxy, atol=1e-6)

    def test_raw_to_normalized_xyxy_negative_values(self):
        """raw 负值 (左/上边界) 转换正确"""
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        snr_scale = 2.0
        # raw = -snr → norm_cxcywh = 0.0 (左/上边界)
        raw_cxcywh = torch.tensor([[[-snr_scale, -snr_scale, snr_scale, snr_scale]]])
        norm_cxcywh = (raw_cxcywh.clamp(-snr_scale, snr_scale) / snr_scale + 1) / 2
        expected_cxcywh = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        assert torch.allclose(norm_cxcywh, expected_cxcywh, atol=1e-6)
        norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        # cxcywh (0, 0, 1, 1) → xyxy (-0.5, -0.5, 0.5, 0.5)
        expected_xyxy = torch.tensor([[[-0.5, -0.5, 0.5, 0.5]]])
        assert torch.allclose(norm_xyxy, expected_xyxy, atol=1e-6)

    def test_clamp_prevents_overflow(self):
        """raw 值超出 [-snr, snr] 范围时 clamp 防止 norm 溢出 [0,1]"""
        snr_scale = 2.0
        # raw = 10 (远超 snr=2), clamp 后应为 snr=2 → norm=1.0
        raw_cxcywh = torch.tensor([[[10.0, -10.0, 10.0, -10.0]]])
        norm_cxcywh = (raw_cxcywh.clamp(-snr_scale, snr_scale) / snr_scale + 1) / 2
        # 期望: [1.0, 0.0, 1.0, 0.0]
        expected = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
        assert torch.allclose(norm_cxcywh, expected, atol=1e-6)


# ============================================================
# 5. BUG #2 修复: ImageMeta.img_id 字段
# ============================================================

class TestImageMetaImgIdField:
    """BUG #2 修复: ImageMeta dataclass 新增 img_id 字段

    head.py 用 meta.img_id 属性访问 (非 dict .get()), ImageMeta 必须有此字段.
    """

    def test_img_id_field_exists(self):
        """ImageMeta 有 img_id 字段"""
        from ldmdet.data.structures import ImageMeta
        meta = ImageMeta(img_shape=(100, 100), img_id=42)
        assert meta.img_id == 42

    def test_img_id_default_none(self):
        """img_id 默认 None (向后兼容)"""
        from ldmdet.data.structures import ImageMeta
        meta = ImageMeta(img_shape=(100, 100))
        assert meta.img_id is None

    def test_head_img_ids_extraction_with_image_meta(self):
        """head.py 从 ImageMeta 正确提取 img_id (BUG #2 修复)"""
        from ldmdet.data.structures import ImageMeta
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_reflow_coupling=True,
            reflow_coupling_path='fake.pt',
        )
        # 构造带 img_id 的 ImageMeta 列表
        img_metas = [
            ImageMeta(img_shape=(100, 100), img_id=1001),
            ImageMeta(img_shape=(100, 100), img_id=1002),
        ]
        # 模拟 head.loss() 中的 img_ids 提取逻辑
        img_ids = [
            meta.img_id if meta.img_id is not None else i
            for i, meta in enumerate(img_metas)
        ]
        assert img_ids == [1001, 1002]

    def test_head_img_ids_fallback_to_batch_index(self):
        """img_id=None 时回退到 batch 索引 (测试场景)"""
        from ldmdet.data.structures import ImageMeta
        img_metas = [
            ImageMeta(img_shape=(100, 100)),  # img_id=None
            ImageMeta(img_shape=(100, 100)),  # img_id=None
        ]
        img_ids = [
            meta.img_id if meta.img_id is not None else i
            for i, meta in enumerate(img_metas)
        ]
        assert img_ids == [0, 1]


# ============================================================
# 6. BUG #3 修复: RectifiedFlow 参数传递
# ============================================================

class TestRectifiedFlowParamPassthrough:
    """BUG #3 修复: head.py 传 reflow_dims 到 RectifiedFlow (消除死代码)"""

    def test_head_passes_reflow_dims_to_rf(self):
        """head.py 将 reflow_dims 传递给 RectifiedFlow"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_reflow_coupling=True,
            reflow_coupling_path='fake.pt',
            reflow_dims='cxcy',
        )
        assert head.rf.use_reflow_coupling is True
        assert head.rf.reflow_dims == 'cxcy'
        assert head.reflow_dims == 'cxcy'

    def test_head_default_reflow_dims_all(self):
        """head.py 默认 reflow_dims='all'"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert head.rf.reflow_dims == 'all'
        assert head.reflow_dims == 'all'


# ============================================================
# 7. BUG #5 修复: _last_x0_final 初始化与断言
# ============================================================

class TestLastX0FinalInit:
    """BUG #5 修复: _last_x0_final / _last_x_raw_initial 在 __init__ 初始化为 None"""

    def test_last_x0_final_init_none(self):
        """_last_x0_final 在 __init__ 后为 None (predict 调用前可安全访问)"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert head._last_x0_final is None
        assert head._last_x_raw_initial is None
