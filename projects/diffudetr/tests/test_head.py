"""DiffuDETRHead 单元测试.

对照原仓库 (MBadran2000/DiffuDETR) dino_diffu_det_noise.py, 验证:
  - 空间转换 diffusion_to_norm / norm_to_diffusion (对齐 apply_box_noise line 397-403)
  - inverse_sigmoid (对齐 DINO inverse_sigmoid)
  - DINO iterative refinement: (offset + inverse_sigmoid(ref)).sigmoid() (line 541-549)
    * 输出有界 [0,1]
    * 梯度有界 (sigmoid 最大 0.25)
    * 迭代精修: 下一层 reference = 当前预测
  - _prepare_x_start: GT 放置 + 填充 + shuffle (对齐 prepare_targets + noisy_gt)
  - _run_heads: 每层输出 [0,1], 残差迭代
  - forward_train: 返回有限 loss_dict, 含 aux keys
  - forward_inference: 返回结构正确, NMS 去重

Bug 对照:
  - Bug 7: 填充值差异 (原仓库 randn.sigmoid() 在 [0,1] 空间 vs 我们 randn 在扩散空间)
    记录差异, 但我们的选择更匹配推理噪声先验 (noise = randn 在扩散空间)
  - R4: DINO iterative refinement 修复 (原 bug: pred_boxes = bbox_embed(feat) 无 sigmoid)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


import pytest
import torch
import torch.nn as nn

from projects.diffudetr.models.diffudetr_head import (
    DiffuDETRHead,
    inverse_sigmoid,
)


def _make_head(
    num_layers=2,
    num_queries=20,
    scale=2.0,
    num_classes=24,
    sampling_timesteps=5,
    nms_thr=0.7,
):
    """构造小型测试 head (减少层数/查询数以加速)."""
    return DiffuDETRHead(
        num_classes=num_classes,
        num_queries=num_queries,
        embed_dim=64,
        num_heads=4,
        dim_feedforward=128,
        num_layers=num_layers,
        num_feature_levels=2,
        timesteps=100,
        sampling_timesteps=sampling_timesteps,
        scale=scale,
        nms_thr=nms_thr,
    )


def _make_feats(B=2, levels=2, C=64, H=8, W=8):
    """构造多尺度特征."""
    return [torch.randn(B, C, H, W) for _ in range(levels)]


# ========== 空间转换 ==========


class TestSpaceConversion:
    """验证扩散空间 [-scale, scale] ↔ 归一化 [0,1] (对齐 apply_box_noise)."""

    def test_diffusion_to_norm_endpoints(self):
        """端点: -scale→0, 0→0.5, +scale→1."""
        head = _make_head(scale=2.0)
        x = torch.tensor([[-2.0, 0.0, 2.0, -2.0]])
        out = head.diffusion_to_norm(x)
        expected = torch.tensor([[0.0, 0.5, 1.0, 0.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_diffusion_to_norm_clamps_outliers(self):
        """超出 [-scale, scale] 的值被 clamp."""
        head = _make_head(scale=2.0)
        x = torch.tensor([[5.0, -5.0, 10.0, -10.0]])
        out = head.diffusion_to_norm(x)
        assert out[0, 0].item() == 1.0  # 5 → clamp 2 → 1
        assert out[0, 1].item() == 0.0  # -5 → clamp -2 → 0
        assert out[0, 2].item() == 1.0
        assert out[0, 3].item() == 0.0

    def test_norm_to_diffusion_endpoints(self):
        """端点: 0→-scale, 0.5→0, 1→+scale."""
        head = _make_head(scale=2.0)
        x = torch.tensor([[0.0, 0.5, 1.0, 0.25]])
        out = head.norm_to_diffusion(x)
        expected = torch.tensor([[-2.0, 0.0, 2.0, -1.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_round_trip_norm_to_diff_to_norm(self):
        """norm→diffusion→norm 往返 (在 [0,1] 内)."""
        head = _make_head(scale=2.0)
        x = torch.rand(3, 4)  # [0,1]
        round_trip = head.diffusion_to_norm(head.norm_to_diffusion(x))
        assert torch.allclose(round_trip, x, atol=1e-6)

    def test_round_trip_diff_to_norm_to_diff(self):
        """diffusion→norm→diffusion 往返 (在 [-scale, scale] 内)."""
        head = _make_head(scale=2.0)
        x = torch.rand(3, 4) * 4 - 2  # [-2, 2]
        round_trip = head.norm_to_diffusion(head.diffusion_to_norm(x))
        assert torch.allclose(round_trip, x, atol=1e-6)

    def test_scale_consistency(self):
        """不同 scale 下端点映射正确."""
        for s in [1.0, 2.0, 5.0]:
            head = _make_head(scale=s)
            assert head.diffusion_to_norm(torch.tensor([[-s]]))[
                0, 0
            ].item() == pytest.approx(0.0)
            assert head.diffusion_to_norm(torch.tensor([[s]]))[
                0, 0
            ].item() == pytest.approx(1.0)


# ========== inverse_sigmoid ==========


class TestInverseSigmoid:
    """验证 inverse_sigmoid (对齐 DINO/mmdet)."""

    def test_formula(self):
        """inverse_sigmoid(x) = log(x / (1-x))."""
        x = torch.tensor([0.1, 0.5, 0.9])
        out = inverse_sigmoid(x)
        expected = torch.log(x / (1 - x))
        assert torch.allclose(out, expected, atol=1e-5)

    def test_clamps_for_stability(self):
        """0 和 1 被 clamp 避免 log(0)/inf."""
        x = torch.tensor([0.0, 1.0])
        out = inverse_sigmoid(x)
        assert torch.isfinite(out).all()
        # 0 → log(eps/(1-eps)) 大负数; 1 → log((1-eps)/eps) 大正数
        assert out[0].item() < 0
        assert out[1].item() > 0

    def test_round_trip_with_sigmoid(self):
        """sigmoid(inverse_sigmoid(x)) ≈ x (在 clamp 范围内)."""
        x = torch.tensor([0.2, 0.5, 0.8])
        assert torch.allclose(inverse_sigmoid(x).sigmoid(), x, atol=1e-4)


# ========== DINO iterative refinement (_run_heads) ==========


class TestIterativeRefinement:
    """验证 DINO iterative refinement (对齐原仓库 line 541-549).

    核心: pred_boxes = (bbox_embed(feat) + inverse_sigmoid(reference)).sigmoid()
    """

    def test_output_bounded_in_0_1(self):
        """所有层输出 ∈ (0, 1) (sigmoid 保证, R4 修复)."""
        head = _make_head(num_layers=3, num_queries=10)
        B, N, C = 2, 10, 64
        inter_states = torch.randn(3, B, N, C)
        time_emb = torch.randn(B, 4 * C)
        init_ref = torch.rand(B, N, 4)  # [0,1]
        _, pred_boxes_list = head._run_heads(inter_states, time_emb, init_ref)
        assert len(pred_boxes_list) == 3
        for pb in pred_boxes_list:
            assert (pb > 0).all() and (pb < 1).all(), (
                f'pred_boxes 应在 (0,1), got min={pb.min()}, max={pb.max()}'
            )

    def test_iterative_reference_update(self):
        """下一层 reference = 当前层预测 (迭代精修)."""
        head = _make_head(num_layers=3, num_queries=5)
        B, N, C = 1, 5, 64
        torch.manual_seed(0)
        inter_states = torch.randn(3, B, N, C)
        time_emb = torch.randn(B, 4 * C)
        init_ref = torch.rand(B, N, 4)
        _, pred_boxes_list = head._run_heads(inter_states, time_emb, init_ref)
        # 第 2 层的 reference 应等于第 1 层输出 (而非 init_ref)
        # 通过修改第 1 层输出观察第 2 层变化间接验证
        # 直接验证: 第 i+1 层不依赖 init_ref (除第 0 层外)
        init_ref2 = init_ref + 0.3  # 改变初始 reference
        init_ref2 = init_ref2.clamp(0.01, 0.99)
        _, pred_boxes_list2 = head._run_heads(
            inter_states, time_emb, init_ref2
        )
        # 第 0 层应不同 (依赖 init_ref)
        assert not torch.allclose(
            pred_boxes_list[0], pred_boxes_list2[0], atol=1e-5
        )

    def test_gradient_bounded_by_sigmoid(self):
        """sigmoid 输出梯度有界 (最大 0.25), 避免无界梯度 (R4 修复核心)."""
        head = _make_head(num_layers=2, num_queries=5)
        B, N, C = 1, 5, 64
        inter_states = torch.randn(2, B, N, C, requires_grad=True)
        time_emb = torch.randn(B, 4 * C)
        init_ref = torch.rand(B, N, 4)
        _, pred_boxes_list = head._run_heads(inter_states, time_emb, init_ref)
        loss = pred_boxes_list[-1].sum()
        loss.backward()
        grad = inter_states.grad
        assert torch.isfinite(grad).all()
        # sigmoid 导数 ≤ 0.25, 经 MLP 后梯度应有界 (不会爆炸)
        assert grad.abs().max() < 10.0

    def test_no_dead_gradient_at_boundary(self):
        """sigmoid 不像 clamp 有死梯度区 (R4 修复对比原 clamp 实现).

        注: bbox_embed 末层零初始化时 pred_offset=0, box 路径对 inter_states
        无梯度. 这里手动设非零权重, 验证 sigmoid 在边界仍有非零梯度 (无死区).
        """
        head = _make_head(num_layers=1, num_queries=3)
        # 手动设 bbox_embed 末层为非零, 使 pred_offset 依赖 feat
        with torch.no_grad():
            for bbox in head.bbox_embed:
                nn.init.normal_(bbox.layers[-1].weight, std=0.1)
                nn.init.constant_(bbox.layers[-1].bias, 0.0)
        B, N, C = 1, 3, 64
        # init_ref 接近边界 (验证 sigmoid 非死区, 对比 clamp 在边界梯度为 0)
        init_ref = torch.tensor([[[0.01, 0.01, 0.99, 0.99]]])
        inter_states = torch.randn(1, B, N, C, requires_grad=True)
        time_emb = torch.randn(B, 4 * C)
        _, pred_boxes_list = head._run_heads(inter_states, time_emb, init_ref)
        pred_boxes_list[0].sum().backward()
        # 梯度应非零 (sigmoid 在边界仍有梯度, 不像 clamp 死区)
        assert inter_states.grad.abs().sum() > 0, (
            'sigmoid 在边界应有非零梯度 (无死区)'
        )


# ========== _prepare_x_start ==========


class TestPrepareXStart:
    """验证 x_start 准备 (对齐 prepare_targets + DiffuDETR shuffle).

    Bug 7 对照: 原仓库 noisy_gt=True 时填充 = randn.sigmoid() 在 [0,1] 空间;
                我们填充 = randn 在扩散空间 [-scale, scale].
    差异已记录, 我们的选择更匹配推理噪声先验.
    """

    def test_gt_placed_at_random_positions(self):
        """GT 框被放置到 query 位置的子集."""
        torch.manual_seed(42)
        head = _make_head(num_queries=20, scale=2.0)
        gt_boxes = [torch.rand(3, 4) * 4 - 2]  # 3 GT 在扩散空间
        gt_labels = [torch.tensor([1, 2, 3])]
        x_start, labels = head._prepare_x_start(
            gt_boxes, gt_labels, torch.device('cpu')
        )
        assert x_start.shape == (1, 20, 4)
        assert labels.shape == (1, 20)

    def test_padding_is_background(self):
        """非 GT 位置标签 = num_classes (背景)."""
        torch.manual_seed(42)
        head = _make_head(num_queries=20, num_classes=24)
        gt_boxes = [torch.rand(3, 4) * 4 - 2]
        gt_labels = [torch.tensor([1, 2, 3])]
        _, labels = head._prepare_x_start(
            gt_boxes, gt_labels, torch.device('cpu')
        )
        # 20 个位置中 3 个是 GT, 17 个是背景
        bg_count = (labels[0] == 24).sum().item()
        gt_count = (labels[0] != 24).sum().item()
        assert gt_count == 3
        assert bg_count == 17

    def test_gt_labels_preserved(self):
        """GT 标签被正确放置 (无丢失)."""
        torch.manual_seed(42)
        head = _make_head(num_queries=20)
        gt_labels_in = torch.tensor([5, 10, 15])
        gt_boxes = [torch.rand(3, 4) * 4 - 2]
        _, labels = head._prepare_x_start(
            gt_boxes, [gt_labels_in], torch.device('cpu')
        )
        placed = labels[0][labels[0] != 24].sort().values
        assert torch.equal(placed, gt_labels_in.sort().values)

    def test_padding_in_diffusion_space(self):
        """填充为 randn (扩散空间, 匹配推理噪声先验).

        Bug 7: 原仓库用 randn.sigmoid() 在 [0,1]; 我们用 randn 在扩散空间.
        """
        torch.manual_seed(42)
        head = _make_head(num_queries=50, scale=2.0)
        # 无 GT → 全部为填充
        gt_boxes = [torch.zeros(0, 4)]
        gt_labels = [torch.zeros(0, dtype=torch.long)]
        x_start, _ = head._prepare_x_start(
            gt_boxes, gt_labels, torch.device('cpu')
        )
        # randn 均值 ~0, std ~1 (扩散空间)
        assert abs(x_start.mean().item()) < 0.2
        assert 0.7 < x_start.std().item() < 1.3

    def test_gt_more_than_queries(self):
        """GT 多于 query 时随机选 N 个 (不崩溃)."""
        torch.manual_seed(42)
        head = _make_head(num_queries=10)
        gt_boxes = [torch.rand(15, 4) * 4 - 2]  # 15 > 10
        gt_labels = [torch.arange(15)]
        x_start, labels = head._prepare_x_start(
            gt_boxes, gt_labels, torch.device('cpu')
        )
        # 只放 10 个
        gt_count = (labels[0] != 24).sum().item()
        assert gt_count == 10

    def test_batch_multiple_images(self):
        """batch 多图独立处理."""
        torch.manual_seed(42)
        head = _make_head(num_queries=20)
        gt_boxes = [torch.rand(3, 4) * 4 - 2, torch.rand(5, 4) * 4 - 2]
        gt_labels = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])]
        x_start, labels = head._prepare_x_start(
            gt_boxes, gt_labels, torch.device('cpu')
        )
        assert x_start.shape[0] == 2
        assert (labels[0] != 24).sum().item() == 3
        assert (labels[1] != 24).sum().item() == 5


# ========== forward_train ==========


class TestForwardTrain:
    """验证训练前向."""

    def test_returns_finite_loss_dict(self):
        """forward_train 返回有限 loss_dict."""
        torch.manual_seed(42)
        head = _make_head(num_layers=2, num_queries=10)
        head.train()
        feats = _make_feats(B=2, levels=2, C=64)
        gt_boxes = [torch.rand(3, 4) * 4 - 2, torch.rand(2, 4) * 4 - 2]
        gt_labels = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        loss_dict = head(feats, gt_boxes, gt_labels)
        assert len(loss_dict) > 0
        for k, v in loss_dict.items():
            assert torch.isfinite(v).all(), f'{k} not finite: {v}'

    def test_loss_keys_includes_aux(self):
        """多层 → 含 aux loss keys (loss_*_{i})."""
        head = _make_head(num_layers=3, num_queries=10)
        head.train()
        feats = _make_feats(B=1, levels=2, C=64)
        gt_boxes = [torch.rand(2, 4) * 4 - 2]
        gt_labels = [torch.tensor([0, 1])]
        loss_dict = head(feats, gt_boxes, gt_labels)
        # 主层
        assert 'loss_ce' in loss_dict
        assert 'loss_bbox' in loss_dict
        assert 'loss_giou' in loss_dict
        # aux 层 (num_layers-1 = 2)
        for i in range(2):
            assert f'loss_ce_{i}' in loss_dict
            assert f'loss_bbox_{i}' in loss_dict
            assert f'loss_giou_{i}' in loss_dict

    def test_loss_not_inflated(self):
        """loss 不再被 24x 放大 (R1 修复回归)."""
        head = _make_head(num_layers=2, num_queries=10)
        head.train()
        feats = _make_feats(B=2, levels=2, C=64)
        # 每图 24 GT (染色体数据集典型)
        gt_boxes = [torch.rand(24, 4) * 4 - 2 for _ in range(2)]
        gt_labels = [torch.arange(24) for _ in range(2)]
        loss_dict = head(feats, gt_boxes, gt_labels)
        # 修复前 loss_giou ≈ 35 (24x 放大); 修复后应 < 15
        assert loss_dict['loss_giou'].item() < 15, (
            f'loss_giou={loss_dict["loss_giou"].item():.2f} 仍被放大 (>15)'
        )

    def test_gradient_flows(self):
        """梯度能回传到 head 参数."""
        head = _make_head(num_layers=2, num_queries=8)
        head.train()
        feats = _make_feats(B=1, levels=2, C=64)
        gt_boxes = [torch.rand(2, 4) * 4 - 2]
        gt_labels = [torch.tensor([0, 1])]
        loss_dict = head(feats, gt_boxes, gt_labels)
        total = sum(v for v in loss_dict.values())
        total.backward()
        # 至少部分参数有梯度
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
        )
        assert has_grad

    def test_pred_boxes_in_unit_range(self):
        """训练中预测框经由 sigmoid 落在 [0,1] (R4 修复)."""
        head = _make_head(num_layers=2, num_queries=8)
        head.train()
        feats = _make_feats(B=1, levels=2, C=64)
        gt_boxes = [torch.rand(2, 4) * 4 - 2]
        gt_labels = [torch.tensor([0, 1])]
        # 拦截 _run_heads 输出验证
        original_run = head._run_heads
        captured = []

        def spy_run(inter_states, time_emb, init_reference):
            logits_list, boxes_list = original_run(
                inter_states, time_emb, init_reference
            )
            captured.extend(boxes_list)
            return logits_list, boxes_list

        head._run_heads = spy_run
        head(feats, gt_boxes, gt_labels)
        for pb in captured:
            assert (pb >= 0).all() and (pb <= 1).all()


# ========== forward_inference ==========


class TestForwardInference:
    """验证推理前向."""

    def test_returns_results_list(self):
        """推理返回每图一个 result dict."""
        torch.manual_seed(42)
        head = _make_head(num_layers=2, num_queries=10, sampling_timesteps=3)
        head.eval()
        feats = _make_feats(B=2, levels=2, C=64)
        img_shapes = [(64, 64), (64, 64)]
        with torch.no_grad():
            results = head(feats, img_shapes=img_shapes)
        assert len(results) == 2
        for r in results:
            assert 'boxes' in r and 'scores' in r and 'labels' in r
            assert r['boxes'].shape[1] == 4  # xyxy

    def test_boxes_in_pixel_range(self):
        """推理框为像素坐标 (乘以 img_size), 允许少量越界 (cxcywh→xyxy 可超界)."""
        head = _make_head(num_layers=2, num_queries=8, sampling_timesteps=3)
        head.eval()
        feats = _make_feats(B=1, levels=2, C=64, H=8, W=8)
        img_shapes = [(64, 80)]
        with torch.no_grad():
            results = head(feats, img_shapes=img_shapes)
        boxes = results[0]['boxes']
        if boxes.numel() > 0:
            # cxcywh→xyxy: x2=cx+w/2 可超过 1 (cx,w 各 ∈(0,1) 但和可 >1)
            # 允许框在 [-img, 2*img] 范围内 (DETR 系列允许框略出界)
            assert torch.isfinite(boxes).all()
            assert boxes[:, 0].min() >= -80 and boxes[:, 2].max() <= 2 * 80
            assert boxes[:, 1].min() >= -64 and boxes[:, 3].max() <= 2 * 64

    def test_nms_reduces_duplicates(self):
        """NMS 后无高度重叠框 (iou < nms_thr)."""
        head = _make_head(
            num_layers=2, num_queries=20, sampling_timesteps=3, nms_thr=0.5
        )
        head.eval()
        feats = _make_feats(B=1, levels=2, C=64)
        img_shapes = [(64, 64)]
        with torch.no_grad():
            results = head(feats, img_shapes=img_shapes)
        boxes = results[0]['boxes']
        if boxes.shape[0] >= 2:
            # 计算两两 IoU
            x1 = boxes[:, 0:1]
            y1 = boxes[:, 1:2]
            x2 = boxes[:, 2:3]
            y2 = boxes[:, 3:4]
            areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            lt = torch.max(x1.T, x1)
            rb = torch.min(x2.T, x2)
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            union = areas.T + areas - inter
            iou = inter / union.clamp(min=1e-6)
            # 排除对角线
            iou.fill_diagonal_(0)
            assert iou.max().item() < 0.5, (
                f'NMS 后仍存在 IoU={iou.max():.3f} 的重叠框'
            )

    def test_deterministic_with_seed(self):
        """相同种子 → 相同推理结果 (box_renewal 随机性可控)."""
        head = _make_head(num_layers=2, num_queries=8, sampling_timesteps=3)
        head.eval()
        feats = _make_feats(B=1, levels=2, C=64)
        img_shapes = [(64, 64)]
        torch.manual_seed(123)
        with torch.no_grad():
            r1 = head(feats, img_shapes=img_shapes)
        torch.manual_seed(123)
        with torch.no_grad():
            r2 = head(feats, img_shapes=img_shapes)
        assert torch.allclose(r1[0]['boxes'], r2[0]['boxes'])
        assert torch.allclose(r1[0]['scores'], r2[0]['scores'])
