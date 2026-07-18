"""SetDiff 修复测试 — snr_scale 应用 + 损失函数方案A + 时间嵌入缩放

TDD 红绿重构: 先写测试 (RED), 再实现 (GREEN), 最后重构。

覆盖三大修复:
1. snr_scale 应用 (根因修复): GT 缩放到 [-s, +s] 匹配 N(0,1) 噪声, 推理逆缩放
2. 损失函数方案 A: loss_bbox(L1, w=5) + loss_giou(GIoU, w=2), 删除 loss_diff, loss_cls=2
3. 时间嵌入缩放: t * 1000 后传入 time_embed (对齐 LDMDet)
"""

import torch

from setdiff.criterion.set_loss import SetCriterion
from setdiff.models.set_head import JointDiffusionHead

# ============================================================
# 修复 2: 损失函数方案 A — 独立测试 (不依赖 mmdet)
# ============================================================


class TestSetCriterionLossA:
    """SetCriterion 方案 A: 2:5:2, 删除 loss_diff, 拆分 loss_box 为 bbox+giou"""

    def test_default_weights_are_2_5_2(self):
        """默认权重应为 cls=2, bbox=5, giou=2 (对齐 LDMDet/DiffusionDet)"""
        criterion = SetCriterion(num_classes=24)
        assert criterion.weight_dict == {
            'loss_cls': 2.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
        }

    def test_loss_diff_removed_from_default_weights(self):
        """loss_diff 不应出现在默认权重中"""
        criterion = SetCriterion(num_classes=24)
        assert 'loss_diff' not in criterion.weight_dict
        assert 'loss_box' not in criterion.weight_dict

    def test_loss_dict_keys_are_cls_bbox_giou(self):
        """forward 返回的 loss_dict 应包含 loss_cls, loss_bbox, loss_giou"""
        criterion = SetCriterion(num_classes=24)
        B, N, C = 2, 10, 24
        outputs = {
            'pred_logits': torch.randn(B, N, C),
            'pred_boxes': torch.randn(B, N, 4),
        }
        targets = {
            'matched_boxes': torch.randn(B, N, 4),
            'matched_labels': torch.randint(-1, C, (B, N)),
        }
        loss_dict, total = criterion(outputs, targets)
        assert set(loss_dict.keys()) == {'loss_cls', 'loss_bbox', 'loss_giou'}
        assert 'loss_diff' not in loss_dict
        assert 'loss_box' not in loss_dict

    def test_loss_bbox_is_l1_loss(self):
        """loss_bbox 应为 L1 损失 (不是 L2/MSE)"""
        criterion = SetCriterion(num_classes=24)
        B, N, C = 1, 5, 24
        # 所有 slot 都是 matched (label >= 0)
        outputs = {
            'pred_logits': torch.randn(B, N, C),
            'pred_boxes': torch.zeros(B, N, 4),
        }
        targets = {
            'matched_boxes': torch.ones(B, N, 4),
            'matched_labels': torch.zeros(B, N, dtype=torch.long),
        }
        loss_dict, _ = criterion(outputs, targets)
        # L1 of [0,0,0,0] vs [1,1,1,1] = 1.0 per element, mean over 4 = 1.0
        # 但实现用 sum/num_pos, 所以是 4*1.0/5 = 0.8
        # 实际: sum(|1-0|) = 4*5 = 20, /num_pos=5 → 4.0
        # 每个slot 4维, 5个slot, sum = 5*4 = 20, /5 = 4.0
        assert torch.isclose(
            loss_dict['loss_bbox'], torch.tensor(4.0), atol=1e-5
        )

    def test_loss_giou_is_giou_loss(self):
        """loss_giou 应为 1 - GIoU"""
        criterion = SetCriterion(num_classes=24)
        B, N, C = 1, 3, 24
        # 完全重叠的框 → GIoU=1 → loss_giou=0
        boxes = torch.tensor(
            [
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.3, 0.3, 0.1, 0.1],
                    [0.7, 0.7, 0.15, 0.15],
                ]
            ]
        )
        outputs = {
            'pred_logits': torch.randn(B, N, C),
            'pred_boxes': boxes.clone(),
        }
        targets = {
            'matched_boxes': boxes.clone(),
            'matched_labels': torch.zeros(B, N, dtype=torch.long),
        }
        loss_dict, _ = criterion(outputs, targets)
        assert torch.isclose(
            loss_dict['loss_giou'], torch.tensor(0.0), atol=1e-5
        )

    def test_loss_weights_applied_correctly(self):
        """total loss 应为 sum(loss_i * weight_i)"""
        criterion = SetCriterion(num_classes=24)
        B, N, C = 1, 5, 24
        outputs = {
            'pred_logits': torch.randn(B, N, C),
            'pred_boxes': torch.randn(B, N, 4),
        }
        targets = {
            'matched_boxes': torch.randn(B, N, 4),
            'matched_labels': torch.zeros(B, N, dtype=torch.long),
        }
        loss_dict, total = criterion(outputs, targets)
        expected = (
            loss_dict['loss_cls'] * 2.0
            + loss_dict['loss_bbox'] * 5.0
            + loss_dict['loss_giou'] * 2.0
        )
        assert torch.isclose(total, expected, atol=1e-5)

    def test_unmatched_slots_excluded_from_bbox_giou(self):
        """label=-1 的 slot 不应参与 loss_bbox/loss_giou"""
        criterion = SetCriterion(num_classes=24)
        B, N, C = 1, 4, 24
        outputs = {
            'pred_logits': torch.randn(B, N, C),
            'pred_boxes': torch.tensor(
                [
                    [
                        [0.5, 0.5, 0.2, 0.2],
                        [0.9, 0.9, 0.9, 0.9],  # unmatched, 应被排除
                        [0.3, 0.3, 0.1, 0.1],
                        [0.8, 0.8, 0.8, 0.8],
                    ]
                ]  # unmatched, 应被排除
            ),
        }
        targets = {
            'matched_boxes': torch.tensor(
                [
                    [
                        [0.5, 0.5, 0.2, 0.2],
                        [0.0, 0.0, 0.0, 0.0],  # unmatched slot 的 GT (不参与)
                        [0.3, 0.3, 0.1, 0.1],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ]  # unmatched slot 的 GT (不参与)
            ),
            'matched_labels': torch.tensor([[0, -1, 0, -1]]),
        }
        loss_dict, _ = criterion(outputs, targets)
        # 只有 2 个 matched slot, 且 pred==gt, 所以 loss_bbox=0, loss_giou=0
        assert torch.isclose(
            loss_dict['loss_bbox'], torch.tensor(0.0), atol=1e-5
        )
        assert torch.isclose(
            loss_dict['loss_giou'], torch.tensor(0.0), atol=1e-5
        )


# ============================================================
# 修复 1: snr_scale 应用 — 核心转换函数测试
# ============================================================


class TestSnrScaleTransform:
    """snr_scale 转换函数: [0,1] ↔ [-snr_scale, +snr_scale]"""

    def test_snr_scale_stored_in_head(self):
        """JointDiffusionHead 应存储 snr_scale 为 self.snr_scale"""
        head = JointDiffusionHead(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=24,
            snr_scale=2.0,
        )
        assert hasattr(head, 'snr_scale')
        assert head.snr_scale == 2.0

    def test_gt_to_diffusion_space_round_trip(self):
        """GT [0,1] → diffusion [-s, s] → [0,1] 往返一致"""
        from experiments.mmdet_bridge.setdiff_detector import SetDiffDetector

        s = 2.0
        gt_norm = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.1, 0.9, 0.05, 0.3]])
        # [0,1] → [-s, s]
        gt_diffusion = SetDiffDetector.gt_to_diffusion_space(
            gt_norm, snr_scale=s
        )
        # [-s, s] → [0,1]
        gt_recovered = SetDiffDetector.diffusion_to_norm_space(
            gt_diffusion, snr_scale=s
        )
        assert torch.allclose(gt_norm, gt_recovered, atol=1e-6)

    def test_gt_to_diffusion_space_value_range(self):
        """[0,1] 的 GT 缩放后应在 [-snr_scale, +snr_scale]"""
        from experiments.mmdet_bridge.setdiff_detector import SetDiffDetector

        s = 2.0
        gt_norm = torch.tensor(
            [[0.0, 1.0, 0.5, 0.25]]  # 边界值
        )
        gt_diffusion = SetDiffDetector.gt_to_diffusion_space(
            gt_norm, snr_scale=s
        )
        # 0 → -2, 1 → +2, 0.5 → 0, 0.25 → -1
        expected = torch.tensor([[-2.0, 2.0, 0.0, -1.0]])
        assert torch.allclose(gt_diffusion, expected, atol=1e-6)

    def test_diffusion_to_norm_space_clamps(self):
        """推理逆缩放应 clamp 到 [-s, s] 防止超出 [0,1]"""
        from experiments.mmdet_bridge.setdiff_detector import SetDiffDetector

        s = 2.0
        # 模型输出超出 [-s, s] 范围 (训练初期可能发生)
        pred_diffusion = torch.tensor(
            [[3.0, -3.0, 0.5, -0.5]]  # 超出 [-2, 2]
        )
        pred_norm = SetDiffDetector.diffusion_to_norm_space(
            pred_diffusion, snr_scale=s
        )
        # 3.0 → clamp 2.0 → (2/2+1)/2 = 1.0
        # -3.0 → clamp -2.0 → (-2/2+1)/2 = 0.0
        # 0.5 → (0.5/2+1)/2 = 0.625
        # -0.5 → (-0.5/2+1)/2 = 0.375
        expected = torch.tensor([[1.0, 0.0, 0.625, 0.375]])
        assert torch.allclose(pred_norm, expected, atol=1e-6)
        assert (pred_norm >= 0).all() and (pred_norm <= 1).all()


# ============================================================
# 修复 3: 时间嵌入缩放 — t * 1000
# ============================================================


class TestTimeEmbedScaling:
    """time_embed 应接收 t * 1000 (对齐 LDMDet)"""

    def test_time_embed_receives_scaled_t(self, monkeypatch):
        """验证 time_embed 接收的输入是 t * 1000 而非 t"""
        head = JointDiffusionHead(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=24,
            snr_scale=2.0,
        )

        # 捕获 time_embed 的输入
        captured_inputs = []
        original_forward = head.time_embed.forward

        def capture_forward(x):
            captured_inputs.append(x.clone())
            return original_forward(x)

        monkeypatch.setattr(head.time_embed, 'forward', capture_forward)

        # 训练前向
        B = 2
        image_features = torch.randn(B, 100, 64)
        gt_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]]) for _ in range(B)]
        gt_labels = [torch.tensor([0]) for _ in range(B)]
        head.train()
        head(image_features, gt_boxes, gt_labels)

        # 验证 time_embed 接收的 t 已被缩放
        assert len(captured_inputs) >= 1
        t_input = captured_inputs[0]
        # t ~ U(0,1), 缩放后 t*1000 ~ U(0,1000)
        # 检查值域: 应在 [0, 1000] 而非 [0, 1]
        assert t_input.max() <= 1000.0 + 1e-3
        assert t_input.min() >= 0.0 - 1e-3
        # 如果未缩放, max 会 < 1; 缩放后应远大于 1 (概率极高)
        assert t_input.max() > 1.0, (
            f't 似乎未被缩放: max={t_input.max()} (应 > 1)'
        )
