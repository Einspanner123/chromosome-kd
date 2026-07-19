"""方案 A/B Matcher 单元测试

验证两种修复策略, 解决 unmatched slot 训练-推理分布不匹配 (mAP=0 根因):
  - 方案 A (RandomMatcher): 放弃 Hungarian, 所有 slot 随机分配 GT,
    对齐 LDMDet `_couple_single_image` 的 `torch.randint(0, num_gt, (N,))`.
  - 方案 B (HungarianMatcher + unmatched_strategy='random_gt'):
    保留 Hungarian global coupled matching (matched slot 最优匹配),
    unmatched slot 分配随机 GT (保证所有 slot 有监督).

核心断言 (两个方案共同):
  1. 所有 slot 的 label >= 0 (没有 -1 padding)
  2. 所有 slot 的 matched_boxes 来自 GT 集合 (不是 noise)
  3. M=0 边界: 仍然所有 slot label=-1, matched_boxes=noise
  4. M>N 边界: 截断到 N 个 GT
  5. 可复现: 同 seed 同结果

方案 B 额外断言:
  - matched slot (Hungarian 分配) 的 GT 严格一对一 (无重复)
  - unmatched slot 的 GT 来自 GT 集合 (允许重复, 随机采样)
"""

import pytest
import torch

from setdiff.matching.hungarian import HungarianMatcher
from setdiff.matching.random_matcher import RandomMatcher


# ============================================================
# 方案 A: RandomMatcher
# ============================================================


class TestRandomMatcher:
    """方案 A: 所有 slot 随机分配 GT."""

    def test_random_matcher_exists(self):
        """RandomMatcher 类可导入可实例化."""
        matcher = RandomMatcher()
        assert matcher is not None

    def test_all_slots_get_gt_label(self):
        """所有 slot 的 label >= 0 (无 -1)."""
        torch.manual_seed(42)
        matcher = RandomMatcher()
        N, M = 300, 10
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        assert matched_labels.shape == (N,)
        assert (matched_labels >= 0).all(), "所有 slot 必须有真实 GT label"
        # label 一定是 gt_labels 的子集
        assert set(matched_labels.tolist()).issubset(set(gt_labels.tolist()))

    def test_all_slots_boxes_from_gt(self):
        """所有 slot 的 matched_boxes 来自 GT 集合 (不是 noise)."""
        torch.manual_seed(42)
        matcher = RandomMatcher()
        N, M = 300, 10
        noise = torch.randn(N, 4) * 100  # 故意放大 noise 与 GT 区分
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        # 每个 slot 的 matched_boxes 必须等于某个 GT
        for i in range(N):
            distances = (gt_boxes - matched_boxes[i].unsqueeze(0)).pow(2).sum(-1)
            min_dist = distances.min().item()
            assert min_dist < 1e-6, (
                f"slot {i} matched_boxes 不来自 GT 集合, min_dist={min_dist}"
            )

    def test_zero_gt_edge_case(self):
        """M=0 边界: 所有 slot label=-1, matched_boxes=noise."""
        matcher = RandomMatcher()
        N = 300
        noise = torch.randn(N, 4)
        gt_boxes = torch.empty(0, 4)
        gt_labels = torch.empty(0, dtype=torch.long)

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        assert (matched_labels == -1).all()
        assert torch.allclose(matched_boxes, noise)

    def test_more_gt_than_slots_edge_case(self):
        """M>N 边界: 仍然所有 slot 都有 GT (从 M 个 GT 中随机采样 N 个)."""
        torch.manual_seed(42)
        matcher = RandomMatcher()
        N, M = 10, 50
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        assert (matched_labels >= 0).all()
        assert matched_boxes.shape == (N, 4)
        assert matched_labels.shape == (N,)

    def test_batch_matching(self):
        """批量匹配: 不同图像独立采样."""
        torch.manual_seed(42)
        matcher = RandomMatcher()
        B, N = 2, 300
        noise_batch = torch.randn(B, N, 4)
        gt_boxes_list = [torch.randn(5, 4), torch.randn(8, 4)]
        gt_labels_list = [
            torch.randint(0, 24, (5,)),
            torch.randint(0, 24, (8,)),
        ]

        matched_boxes, matched_labels = matcher.match_batch(
            noise_batch, gt_boxes_list, gt_labels_list
        )

        assert matched_boxes.shape == (B, N, 4)
        assert matched_labels.shape == (B, N)
        assert (matched_labels >= 0).all()

    def test_reproducibility(self):
        """同 seed 同结果."""
        matcher = RandomMatcher()

        torch.manual_seed(123)
        noise = torch.randn(300, 4)
        gt_boxes = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))

        torch.manual_seed(456)
        mb1, ml1 = matcher.match(noise, gt_boxes, gt_labels)
        torch.manual_seed(456)
        mb2, ml2 = matcher.match(noise, gt_boxes, gt_labels)

        assert torch.equal(ml1, ml2)
        assert torch.equal(mb1, mb2)

    def test_all_slots_have_velocity(self):
        """所有 slot 的 x_start=GT, x_t=(1-t)*GT + t*noise, velocity=noise-GT != 0.

        这是方案 A 的核心: 所有 slot 都在 [GT, noise] 插值轨迹上,
        训练时 box_head 对所有 slot 都有监督信号.
        """
        torch.manual_seed(42)
        matcher = RandomMatcher()
        N, M = 300, 10
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, _ = matcher.match(noise, gt_boxes, gt_labels)

        # 逐 slot 验证: velocity = noise - matched_boxes != 0
        # (即每个 slot 的 x_start 不等于 noise, 在 [GT, noise] 轨迹上)
        velocity = noise - matched_boxes  # [N, 4]
        velocity_norm = velocity.abs().sum(-1)  # [N]
        # GT 与 noise 来自不同分布, velocity 应显著非零
        assert (velocity_norm > 1e-3).all(), (
            f"所有 slot velocity 必须非零, "
            f"最小 velocity_norm={velocity_norm.min().item():.6f}"
        )


# ============================================================
# 方案 B: HungarianMatcher + unmatched_strategy='random_gt'
# ============================================================


class TestHungarianMatcherPlanB:
    """方案 B: 保留 Hungarian, unmatched slot 分配随机 GT."""

    def test_default_behavior_unchanged(self):
        """默认 unmatched_strategy='noise' 保持原行为 (向后兼容)."""
        torch.manual_seed(42)
        matcher = HungarianMatcher()  # 默认
        N, M = 300, 10
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        # 原行为: 只有 M 个 slot 有 label, 其余 -1
        num_matched = (matched_labels >= 0).sum().item()
        assert num_matched == M, f"默认应只有 M={M} 个 matched slot, 实际 {num_matched}"

    def test_random_gt_strategy_all_slots_have_gt(self):
        """方案 B: unmatched_strategy='random_gt', 所有 slot 都有 GT."""
        torch.manual_seed(42)
        matcher = HungarianMatcher(unmatched_strategy='random_gt')
        N, M = 300, 10
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        assert (matched_labels >= 0).all(), "方案 B: 所有 slot 必须有真实 GT label"

    def test_random_gt_strategy_matched_slots_cover_all_gt(self):
        """方案 B: Hungarian matched slot 覆盖所有 M 个 GT (一对一分配).

        Hungarian 是 one-to-one, M 个 GT 应该被分配到 M 个不同 slot.
        注意: unmatched slot 随机采样可能选到已被 Hungarian 分配的 GT,
        所以每个 GT 出现次数 >= 1 (而不是恰好 1).
        """
        torch.manual_seed(42)
        matcher = HungarianMatcher(unmatched_strategy='random_gt')
        N, M = 300, 10
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        # 验证: 每个 GT 至少被 Hungarian 分配到 1 个 slot
        # (unmatched slot 随机采样可能让某些 GT 出现多次, 但至少 1 次)
        for j in range(M):
            dist = (matched_boxes - gt_boxes[j].unsqueeze(0)).pow(2).sum(-1)
            count = (dist < 1e-6).sum().item()
            assert count >= 1, (
                f"GT {j} 应该被 Hungarian 至少分配到 1 个 slot, 实际 {count}"
            )

    def test_random_gt_strategy_matched_labels_consistent_with_boxes(self):
        """方案 B: matched_labels 与 matched_boxes 的 GT label 严格一致.

        对每个 slot, 若 matched_boxes == gt_boxes[j], 则 matched_labels == gt_labels[j].
        """
        torch.manual_seed(42)
        matcher = HungarianMatcher(unmatched_strategy='random_gt')
        N, M = 300, 10
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        for i in range(N):
            # 找 matched_boxes[i] 对应的 GT index
            dist = (gt_boxes - matched_boxes[i].unsqueeze(0)).pow(2).sum(-1)
            gt_idx = dist.argmin().item()
            assert dist[gt_idx].item() < 1e-6, (
                f"slot {i} matched_boxes 不来自 GT 集合"
            )
            assert matched_labels[i].item() == gt_labels[gt_idx].item(), (
                f"slot {i} label {matched_labels[i].item()} != "
                f"GT[{gt_idx}] label {gt_labels[gt_idx].item()}"
            )

    def test_random_gt_strategy_unmatched_from_gt_set(self):
        """方案 B: unmatched slot 的 boxes 也来自 GT 集合 (允许重复)."""
        torch.manual_seed(42)
        matcher = HungarianMatcher(unmatched_strategy='random_gt')
        N, M = 300, 10
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 24, (M,))

        matched_boxes, _ = matcher.match(noise, gt_boxes, gt_labels)

        # 所有 slot 的 matched_boxes 必须来自 GT 集合
        for i in range(N):
            distances = (gt_boxes - matched_boxes[i].unsqueeze(0)).pow(2).sum(-1)
            min_dist = distances.min().item()
            assert min_dist < 1e-6, (
                f"slot {i} matched_boxes 不来自 GT 集合 (方案 B), min_dist={min_dist}"
            )

    def test_random_gt_zero_gt_edge_case(self):
        """方案 B: M=0 边界, 所有 slot label=-1, matched_boxes=noise."""
        matcher = HungarianMatcher(unmatched_strategy='random_gt')
        N = 300
        noise = torch.randn(N, 4)
        gt_boxes = torch.empty(0, 4)
        gt_labels = torch.empty(0, dtype=torch.long)

        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )

        # M=0 时无 GT 可分配, 回退到原行为
        assert (matched_labels == -1).all()
        assert torch.allclose(matched_boxes, noise)

    def test_random_gt_batch_matching(self):
        """方案 B: 批量匹配."""
        torch.manual_seed(42)
        matcher = HungarianMatcher(unmatched_strategy='random_gt')
        B, N = 2, 300
        noise_batch = torch.randn(B, N, 4)
        gt_boxes_list = [torch.randn(5, 4), torch.randn(8, 4)]
        gt_labels_list = [
            torch.randint(0, 24, (5,)),
            torch.randint(0, 24, (8,)),
        ]

        matched_boxes, matched_labels = matcher.match_batch(
            noise_batch, gt_boxes_list, gt_labels_list
        )

        assert matched_boxes.shape == (B, N, 4)
        assert matched_labels.shape == (B, N)
        assert (matched_labels >= 0).all()

    def test_random_gt_reproducibility(self):
        """方案 B: 同 seed 同结果."""
        matcher = HungarianMatcher(unmatched_strategy='random_gt')

        torch.manual_seed(789)
        noise = torch.randn(300, 4)
        gt_boxes = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))

        torch.manual_seed(999)
        mb1, ml1 = matcher.match(noise, gt_boxes, gt_labels)
        torch.manual_seed(999)
        mb2, ml2 = matcher.match(noise, gt_boxes, gt_labels)

        assert torch.equal(ml1, ml2)
        assert torch.equal(mb1, mb2)


# ============================================================
# 集成测试: matcher 在 set_head 中的选择
# ============================================================


class TestMatcherSelection:
    """set_head 支持通过 matcher_type 选择 matcher."""

    def test_set_head_accepts_matcher_type(self):
        """JointDiffusionHead 接受 matcher_type 参数 ('hungarian' | 'random')."""
        from setdiff.models.set_head import JointDiffusionHead

        head_a = JointDiffusionHead(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=24,
            matcher_type='random',  # 方案 A
        )
        assert head_a.matcher.__class__.__name__ == 'RandomMatcher'

        head_b = JointDiffusionHead(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=24,
            matcher_type='hungarian',
            unmatched_strategy='random_gt',  # 方案 B
        )
        assert head_b.matcher.__class__.__name__ == 'HungarianMatcher'
        assert head_b.matcher.unmatched_strategy == 'random_gt'

    def test_set_head_default_matcher_unchanged(self):
        """默认 matcher_type='hungarian', unmatched_strategy='noise' (向后兼容)."""
        from setdiff.models.set_head import JointDiffusionHead

        head = JointDiffusionHead(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=24,
        )
        assert head.matcher.__class__.__name__ == 'HungarianMatcher'
        assert head.matcher.unmatched_strategy == 'noise'
