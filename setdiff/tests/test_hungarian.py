"""测试 setdiff.matching.hungarian — HungarianMatcher"""

import torch
import pytest
from setdiff.matching.hungarian import HungarianMatcher


class TestHungarianMatcher:
    @pytest.fixture
    def matcher(self):
        return HungarianMatcher(cost_type='l2')

    def test_basic_matching_shapes(self, matcher):
        """N=10, M=5: 输出形状正确"""
        N, M = 10, 5
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 10, (M,))
        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )
        assert matched_boxes.shape == (N, 4)
        assert matched_labels.shape == (N,)
        assert matched_labels.dtype == torch.long

    def test_all_gt_matched(self, matcher):
        """所有 M 个 GT 都应被匹配到某个 noise slot"""
        N, M = 10, 5
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 10, (M,))
        _, matched_labels = matcher.match(noise, gt_boxes, gt_labels)
        # 恰好有 M 个 slot 的 label >= 0
        num_matched = (matched_labels >= 0).sum().item()
        assert num_matched == M

    def test_unmatched_slots_get_negative_one(self, matcher):
        """未匹配的 slot 应得 label=-1"""
        N, M = 10, 5
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 10, (M,))
        _, matched_labels = matcher.match(noise, gt_boxes, gt_labels)
        # N-M 个 slot 应为 -1
        num_unmatched = (matched_labels == -1).sum().item()
        assert num_unmatched == N - M

    def test_unmatched_boxes_equal_noise(self, matcher):
        """未匹配 slot 的 matched_box 应等于 noise 自身 (velocity=0)"""
        N, M = 8, 3
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 10, (M,))
        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )
        unmatched_mask = matched_labels == -1
        # 未匹配 slot 的 box 应等于 noise
        assert torch.allclose(
            matched_boxes[unmatched_mask], noise[unmatched_mask]
        )

    def test_matched_labels_are_correct(self, matcher):
        """匹配 slot 的 label 应来自 gt_labels 的某个值"""
        N, M = 6, 3
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.tensor([2, 5, 7])
        _, matched_labels = matcher.match(noise, gt_boxes, gt_labels)
        matched = matched_labels[matched_labels >= 0]
        # 所有匹配的 label 应来自 {2, 5, 7}
        label_set = set(matched.tolist())
        assert label_set.issubset({2, 5, 7})
        # 每个 GT label 恰好出现一次
        assert len(label_set) == M

    def test_optimal_assignment_minimizes_cost(self, matcher):
        """验证匹配确实最小化总 L2 代价"""
        N, M = 4, 2
        # 构造 noise 和 gt 使得最优匹配是明确的
        noise = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],  # 接近 gt[0]
                [10.0, 10.0, 10.0, 10.0],  # 远离所有
                [1.0, 1.0, 1.0, 1.0],  # 接近 gt[1]
                [20.0, 20.0, 20.0, 20.0],  # 远离所有
            ]
        )
        gt_boxes = torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1],  # 接近 noise[0]
                [1.1, 1.1, 1.1, 1.1],  # 接近 noise[2]
            ]
        )
        gt_labels = torch.tensor([0, 1])
        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )
        # noise[0] 应匹配 gt[0] (label=0)
        assert matched_labels[0].item() == 0
        # noise[2] 应匹配 gt[1] (label=1)
        assert matched_labels[2].item() == 1
        # noise[1], noise[3] 未匹配
        assert matched_labels[1].item() == -1
        assert matched_labels[3].item() == -1

    def test_empty_gt(self, matcher):
        """M=0 时所有 slot 未匹配"""
        N = 5
        noise = torch.randn(N, 4)
        gt_boxes = torch.zeros(0, 4)
        gt_labels = torch.zeros(0, dtype=torch.long)
        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )
        assert matched_boxes.shape == (N, 4)
        assert (matched_labels == -1).all()
        # 全部等于 noise 自身
        assert torch.allclose(matched_boxes, noise)

    def test_n_equals_m(self, matcher):
        """N == M 时所有 slot 都匹配"""
        N = M = 5
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 10, (M,))
        _, matched_labels = matcher.match(noise, gt_boxes, gt_labels)
        assert (matched_labels >= 0).all()

    def test_batch_matching_shapes(self, matcher):
        """批量匹配输出形状正确"""
        B, N = 2, 10
        noise_batch = torch.randn(B, N, 4)
        gt_boxes_list = [torch.randn(5, 4), torch.randn(3, 4)]
        gt_labels_list = [
            torch.randint(0, 10, (5,)),
            torch.randint(0, 10, (3,)),
        ]
        matched_boxes, matched_labels = matcher.match_batch(
            noise_batch, gt_boxes_list, gt_labels_list
        )
        assert matched_boxes.shape == (B, N, 4)
        assert matched_labels.shape == (B, N)

    def test_batch_matching_correct_counts(self, matcher):
        """批量匹配: 每张图的匹配数等于 GT 数"""
        B, N = 2, 10
        noise_batch = torch.randn(B, N, 4)
        gt_boxes_list = [torch.randn(5, 4), torch.randn(3, 4)]
        gt_labels_list = [
            torch.randint(0, 10, (5,)),
            torch.randint(0, 10, (3,)),
        ]
        _, matched_labels = matcher.match_batch(
            noise_batch, gt_boxes_list, gt_labels_list
        )
        # 第一张图 5 个匹配
        assert (matched_labels[0] >= 0).sum().item() == 5
        # 第二张图 3 个匹配
        assert (matched_labels[1] >= 0).sum().item() == 3

    def test_batch_with_empty_gt(self, matcher):
        """批量匹配: 某张图 GT 为空"""
        B, N = 2, 8
        noise_batch = torch.randn(B, N, 4)
        gt_boxes_list = [torch.randn(3, 4), torch.zeros(0, 4)]
        gt_labels_list = [
            torch.randint(0, 10, (3,)),
            torch.zeros(0, dtype=torch.long),
        ]
        _, matched_labels = matcher.match_batch(
            noise_batch, gt_boxes_list, gt_labels_list
        )
        # 第一张图 3 个匹配
        assert (matched_labels[0] >= 0).sum().item() == 3
        # 第二张图全部未匹配
        assert (matched_labels[1] == -1).all()

    def test_l1_cost_type(self):
        """cost_type='l1' 也能正常工作"""
        matcher = HungarianMatcher(cost_type='l1')
        N, M = 6, 3
        noise = torch.randn(N, 4)
        gt_boxes = torch.randn(M, 4)
        gt_labels = torch.randint(0, 10, (M,))
        matched_boxes, matched_labels = matcher.match(
            noise, gt_boxes, gt_labels
        )
        assert matched_boxes.shape == (N, 4)
        assert matched_labels.shape == (N,)
        assert (matched_labels >= 0).sum().item() == M
