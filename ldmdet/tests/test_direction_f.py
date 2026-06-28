"""方向 F 测试: 扩散框架创新 (Diffusion Framework Innovation)

F1: StructuredPrior — 结构化噪声先验 (用训练集框分布代替高斯噪声)
F3: 数量约束 (num_proposals 100 + 计数分支, 配置层验证)
"""

import pytest
import torch

from ldmdet.diffusion.structured_prior import StructuredPrior


class TestStructuredPrior:
    """F1: 结构化噪声先验单元测试"""

    @pytest.fixture
    def prior(self):
        # 模拟训练集统计: 2 个高斯分量
        # 分量 0: 中心 [0.3, 0.4, 0.1, 0.1], 小框
        # 分量 1: 中心 [0.5, 0.5, 0.2, 0.2], 大框
        means = torch.tensor([[0.3, 0.4, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]])
        stds = torch.tensor([[0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.05, 0.05]])
        weights = torch.tensor([0.5, 0.5])
        return StructuredPrior(means=means, stds=stds, weights=weights)

    def test_sample_shape(self, prior):
        """F1: 采样输出 shape 应为 (n, 4)"""
        samples = prior.sample(100)
        assert samples.shape == (100, 4)

    def test_sample_range(self, prior):
        """F1: 采样值应在合理范围内 (归一化坐标 0~1)"""
        samples = prior.sample(1000)
        assert samples.min() >= -0.5  # 允许少量超出
        assert samples.max() <= 1.5

    def test_not_standard_gaussian(self, prior):
        """F1: 采样分布不应是标准高斯 N(0,1)"""
        samples = prior.sample(10000)
        # 均值应接近数据统计的均值, 而非 0
        mean = samples.mean(dim=0)
        # 分量 0 均值 [0.3, 0.4, 0.1, 0.1], 分量 1 [0.5, 0.5, 0.2, 0.2]
        # 混合均值 ≈ [0.4, 0.45, 0.15, 0.15]
        assert mean[0] > 0.2  # 不接近 0
        assert mean[1] > 0.2

    def test_from_statistics(self):
        """F1: 应能从框统计构建 (from_statistics 类方法)"""
        # 模拟训练集框 (cxcywh 归一化)
        boxes = torch.tensor([
            [0.3, 0.4, 0.1, 0.1],
            [0.5, 0.5, 0.2, 0.2],
            [0.3, 0.4, 0.1, 0.1],
            [0.5, 0.5, 0.2, 0.2],
            [0.4, 0.45, 0.15, 0.15],
        ])
        prior = StructuredPrior.from_statistics(boxes, num_components=2)
        samples = prior.sample(100)
        assert samples.shape == (100, 4)

    def test_clamped_output(self, prior):
        """F1: 采样应支持 clamp 到有效范围 (0~1)"""
        samples = prior.sample(100, clamp_to_unit=True)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_diversity(self, prior):
        """F1: 采样应有多样性 (std > 0)"""
        samples = prior.sample(1000)
        std = samples.std(dim=0)
        assert (std > 0.01).all()
