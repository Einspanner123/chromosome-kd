"""测试 Phase 0 探针实验: CKA (Centered Kernel Alignment)

测量 ChromoGen 特征与 LDMDet FPN 特征的相似度。
"""

import pytest
import torch

from projects.ChromoGen.evaluation.cka import (
    CKAAnalyzer,
    linear_CKA,
    kernel_CKA,
)


class TestCKAFunctions:
    """测试 CKA 计算函数"""

    def test_linear_cka_identical_features(self):
        """相同特征的 CKA 应该接近 1"""
        X = torch.randn(100, 64)
        cka = linear_CKA(X, X)
        assert cka > 0.99

    def test_linear_cka_uncorrelated_features(self):
        """不相关特征的 CKA 应该较低"""
        torch.manual_seed(42)
        N, D = 500, 64  # 大样本降低随机相关性
        X = torch.randn(N, D)
        Y = torch.randn(N, D)
        cka = linear_CKA(X, Y)
        # 不相关特征的 CKA 应远低于 1
        assert cka < 0.2, f'CKA should be low for uncorrelated, got {cka}'

    def test_linear_cka_range(self):
        """CKA 值应在 [0, 1]"""
        X = torch.randn(100, 64)
        Y = torch.randn(100, 128)
        cka = linear_CKA(X, Y)
        assert 0 <= cka <= 1

    def test_kernel_cka_identical_features(self):
        """相同特征的 kernel CKA 应该接近 1"""
        X = torch.randn(100, 64)
        cka = kernel_CKA(X, X)
        assert cka > 0.99

    def test_kernel_cka_range(self):
        """kernel CKA 值应在 [0, 1]"""
        X = torch.randn(100, 64)
        Y = torch.randn(100, 128)
        cka = kernel_CKA(X, Y)
        assert 0 <= cka <= 1

    def test_different_feature_dimensions(self):
        """不同维度的特征也能计算 CKA"""
        X = torch.randn(100, 320)
        Y = torch.randn(100, 1280)
        cka_linear = linear_CKA(X, Y)
        cka_kernel = kernel_CKA(X, Y)
        assert 0 <= cka_linear <= 1
        assert 0 <= cka_kernel <= 1


class TestCKAAnalyzer:
    """测试 CKA 分析器"""

    def test_init(self):
        """初始化分析器"""
        analyzer = CKAAnalyzer(device='cpu')
        assert str(analyzer.device) == 'cpu'

    def test_compute_similarity_matrix(self):
        """计算两个特征集的相似度矩阵"""
        analyzer = CKAAnalyzer(device='cpu')
        # 模拟 4 层 ChromoGen 特征和 4 层 LDMDet 特征
        chromogen_feats = {
            'down1': torch.randn(100, 320),
            'down2': torch.randn(100, 640),
            'mid': torch.randn(100, 1280),
        }
        ldmdet_feats = {
            'fpn_l2': torch.randn(100, 256),
            'fpn_l3': torch.randn(100, 256),
            'fpn_l4': torch.randn(100, 256),
        }

        sim_matrix = analyzer.compute_similarity_matrix(
            chromogen_feats, ldmdet_feats
        )

        assert sim_matrix.shape == (3, 3)
        # 对角线不一定最大，但所有值应在 [0, 1]
        assert (sim_matrix >= 0).all() and (sim_matrix <= 1).all()

    def test_find_best_matches(self):
        """找到最佳匹配层"""
        analyzer = CKAAnalyzer(device='cpu')
        # 构造已知匹配: down1 ≈ fpn_l2, down2 ≈ fpn_l3
        torch.manual_seed(42)
        base1 = torch.randn(100, 64)
        base2 = torch.randn(100, 64)

        chromogen_feats = {
            'down1': base1 + 0.1 * torch.randn(100, 64),
            'down2': base2 + 0.1 * torch.randn(100, 64),
        }
        ldmdet_feats = {
            'fpn_l2': base1 + 0.1 * torch.randn(100, 64),
            'fpn_l3': base2 + 0.1 * torch.randn(100, 64),
        }

        matches = analyzer.find_best_matches(chromogen_feats, ldmdet_feats)

        assert 'down1' in matches
        assert 'down2' in matches
        assert matches['down1']['best_match'] == 'fpn_l2'
        assert matches['down2']['best_match'] == 'fpn_l3'

    def test_permutation_test(self):
        """置换检验: 验证 CKA 显著性"""
        analyzer = CKAAnalyzer(device='cpu', n_permutations=100)
        X = torch.randn(100, 64)
        Y = X + 0.1 * torch.randn(100, 64)  # 相关特征

        cka, p_value = analyzer.permutation_test(X, Y)

        assert 0 <= cka <= 1
        assert 0 <= p_value <= 1
        # 相关特征应有低 p-value
        assert p_value < 0.05
