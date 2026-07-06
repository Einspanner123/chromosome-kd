"""Phase 0 探针实验: CKA (Centered Kernel Alignment)

测量 ChromoGen 特征与 LDMDet FPN 特征的相似度。

CKA 公式 (Kornblith et al., 2019):
  线性 CKA: CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
  核 CKA:   CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

实验设计:
  1. 对同一批图像, 分别提取 ChromoGen 特征和 LDMDet FPN 特征
  2. 对每对层计算 CKA
  3. 找到最佳匹配层
  4. 用置换检验验证显著性

预期结果:
  - 如果 ChromoGen 特征与 LDMDet 特征有高 CKA, 说明生成模型学到了检测特征
  - 浅层 (down1) 可能与 FPN 浅层 (l2) 更相似
  - 深层 (mid) 可能与 FPN 深层 (l4) 更相似
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """线性 CKA

    Args:
        X: (N, D1) 特征矩阵
        Y: (N, D2) 特征矩阵

    Returns:
        cka: 标量, [0, 1]
    """
    assert X.shape[0] == Y.shape[0], '样本数必须相同'

    # 中心化
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # 计算 Gram 矩阵的 Frobenius 范数
    # ||Y^T X||_F^2 = trace(X X^T Y Y^T) ... 简化为 trace(X^T Y Y^T X)
    # 但更高效: ||X^T Y||_F^2
    cross = X.T @ Y  # (D1, D2)
    cross_norm_sq = (cross**2).sum()

    X_norm_sq = (X.T @ X).pow(2).sum()
    Y_norm_sq = (Y.T @ Y).pow(2).sum()

    # 避免 division by zero
    denom = torch.sqrt(X_norm_sq * Y_norm_sq) + 1e-12
    cka = cross_norm_sq / denom

    return cka.item()


def kernel_CKA(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: float = None,
) -> float:
    """核 CKA (使用 RBF 核)

    Args:
        X: (N, D1) 特征矩阵
        Y: (N, D2) 特征矩阵
        sigma: RBF 核带宽, None 则使用中位数启发式

    Returns:
        cka: 标量, [0, 1]
    """
    assert X.shape[0] == Y.shape[0], '样本数必须相同'

    N = X.shape[0]

    # 中心化
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # RBF 核
    if sigma is None:
        # 中位数启发式
        with torch.no_grad():
            dists = torch.cdist(X, X)
            sigma = dists.median().item()
            if sigma == 0:
                sigma = 1.0

    K = torch.exp(-(torch.cdist(X, X) ** 2) / (2 * sigma**2))
    L = torch.exp(-(torch.cdist(Y, Y) ** 2) / (2 * sigma**2))

    # 中心化核矩阵
    H = torch.eye(N, device=X.device) - 1.0 / N
    Kc = H @ K @ H
    Lc = H @ L @ H

    # HSIC
    hsic_xy = (Kc * Lc).sum()
    hsic_xx = (Kc * Kc).sum()
    hsic_yy = (Lc * Lc).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy) + 1e-12
    cka = hsic_xy / denom

    return cka.item()


class CKAAnalyzer:
    """CKA 分析器: 比较 ChromoGen 和 LDMDet 特征空间"""

    def __init__(self, device: str = 'cpu', n_permutations: int = 1000):
        self.device = torch.device(device)
        self.n_permutations = n_permutations

    def compute_similarity_matrix(
        self,
        feats_A: dict,
        feats_B: dict,
        method: str = 'linear',
    ) -> torch.Tensor:
        """计算两个特征集之间的 CKA 相似度矩阵

        Args:
            feats_A: dict {layer_name: (N, D)} 特征 A
            feats_B: dict {layer_name: (N, D)} 特征 B
            method: 'linear' or 'kernel'

        Returns:
            sim_matrix: (len(A), len(B)) CKA 矩阵
        """
        keys_A = list(feats_A.keys())
        keys_B = list(feats_B.keys())
        sim = torch.zeros(len(keys_A), len(keys_B))

        for i, kA in enumerate(keys_A):
            for j, kB in enumerate(keys_B):
                X = feats_A[kA]
                Y = feats_B[kB]
                # 展平特征 (如果是多维度)
                if X.dim() > 2:
                    X = X.view(X.shape[0], -1)
                if Y.dim() > 2:
                    Y = Y.view(Y.shape[0], -1)

                if method == 'linear':
                    sim[i, j] = linear_CKA(X, Y)
                else:
                    sim[i, j] = kernel_CKA(X, Y)

        return sim

    def find_best_matches(
        self,
        feats_A: dict,
        feats_B: dict,
        method: str = 'linear',
    ) -> dict:
        """为 feats_A 的每层找到 feats_B 中最相似的层

        Returns:
            matches: {layer_A: {'best_match': layer_B, 'cka': float, 'all_ckas': {layer_B: cka}}}
        """
        sim = self.compute_similarity_matrix(feats_A, feats_B, method)
        keys_A = list(feats_A.keys())
        keys_B = list(feats_B.keys())

        matches = {}
        for i, kA in enumerate(keys_A):
            best_j = sim[i].argmax().item()
            matches[kA] = {
                'best_match': keys_B[best_j],
                'cka': sim[i, best_j].item(),
                'all_ckas': {
                    keys_B[j]: sim[i, j].item() for j in range(len(keys_B))
                },
            }

        return matches

    def permutation_test(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        method: str = 'linear',
    ) -> tuple:
        """置换检验: 验证 CKA 的显著性

        通过打乱 Y 的样本顺序, 生成零分布, 计算 p-value

        Returns:
            cka_observed: 观察到的 CKA 值
            p_value: p-value
        """
        N = X.shape[0]

        # 观察到的 CKA
        if method == 'linear':
            cka_observed = linear_CKA(X, Y)
        else:
            cka_observed = kernel_CKA(X, Y)

        # 置换检验
        count_extreme = 0
        for _ in range(self.n_permutations):
            perm = torch.randperm(N)
            Y_perm = Y[perm]
            if method == 'linear':
                cka_perm = linear_CKA(X, Y_perm)
            else:
                cka_perm = kernel_CKA(X, Y_perm)

            if cka_perm >= cka_observed:
                count_extreme += 1

        p_value = (count_extreme + 1) / (self.n_permutations + 1)
        return cka_observed, p_value
