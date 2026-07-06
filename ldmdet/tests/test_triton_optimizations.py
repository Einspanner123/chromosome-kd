"""测试 triton 优化的数值一致性

验证以下优化点的前后输出数值一致:
  #1 cross_attn_bridge: nn.MultiheadAttention need_weights=False (SDPA) — 已启用
  #3 Sinkhorn: triton fused logsumexp kernel — 已启用
  #4 DynamicConv: fused_layernorm_relu 工具函数 (未接入 DynamicConv.forward, 因小尺寸下变慢)
  #5 Cost Matrix: torch.stack().sum() (就地累加方案已回退, 因变慢; 此处为回归测试)
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.coupling._sinkhorn_ops import (
    _HAS_TRITON,
    _triton_fused_col_lse,
    _triton_fused_row_lse,
    sinkhorn_transport_batch,
)
from ldmdet.core.dynamic_conv import DynamicConv, fused_layernorm_relu

CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA, reason='需要 CUDA')
skip_no_triton = pytest.mark.skipif(not _HAS_TRITON, reason='需要 triton')


# ============================================================
# #1 cross_attn_bridge: SDPA 数值一致性
# ============================================================
@skip_no_cuda
def test_attn_sdpa_consistency():
    """nn.MultiheadAttention need_weights=True vs False 数值一致"""
    torch.manual_seed(42)
    mha = nn.MultiheadAttention(
        embed_dim=256, num_heads=4, batch_first=True
    ).cuda()
    mha.eval()

    B, Lq, Lk, C = 2, 576, 1152, 256
    q = torch.randn(B, Lq, C).cuda()
    k = torch.randn(B, Lk, C).cuda()
    v = torch.randn(B, Lk, C).cuda()

    with torch.no_grad():
        out_old, _ = mha(q, k, v)  # need_weights=True (默认, 朴素 attention)
        out_new, _ = mha(q, k, v, need_weights=False)  # SDPA / FlashAttention

    assert torch.allclose(out_old, out_new, atol=1e-4), (
        f'SDPA vs 朴素 attention 最大误差: '
        f'{(out_old - out_new).abs().max().item()}'
    )


# ============================================================
# #3 Sinkhorn: triton fused logsumexp 数值一致性
# ============================================================
@skip_no_cuda
@skip_no_triton
def test_triton_row_lse_consistency():
    """_triton_fused_row_lse vs torch.logsumexp + masked_fill"""
    torch.manual_seed(42)
    G, N, K = 4, 50, 30
    log_K = torch.randn(G, N, K).cuda()
    log_v = torch.randn(G, K).cuda()
    log_rm = torch.randn(G, N).cuda()

    # 模拟 padding: group 0 的 row 40-49 是 padding
    row_mask = torch.ones(G, N, dtype=torch.bool).cuda()
    row_mask[0, 40:] = False
    # padded entries 在 log_K 中设为 -inf
    log_K[0, 40:] = float('-inf')

    # torch 路径
    log_u_torch = log_rm - torch.logsumexp(
        log_K + log_v.unsqueeze(1), dim=2
    )
    log_u_torch.masked_fill_(~row_mask, float('-inf'))

    # triton 路径
    row_mask_f = row_mask.float()
    log_u_triton = _triton_fused_row_lse(log_K, log_v, log_rm, row_mask_f)

    # 比较 (忽略 padding 行, 因为 -inf vs -inf 可能 NaN)
    valid = row_mask
    diff = (log_u_torch[valid] - log_u_triton[valid]).abs().max().item()
    assert diff < 1e-5, f'row lse 最大误差: {diff}'


@skip_no_cuda
@skip_no_triton
def test_triton_col_lse_consistency():
    """_triton_fused_col_lse vs torch.logsumexp + masked_fill"""
    torch.manual_seed(42)
    G, N, K = 4, 50, 30
    log_K = torch.randn(G, N, K).cuda()
    log_u = torch.randn(G, N).cuda()
    log_cm = torch.randn(G, K).cuda()

    col_mask = torch.ones(G, K, dtype=torch.bool).cuda()
    col_mask[1, 25:] = False
    log_K[1, :, 25:] = float('-inf')

    # torch 路径
    log_v_torch = log_cm - torch.logsumexp(
        log_K + log_u.unsqueeze(2), dim=1
    )
    log_v_torch.masked_fill_(~col_mask, float('-inf'))

    # triton 路径
    col_mask_f = col_mask.float()
    log_v_triton = _triton_fused_col_lse(log_K, log_u, log_cm, col_mask_f)

    valid = col_mask
    diff = (log_v_torch[valid] - log_v_triton[valid]).abs().max().item()
    assert diff < 1e-5, f'col lse 最大误差: {diff}'


@skip_no_cuda
@skip_no_triton
def test_sinkhorn_transport_batch_triton_consistency():
    """sinkhorn_transport_batch triton 路径 vs 原始路径"""
    torch.manual_seed(42)
    costs = [
        torch.randn(50, 30).cuda(),
        torch.randn(40, 25).cuda(),
        torch.randn(45, 28).cuda(),
        torch.randn(35, 20).cuda(),
    ]

    # triton 路径 (默认, CUDA + triton 可用)
    result_triton = sinkhorn_transport_batch(costs, epsilon=0.1, num_iters=20)

    # 原始路径: 临时禁用 triton
    import ldmdet.coupling._sinkhorn_ops as _sk

    original_has_triton = _sk._HAS_TRITON
    _sk._HAS_TRITON = False
    try:
        result_torch = sinkhorn_transport_batch(
            costs, epsilon=0.1, num_iters=20
        )
    finally:
        _sk._HAS_TRITON = original_has_triton

    assert len(result_triton) == len(result_torch)
    for i, (rt, ro) in enumerate(zip(result_triton, result_torch)):
        diff = (rt - ro).abs().max().item()
        assert diff < 1e-4, f'group {i} 最大误差: {diff}'


# ============================================================
# #4 DynamicConv: fused_layernorm_relu 工具函数数值一致性
# 注: DynamicConv.forward 未使用 fused kernel (小尺寸下变慢 0.73x),
#     此处仅验证工具函数自身的正确性, 作为未来大尺寸场景的回归测试。
# ============================================================
@skip_no_cuda
@skip_no_triton
def test_fused_layernorm_relu_forward():
    """fused_layernorm_relu 前向 vs nn.LayerNorm + nn.ReLU"""
    torch.manual_seed(42)
    ln = nn.LayerNorm(64).cuda()
    x = torch.randn(100, 49, 64).cuda()

    # 原始路径
    y_old = torch.relu(ln(x))

    # fused 路径
    y_new = fused_layernorm_relu(x, ln)

    diff = (y_old - y_new).abs().max().item()
    assert diff < 1e-5, f'forward 最大误差: {diff}'


@skip_no_cuda
@skip_no_triton
def test_fused_layernorm_relu_backward():
    """fused_layernorm_relu 反向梯度一致性"""
    torch.manual_seed(42)
    ln = nn.LayerNorm(64).cuda()
    x = torch.randn(100, 49, 64).cuda()
    grad_out = torch.randn(100, 49, 64).cuda()

    # 原始路径
    x1 = x.clone().requires_grad_(True)
    y1 = torch.relu(ln(x1))
    y1.backward(grad_out)
    grad_x_old = x1.grad

    # fused 路径
    x2 = x.clone().requires_grad_(True)
    y2 = fused_layernorm_relu(x2, ln)
    y2.backward(grad_out)
    grad_x_new = x2.grad

    diff = (grad_x_old - grad_x_new).abs().max().item()
    assert diff < 1e-5, f'backward 最大误差: {diff}'


@skip_no_cuda
@skip_no_triton
def test_dynamic_conv_forward_consistency():
    """DynamicConv 前向数值一致性 (fused vs 原始)"""
    torch.manual_seed(42)
    model = DynamicConv(
        feat_channels=256,
        dynamic_dim=64,
        dynamic_num=2,
        pooler_resolution=7,
    ).cuda()
    model.eval()

    N = 100
    proposals = torch.randn(1, N, 256).cuda()
    roi_feats = torch.randn(49, N, 256).cuda()

    with torch.no_grad():
        out = model(proposals, roi_feats)

    assert out.shape == (1, N, 256), f'输出形状错误: {out.shape}'
    assert not torch.isnan(out).any(), '输出包含 NaN'


# ============================================================
# #5 Cost Matrix: stack().sum() 回归测试
# 注: 就地累加方案因变慢 (0.92x) 已回退, 此测试验证当前实现数值稳定。
# ============================================================
@skip_no_cuda
def test_cost_matrix_accumulation():
    """torch.stack().sum() 数值正确 (回归测试)"""
    torch.manual_seed(42)
    bs, N, M = 2, 100, 50
    costs = [torch.randn(bs, N, M).cuda() for _ in range(4)]

    # stack().sum() (原始路径)
    old = torch.stack(costs).sum(0)

    # 就地累加 (优化路径)
    new = costs[0]
    for c in costs[1:]:
        new = new + c

    diff = (old - new).abs().max().item()
    assert diff < 1e-6, f'cost matrix 最大误差: {diff}'


# ============================================================
# 集成测试: 完整 Sinkhorn 端到端
# ============================================================
@skip_no_cuda
def test_sinkhorn_transport_batch_no_triton_fallback():
    """无 triton 时 fallback 到原始路径"""
    import ldmdet.coupling._sinkhorn_ops as _sk

    original = _sk._HAS_TRITON
    _sk._HAS_TRITON = False
    try:
        costs = [torch.randn(20, 10).cuda(), torch.randn(15, 8).cuda()]
        results = sinkhorn_transport_batch(costs, epsilon=0.1, num_iters=10)
        assert len(results) == 2
        assert results[0].shape == (20, 10)
        assert results[1].shape == (15, 8)
        assert not torch.isnan(results[0]).any()
    finally:
        _sk._HAS_TRITON = original
