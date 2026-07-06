"""Benchmark: triton 优化前后性能对比

对比以下优化点的前后速度:
  #1 cross_attn_bridge: SDPA (need_weights=False) — 已启用, 1.46x 加速
  #3 Sinkhorn: triton fused logsumexp — 已启用, 1.58x 加速
  #4 DynamicConv: fused LayerNorm + ReLU — 未启用 (小尺寸下 0.73x 变慢), 仅监控工具函数性能
  #5 Cost Matrix: stack().sum() vs 就地累加 — 未启用 (0.92x 变慢), 仅作性能监控
"""

import torch
import torch.nn as nn

import ldmdet.coupling._sinkhorn_ops as _sk
from ldmdet.core.dynamic_conv import DynamicConv, fused_layernorm_relu
from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport_batch


def cuda_time(fn, warmup=5, iters=50):
    """测量 CUDA 函数执行时间 (ms)"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ============================================================
# #1 cross_attn_bridge: SDPA
# ============================================================
def benchmark_attn_sdpa():
    """nn.MultiheadAttention need_weights=True vs False"""
    print('\n=== #1 cross_attn_bridge: SDPA ===')
    torch.manual_seed(42)
    mha = nn.MultiheadAttention(
        embed_dim=256, num_heads=4, batch_first=True
    ).cuda()
    mha.eval()

    B, Lq, Lk, C = 2, 576, 1152, 256
    q = torch.randn(B, Lq, C).cuda()
    k = torch.randn(B, Lk, C).cuda()
    v = torch.randn(B, Lk, C).cuda()

    def old_fn():
        with torch.no_grad():
            mha(q, k, v)  # need_weights=True (朴素 attention)

    def new_fn():
        with torch.no_grad():
            mha(q, k, v, need_weights=False)  # SDPA

    t_old = cuda_time(old_fn)
    t_new = cuda_time(new_fn)
    print(f'  朴素 attention (need_weights=True):  {t_old:.3f} ms')
    print(f'  SDPA (need_weights=False):           {t_new:.3f} ms')
    print(f'  加速比: {t_old / t_new:.2f}x')

    # 显存对比
    torch.cuda.reset_peak_memory_stats()
    old_fn()
    mem_old = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    new_fn()
    mem_new = torch.cuda.max_memory_allocated() / 1024**2
    print(f'  显存: {mem_old:.1f} MB → {mem_new:.1f} MB')


# ============================================================
# #3 Sinkhorn: triton fused logsumexp
# ============================================================
def benchmark_sinkhorn():
    """sinkhorn_transport_batch triton vs torch.logsumexp"""
    print('\n=== #3 Sinkhorn: triton fused logsumexp ===')
    torch.manual_seed(42)

    # 典型场景: 8 个 GT group, N~50 proposals, K~30
    costs = [torch.randn(50, 30).cuda() for _ in range(8)]

    # triton 路径 (默认)
    def triton_fn():
        sinkhorn_transport_batch(costs, epsilon=0.1, num_iters=20)

    # 原始路径: 临时禁用 triton
    original = _sk._HAS_TRITON
    _sk._HAS_TRITON = False

    def torch_fn():
        sinkhorn_transport_batch(costs, epsilon=0.1, num_iters=20)

    t_torch = cuda_time(torch_fn)
    _sk._HAS_TRITON = original
    t_triton = cuda_time(triton_fn)

    print(f'  torch.logsumexp (原始):  {t_torch:.3f} ms')
    print(f'  triton fused:            {t_triton:.3f} ms')
    print(f'  加速比: {t_torch / t_triton:.2f}x')

    # 显存对比
    torch.cuda.reset_peak_memory_stats()
    torch_fn()
    mem_old = torch.cuda.max_memory_allocated() / 1024**2
    _sk._HAS_TRITON = False
    torch.cuda.reset_peak_memory_stats()
    _sk._HAS_TRITON = original
    torch.cuda.reset_peak_memory_stats()
    triton_fn()
    mem_new = torch.cuda.max_memory_allocated() / 1024**2
    print(f'  显存: {mem_old:.1f} MB → {mem_new:.1f} MB')


# ============================================================
# #4 DynamicConv: fused LayerNorm + ReLU (未启用, 仅监控工具函数性能)
# 注: DynamicConv.forward 未使用 fused kernel, 此处对比 fused 工具函数
#     vs 原始 norm+act, 用于确认小尺寸下确实无收益。
# ============================================================
def benchmark_dynamic_conv():
    """DynamicConv 原始实现 vs fused (监控用, 当前 forward 用原始实现)"""
    print('\n=== #4 DynamicConv: fused LayerNorm+ReLU ===')
    torch.manual_seed(42)

    # 原始 DynamicConv (不使用 fused kernel)
    class OriginalDynamicConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_channels = 256
            self.dynamic_dim = 64
            self.dynamic_num = 2
            self.num_params = self.feat_channels * self.dynamic_dim
            self.dynamic_layer = nn.Linear(
                self.feat_channels, self.dynamic_num * self.num_params
            )
            self.norm1 = nn.LayerNorm(self.dynamic_dim)
            self.norm2 = nn.LayerNorm(self.feat_channels)
            self.act = nn.ReLU(inplace=True)
            num_output = self.feat_channels * 49
            self.out_layer = nn.Linear(num_output, self.feat_channels)
            self.norm3 = nn.LayerNorm(self.feat_channels)

        def forward(self, proposals, roi_feats):
            features = roi_feats.transpose(0, 1)
            parameters = self.dynamic_layer(proposals.squeeze(0))
            param_list = parameters.chunk(self.dynamic_num, dim=1)
            param1 = param_list[0].view(
                -1, self.feat_channels, self.dynamic_dim
            )
            features = torch.bmm(features, param1)
            features = self.norm1(features)
            features = self.act(features)
            param2 = param_list[1].view(
                -1, self.dynamic_dim, self.feat_channels
            )
            features = torch.bmm(features, param2)
            features = self.norm2(features)
            features = self.act(features)
            features = features.reshape(features.size(0), -1)
            features = self.out_layer(features)
            features = self.norm3(features)
            features = self.act(features)
            return features.unsqueeze(0)

    old_model = OriginalDynamicConv().cuda().eval()
    new_model = DynamicConv(256, 64, 2, 7).cuda().eval()
    # 同步权重
    new_model.load_state_dict(old_model.state_dict())

    N = 100
    proposals = torch.randn(1, N, 256).cuda()
    roi_feats = torch.randn(49, N, 256).cuda()

    def old_fn():
        with torch.no_grad():
            old_model(proposals, roi_feats)

    def new_fn():
        with torch.no_grad():
            new_model(proposals, roi_feats)

    t_old = cuda_time(old_fn)
    t_new = cuda_time(new_fn)
    print(f'  原始 (norm+act 串行):  {t_old:.3f} ms')
    print(f'  fused (triton kernel): {t_new:.3f} ms')
    print(f'  加速比: {t_old / t_new:.2f}x')

    # 单独 benchmark fused_layernorm_relu vs norm+relu
    ln = nn.LayerNorm(64).cuda()
    x = torch.randn(100, 49, 64).cuda()

    def old_ln_fn():
        with torch.no_grad():
            torch.relu(ln(x))

    def new_ln_fn():
        with torch.no_grad():
            fused_layernorm_relu(x, ln)

    t_old_ln = cuda_time(old_ln_fn)
    t_new_ln = cuda_time(new_ln_fn)
    print(f'  [单次 LN+ReLU] 原始: {t_old_ln:.3f} ms → fused: {t_new_ln:.3f} ms '
          f'({t_old_ln / t_new_ln:.2f}x)')


# ============================================================
# #5 Cost Matrix: stack().sum() vs 就地累加 (未启用, 仅性能监控)
# 注: 当前 matcher.py 使用 torch.stack().sum(), 就地累加方案因变慢已回退。
# ============================================================
def benchmark_cost_matrix():
    """stack().sum() vs 就地累加 (监控用, 当前用 stack().sum())"""
    print('\n=== #5 Cost Matrix: 就地累加 ===')
    torch.manual_seed(42)

    # 典型场景: bs=4, N=300 proposals, M=100 GT, 4 个 cost 项
    bs, N, M = 4, 300, 100
    costs = [torch.randn(bs, N, M).cuda() for _ in range(4)]

    def old_fn():
        return torch.stack(costs).sum(0)

    def new_fn():
        result = costs[0]
        for c in costs[1:]:
            result = result + c
        return result

    t_old = cuda_time(old_fn)
    t_new = cuda_time(new_fn)
    print(f'  stack().sum():  {t_old:.3f} ms')
    print(f'  就地累加:       {t_new:.3f} ms')
    print(f'  加速比: {t_old / t_new:.2f}x')

    # 显存对比
    torch.cuda.reset_peak_memory_stats()
    old_fn()
    mem_old = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    new_fn()
    mem_new = torch.cuda.max_memory_allocated() / 1024**2
    print(f'  显存: {mem_old:.1f} MB → {mem_new:.1f} MB')


if __name__ == '__main__':
    print('=' * 60)
    print('Triton 优化 Benchmark')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    try:
        import triton

        print(f'Triton: {triton.__version__}')
    except ImportError:
        print('Triton: not available')
    print('=' * 60)

    benchmark_attn_sdpa()
    benchmark_sinkhorn()
    benchmark_dynamic_conv()
    benchmark_cost_matrix()
    print('\n' + '=' * 60)
    print('Benchmark 完成')
