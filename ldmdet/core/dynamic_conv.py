"""动态卷积模块

通过 ROI 特征与提案特征的交互，生成动态卷积参数并应用。
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ============================================================
# Triton kernel: fused LayerNorm + ReLU
# ============================================================
if _HAS_TRITON:

    @triton.jit
    def _layernorm_relu_kernel(
        X_ptr,  # [..., M] input
        W_ptr,  # [M] weight
        B_ptr,  # [M] bias
        Y_ptr,  # [..., M] output
        M,
        stride_row,
        stride_m,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """融合 LayerNorm + ReLU: y = relu(layer_norm(x, w, b))"""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_M)
        mask = offs < M

        x = tl.load(
            X_ptr + row * stride_row + offs * stride_m,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        mean = tl.sum(x, axis=0) / M
        # triton 2.1 不支持 ** 运算符, 用 * 替代
        diff = x - mean
        var = tl.sum(diff * diff, axis=0) / M
        x_norm = (x - mean) / tl.sqrt(var + eps)

        w = tl.load(W_ptr + offs, mask=mask, other=0.0)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0)
        y = x_norm * w + b
        y = tl.where(y > 0, y, 0.0)  # ReLU

        tl.store(
            Y_ptr + row * stride_row + offs * stride_m, y, mask=mask
        )

    def _triton_fused_layernorm_relu(
        x: Tensor, weight: Tensor, bias: Tensor, eps: float
    ) -> Tensor:
        """triton fused LayerNorm + ReLU 前向"""
        M = x.shape[-1]
        x_2d = x.reshape(-1, M)
        Y = torch.empty_like(x)
        y_2d = Y.reshape(-1, M)
        N_rows = x_2d.shape[0]

        stride_row, stride_m = x_2d.stride()
        BLOCK_M = max(16, min(4096, triton.next_power_of_2(M)))
        grid = (N_rows,)
        _layernorm_relu_kernel[grid](
            x_2d,
            weight,
            bias,
            y_2d,
            M,
            stride_row,
            stride_m,
            eps=eps,
            BLOCK_M=BLOCK_M,
        )
        return Y


class _FusedLayerNormReLU(torch.autograd.Function):
    """融合 LayerNorm + ReLU (前向用 triton, 反向用 PyTorch 原生 autograd)

    前向: y = relu(layer_norm(x, w, b, eps))
    反向: 通过 PyTorch 原生 F.layer_norm + F.relu 的 autograd 计算,
          保证梯度数值与原始串行实现完全一致。
    """

    @staticmethod
    def forward(ctx, x, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.save_for_backward(x, weight, bias)
        if x.is_cuda and _HAS_TRITON:
            return _triton_fused_layernorm_relu(x, weight, bias, eps)
        else:
            return torch.relu(
                torch.nn.functional.layer_norm(
                    x, normalized_shape, weight, bias, eps
                )
            )

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            w_req = weight.detach().requires_grad_(True)
            b_req = bias.detach().requires_grad_(True)
            out = torch.relu(
                torch.nn.functional.layer_norm(
                    x_req, normalized_shape, w_req, b_req, eps
                )
            )
            out.backward(grad_output)
        return x_req.grad, w_req.grad, b_req.grad, None, None


def fused_layernorm_relu(x: Tensor, norm_layer: nn.LayerNorm) -> Tensor:
    """融合 LayerNorm + ReLU

    Args:
        x: 输入 tensor
        norm_layer: nn.LayerNorm 模块 (使用其 weight, bias, eps, normalized_shape)
    """
    return _FusedLayerNormReLU.apply(
        x,
        norm_layer.weight,
        norm_layer.bias,
        norm_layer.normalized_shape,
        norm_layer.eps,
    )


class DynamicConv(nn.Module):
    """动态卷积: 提案特征 → 卷积参数 → 作用于 ROI 特征

    注: 测试表明对当前小尺寸 (N=100, C=256, dynamic_dim=64) 的 LayerNorm,
    PyTorch 原生 CUDA kernel + inplace ReLU 已高度优化, triton fused kernel
    因额外 reshape/alloc 开销反而变慢 (0.73x)。因此保留原始实现。
    fused_layernorm_relu 函数保留供未来大尺寸场景使用。
    """

    def __init__(
        self,
        feat_channels: int,
        dynamic_dim: int = 64,
        dynamic_num: int = 2,
        pooler_resolution: int = 7,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        self.num_params = self.feat_channels * self.dynamic_dim

        self.dynamic_layer = nn.Linear(
            self.feat_channels, self.dynamic_num * self.num_params
        )
        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)
        self.act = nn.ReLU(inplace=True)
        num_output = self.feat_channels * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.feat_channels)
        self.norm3 = nn.LayerNorm(self.feat_channels)

    def forward(self, proposals: Tensor, roi_feats: Tensor) -> Tensor:
        """Args:
            proposals: (1, N, C) 提案特征
            roi_feats: (P*P, N, C) ROI 特征
        Returns: (1, N, C)
        """
        features = roi_feats.transpose(0, 1)  # (N, P*P, C)
        parameters = self.dynamic_layer(proposals.squeeze(0))  # (N, num_params)
        param_list = parameters.chunk(self.dynamic_num, dim=1)

        param1 = param_list[0].view(-1, self.feat_channels, self.dynamic_dim)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.act(features)

        param2 = param_list[1].view(-1, self.dynamic_dim, self.feat_channels)
        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.act(features)

        features = features.reshape(features.size(0), -1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.act(features)
        return features.unsqueeze(0)
