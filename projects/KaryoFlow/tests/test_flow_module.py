"""FlowModule 单元测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from karyoflow.encoder import ChromosomeEncoder
from karyoflow.flow_module import KaryoFlowModule
from karyoflow.loss import KaryoFlowLoss


def test_forward_shape():
    """验证 forward 输出形状"""
    B, N, d = 2, 46, 128
    model = KaryoFlowModule(d_model=d, num_layers=2, nhead=4, dim_feedforward=256, num_slots=N)

    chrom_features = torch.randn(B, N, d)
    perm_noisy = torch.randint(0, N, (B, N))
    perm_noisy[:, :10] = model.mask_token_id  # mask 前 10 个位置
    t = torch.rand(B)

    logits = model(chrom_features, perm_noisy, t)
    assert logits.shape == (B, N, N), f"Expected ({B},{N},{N}), got {logits.shape}"

    print("  test_forward_shape PASSED")


def test_gradient_flow():
    """验证梯度可以正常回传"""
    B, N, d = 2, 46, 128
    model = KaryoFlowModule(d_model=d, num_layers=2, nhead=4, dim_feedforward=256, num_slots=N)
    criterion = KaryoFlowLoss()

    chrom_features = torch.randn(B, N, d, requires_grad=True)
    perm_gt = torch.arange(N).unsqueeze(0).expand(B, -1)  # 恒等排列
    perm_noisy = torch.full((B, N), model.mask_token_id, dtype=torch.long)
    t = torch.ones(B)
    mask = torch.ones(B, N, dtype=torch.bool)

    logits = model(chrom_features, perm_noisy, t)
    loss = criterion(logits, perm_gt, mask)
    loss.backward()

    assert chrom_features.grad is not None, "No gradient on input features"
    # AdaLN-Zero gate 初始为 0 → cross-attn 不传梯度到 memory (chrom_features)
    # 但 output_head 仍有梯度流经 lehmer_embed → model 参数
    # 所以检查模型参数有梯度即可
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "No gradient on any model parameter"

    # 检查 AdaLN-Zero gate 初始化为 0 → 初始 gradient 应来自 output_head
    for layer in model.layers:
        gate_weight = layer.adaln_mlp[-1].weight
        assert gate_weight.grad is not None, "No gradient on AdaLN gate"

    print("  test_gradient_flow PASSED")


def test_encoder_forward():
    """验证 Encoder forward"""
    N, d = 10, 128
    encoder = ChromosomeEncoder(d_model=d, crop_size=64, pretrained=False)

    crops = torch.randn(N, 3, 64, 64)
    bboxes = torch.tensor([[10, 20, 50, 60]] * N, dtype=torch.float32)

    features = encoder(crops, bboxes, image_size=(800, 800))
    assert features.shape == (N, d), f"Expected ({N},{d}), got {features.shape}"

    print("  test_encoder_forward PASSED")


def test_loss_masked_only():
    """验证 loss 只在 masked 位置计算"""
    B, N, V = 2, 10, 10
    criterion = KaryoFlowLoss()

    logits = torch.randn(B, N, V)
    target = torch.zeros(B, N, dtype=torch.long)

    # 全部 unmask → loss 应为 0 (没有 masked 位置)
    mask_none = torch.zeros(B, N, dtype=torch.bool)
    loss_none = criterion(logits, target, mask_none)
    assert loss_none.item() == 0.0, f"Expected 0, got {loss_none.item()}"

    # 全部 mask → loss 应为正数
    mask_all = torch.ones(B, N, dtype=torch.bool)
    loss_all = criterion(logits, target, mask_all)
    assert loss_all.item() > 0, f"Expected positive loss, got {loss_all.item()}"

    print("  test_loss_masked_only PASSED")


def test_overfit_single_sample():
    """过拟合单样本: loss 应趋近于 0"""
    import random
    random.seed(42)
    torch.manual_seed(42)

    B, N, d = 1, 10, 64
    model = KaryoFlowModule(d_model=d, num_layers=2, nhead=4, dim_feedforward=128, num_slots=N)
    criterion = KaryoFlowLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 固定输入
    chrom_features = torch.randn(B, N, d)
    perm_gt = torch.tensor([[5, 3, 2, 4, 1, 0, 9, 8, 7, 6]])  # 一个排列

    for step in range(200):
        t = torch.rand(B)
        mask = torch.rand(B, N) < t.unsqueeze(1)
        if not mask.any():
            mask[0, 0] = True

        perm_noisy = perm_gt.clone()
        perm_noisy[mask] = model.mask_token_id

        logits = model(chrom_features, perm_noisy, t)
        loss = criterion(logits, perm_gt, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 最终 loss 应该很小
    assert loss.item() < 0.5, f"Overfit failed: loss={loss.item():.4f} (expected < 0.5)"
    print(f"  test_overfit_single_sample PASSED (final loss={loss.item():.4f})")


if __name__ == "__main__":
    print("Running FlowModule tests...")
    test_forward_shape()
    test_gradient_flow()
    test_encoder_forward()
    test_loss_masked_only()
    test_overfit_single_sample()
    print("All FlowModule tests PASSED!")
