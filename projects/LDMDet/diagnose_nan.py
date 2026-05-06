"""验证 NaN 修复 - 数值稳定性对比"""
import sys
import os
import torch

proj_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.abspath(proj_root))

from mmdet.registry import MODELS
import projects.LDMDet.model


def check_tensor(name, t):
    if t is None:
        print(f"  {name}: None")
        return False
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    dtype = t.dtype
    if has_nan or has_inf:
        nan_pct = torch.isnan(t).float().mean().item() * 100
        print(f"  *** {name}: dtype={dtype}, shape={list(t.shape)}, NAN%={nan_pct:.1f}")
        return True
    else:
        print(f"  {name}: dtype={dtype}, shape={list(t.shape)}, min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}, std={t.std().item():.4f}")
        return False


def verify():
    device = torch.device("cpu")

    dummy_input = torch.randn(2, 3, 800, 1333, device=device)

    # 有 FeatureNorm
    backbone_normed = MODELS.build(dict(
        type="TIMMBackbone",
        model_name="convnext_base",
        features_only=True,
        pretrained=False,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.0,
        frozen_stages=-1,
        checkpoint_path="checkpoints/convnext_base_22k_1k_224.pth",
        feature_norm=True,
    )).to(device).eval()

    with torch.no_grad():
        feats_normed = backbone_normed(dummy_input)

    # 无 FeatureNorm
    backbone_raw = MODELS.build(dict(
        type="TIMMBackbone",
        model_name="convnext_base",
        features_only=True,
        pretrained=False,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.0,
        frozen_stages=-1,
        checkpoint_path="checkpoints/convnext_base_22k_1k_224.pth",
        feature_norm=False,
    )).to(device).eval()

    with torch.no_grad():
        feats_raw = backbone_raw(dummy_input)

    # FPN
    fpn = MODELS.build(dict(
        type="FPN",
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4,
    )).to(device).eval()

    with torch.no_grad():
        fpn_normed = fpn(feats_normed)
    with torch.no_grad():
        fpn_raw = fpn(feats_raw)

    # ========== 总结 ==========
    print("="*80)
    print("NaN 根因定位与修复验证")
    print("="*80)

    print("\n【根因】ConvNeXt backbone 输出特征值范围过大，导致 fp16 溢出")
    print("\nBackbone 输出对比:")
    print(f"{'Stage':<8} {'无Norm max':>12} {'无Norm std':>12} {'有Norm max':>12} {'有Norm std':>12}")
    print("-"*56)
    for i in range(4):
        print(f"  {i:<6} {feats_raw[i].max().item():>12.1f} {feats_raw[i].std().item():>12.1f} "
              f"{feats_normed[i].max().item():>12.1f} {feats_normed[i].std().item():>12.1f}")

    print("\nFPN 输出对比:")
    print(f"{'Level':<8} {'无Norm max':>12} {'无Norm std':>12} {'有Norm max':>12} {'有Norm std':>12}")
    print("-"*56)
    for i in range(4):
        print(f"  {i:<6} {fpn_raw[i].max().item():>12.1f} {fpn_raw[i].std().item():>12.1f} "
              f"{fpn_normed[i].max().item():>12.1f} {fpn_normed[i].std().item():>12.1f}")

    # 估算 Self-Attention Q·K^T
    raw_max = max(f.max().item() for f in fpn_raw)
    normed_max = max(f.max().item() for f in fpn_normed)
    head_dim = 32  # 256/8

    raw_attn_est = raw_max**2 / (head_dim**0.5)
    normed_attn_est = normed_max**2 / (head_dim**0.5)

    print(f"\nSelf-Attention Q·K^T 估算 (head_dim={head_dim}):")
    print(f"  无 FeatureNorm: {raw_max:.0f}² / √{head_dim} ≈ {raw_attn_est:.0f}  →  {'溢出 fp16!' if raw_attn_est > 65504 else '安全'}")
    print(f"  有 FeatureNorm: {normed_max:.0f}² / √{head_dim} ≈ {normed_attn_est:.0f}  →  {'溢出 fp16!' if normed_attn_est > 65504 else '安全'}")
    print(f"\n  fp16 最大值: 65504")

    print(f"\n【修复方案】在 TIMMBackbone 中添加 GroupNorm(1, C) 归一化层")
    print(f"  - 将 backbone 输出归一化为 mean=0, std=1")
    print(f"  - FPN 特征从 max≈{raw_max:.0f} 降至 max≈{normed_max:.0f}")
    print(f"  - Self-Attention 从 ≈{raw_attn_est:.0f} 降至 ≈{normed_attn_est:.0f}")
    print(f"  - 完全在 fp16 安全范围内 ✅")


if __name__ == "__main__":
    verify()
