"""深度诊断: 分析 DiT 模型为何无法训练 (mAP=0)

诊断项目:
  1. 不同 bbox 输入是否产生不同的 box tokens (token 多样性)
  2. Deformable Cross-Attention 是否产生有意义的输出
  3. 不同时间步 t 对 token 的影响
  4. Classification logits 的分布
  5. Regression predictions 的范围和多样性
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from mods.dit_head import DiTDiffusionDetHead
from mods.structures import ImageMeta


def create_mock_fpn(batch_size=2, feat_channels=384):
    strides = [4, 8, 16, 32]
    feats = []
    for s in strides:
        h = 800 // s
        w = 1200 // s
        feats.append(torch.randn(batch_size, feat_channels, h, w))
    return feats


def create_one_meta(img_size=(800, 1200)):
    return ImageMeta(
        img_shape=(img_size[0], img_size[1], 3),
        pad_shape=(img_size[0], img_size[1], 3),
        scale_factor=[1.0, 1.0, 1.0, 1.0],
    )


def main():
    print("=" * 70)
    print("DiT Deep Diagnostic")
    print("=" * 70)

    bs, N, C = 2, 100, 384
    num_classes = 24

    # 创建带 roi_extractor 的模型（与当前配置一致）
    head = DiTDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=C,
        num_proposals=N,
        num_heads=3,
        num_blocks=1,
        prior_prob=0.5,
        deep_supervision=True,
        diffusion_type='rectified_flow',
        box_init_mode='spatial_prior',
        use_adaln_zero=False,
        share_heads=False,
        regression_mode='direct',
        roi_extractor={
            'roi_layer': {'output_size': (7, 7), 'sampling_ratio': 2},
            'featmap_strides': [4, 8, 16, 32],
        },
    )

    fpn = create_mock_fpn(batch_size=bs, feat_channels=C)
    img_metas = [create_one_meta() for _ in range(bs)]

    # ================================================================
    # Test 1: Token diversity — 不同 bbox 输入是否产生不同的 token
    # ================================================================
    print("\n--- Test 1: Token Diversity ---")

    # 输入 A: 小框
    bboxes_a = torch.tensor([[
        [0.1, 0.1, 0.2, 0.2],
    ] * N]).float().expand(bs, -1, -1)
    # 输入 B: 大框
    bboxes_b = torch.tensor([[
        [0.3, 0.3, 0.7, 0.7],
    ] * N]).float().expand(bs, -1, -1)

    t = torch.full((bs,), 500.0)

    with torch.no_grad():
        tokens_a, _ = head.box_tokenizer(
            bboxes_a, fpn, roi_content=None
        )
        tokens_b, _ = head.box_tokenizer(
            bboxes_b, fpn, roi_content=None
        )

    diff_between_bboxes = (tokens_a - tokens_b).abs().mean().item()
    print(f"  Token diff (small vs large bbox): {diff_between_bboxes:.4f}")
    if diff_between_bboxes < 0.01:
        print("  WARNING: Very small difference — tokens may be collapsing!")

    # ================================================================
    # Test 2: Deformable Cross-Attention output 分析
    # ================================================================
    print("\n--- Test 2: Deformable Cross-Attention Output ---")

    from mods.deformable_attn import flatten_fpn_features

    fpn_flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)

    time_dim = C * 4
    time_emb = torch.randn(bs, time_dim)

    with torch.no_grad():
        cls_out, pred_bboxes, updated_tokens, obj, vel = head.head_series[0](
            tokens_a, fpn_flattened, spatial_shapes,
            level_start_index, time_emb, bboxes_a,
        )

    print(f"  cls_logits range: [{cls_out.min():.4f}, {cls_out.max():.4f}]")
    print(f"  cls_logits mean: {cls_out.mean():.4f}, std: {cls_out.std():.4f}")

    sigmoid_cls = torch.sigmoid(cls_out)
    max_scores = sigmoid_cls.max(-1)[0]
    print(f"  sigmoid max_scores: mean={max_scores.mean():.4f}, "
          f"max={max_scores.max():.4f}")

    print(f"  pred_bboxes range: [{pred_bboxes.min():.4f}, {pred_bboxes.max():.4f}]")
    print(f"  pred_bboxes x1 range: [{pred_bboxes[..., 0].min():.4f}, "
          f"{pred_bboxes[..., 0].max():.4f}]")
    print(f"  pred_bboxes x2 range: [{pred_bboxes[..., 2].min():.4f}, "
          f"{pred_bboxes[..., 2].max():.4f}]")

    # Check if all predictions are similar (collapse)
    pred_std_per_sample = pred_bboxes.std(dim=1)  # (bs, N, 4)
    print(f"  pred bbox std per sample: mean={pred_std_per_sample.mean():.4f}, "
          f"max={pred_std_per_sample.max():.4f}")

    # ================================================================
    # Test 3: Check token changes through DiT blocks
    # ================================================================
    print("\n--- Test 3: Token Evolution through DiT Blocks ---")
    current = tokens_a.clone()
    for i, block in enumerate(head.head_series[0].dit_blocks):
        with torch.no_grad():
            next_tokens = block(
                current, fpn_flattened, spatial_shapes,
                level_start_index, time_emb, bboxes_a,
            )
        change = (next_tokens - current).abs().mean().item()
        current = next_tokens
        print(f"  Block {i}: token change = {change:.6f}, "
              f"token range = [{current.min():.4f}, {current.max():.4f}]")

    # ================================================================
    # Test 4: Token diversity AFTER DiT blocks
    # ================================================================
    print("\n--- Test 4: Token Diversity After DiT Block ---")
    from mods.deformable_attn import flatten_fpn_features

    # Create two different bbox inputs
    bboxes_1 = torch.rand(bs, N, 4)
    bboxes_1[..., [0, 2]] = bboxes_1[..., [0, 2]].sort(dim=-1)[0]
    bboxes_1[..., [1, 3]] = bboxes_1[..., [1, 3]].sort(dim=-1)[0]
    bboxes_2 = torch.rand(bs, N, 4)
    bboxes_2[..., [0, 2]] = bboxes_2[..., [0, 2]].sort(dim=-1)[0]
    bboxes_2[..., [1, 3]] = bboxes_2[..., [1, 3]].sort(dim=-1)[0]

    with torch.no_grad():
        tokens_1, _ = head.box_tokenizer(bboxes_1, fpn, roi_content=None)
        tokens_2, _ = head.box_tokenizer(bboxes_2, fpn, roi_content=None)

        # Forward through first head's DiT blocks
        fpn_flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)
        time_emb = torch.randn(bs, C * 4)

        updated_1 = tokens_1
        updated_2 = tokens_2
        for block in head.head_series[0].dit_blocks:
            updated_1 = block(updated_1, fpn_flattened, spatial_shapes,
                              level_start_index, time_emb, bboxes_1)
            updated_2 = block(updated_2, fpn_flattened, spatial_shapes,
                              level_start_index, time_emb, bboxes_2)

    pre_diff = (tokens_1 - tokens_2).abs().mean().item()
    post_diff = (updated_1 - updated_2).abs().mean().item()
    token_std_per_sample = updated_1.std(dim=1).mean().item()

    print(f"  Token diff (pre-block):  {pre_diff:.4f}")
    print(f"  Token diff (post-block): {post_diff:.4f}")
    print(f"  Token std within sample: {token_std_per_sample:.4f}")
    if post_diff < 0.01:
        print("  CRITICAL: Tokens collapse to identical values after DiT block!")
    if token_std_per_sample < 0.05:
        print("  WARNING: All tokens within a sample are nearly identical!")

    # Also check regression head output diversity
    with torch.no_grad():
        reg_out_1 = head.head_series[0].reg_head(updated_1)
        reg_out_2 = head.head_series[0].reg_head(updated_2)
    reg_diff = (reg_out_1 - reg_out_2).abs().mean().item()
    reg_std = reg_out_1.std(dim=1).mean().item()
    print(f"  Reg output diff (post-block): {reg_diff:.4f}")
    print(f"  Reg output std within sample: {reg_std:.4f}")
    if reg_std < 0.01:
        print("  CRITICAL: Regression head produces identical outputs for all tokens!")

    # ================================================================
    # Test 5: Content injection effectiveness
    # ================================================================
    print("\n--- Test 5: RoI Content Injection Effect ---")
    roi_content = torch.randn(bs * N, C, 7, 7)

    with torch.no_grad():
        tokens_no_roi, _ = head.box_tokenizer(bboxes_a, fpn, roi_content=None)
        tokens_with_roi, _ = head.box_tokenizer(bboxes_a, fpn, roi_content=roi_content)

    roi_effect = (tokens_with_roi - tokens_no_roi).abs().mean().item()
    print(f"  RoI injection effect: {roi_effect:.4f}")

    # ================================================================
    # Test 6: Full forward with different t values
    # ================================================================
    print("\n--- Test 6: Effect of t on predictions ---")
    results = {}
    for t_val in [0.0, 250.0, 500.0, 750.0, 1000.0]:
        t_input = torch.full((bs,), t_val)
        with torch.no_grad():
            cls_logits, pred_bboxes, _, _, _ = head(
                tuple(fpn), bboxes_a * 800, t_input, img_metas=img_metas
            )
        sig = torch.sigmoid(cls_logits[-1])
        results[t_val] = {
            'cls_mean': cls_logits[-1].mean().item(),
            'sig_max': sig.max().item(),
            'bbox_mean': pred_bboxes[-1].mean().item(),
            'bbox_std': pred_bboxes[-1].std().item(),
        }
        print(f"  t={t_val:6.1f}: cls_mean={results[t_val]['cls_mean']:.4f}, "
              f"sig_max={results[t_val]['sig_max']:.4f}, "
              f"bbox_mean={results[t_val]['bbox_mean']:.4f}, "
              f"bbox_std={results[t_val]['bbox_std']:.4f}")

    # Check if t significantly affects outputs
    cls_means = [v['cls_mean'] for v in results.values()]
    cls_range = max(cls_means) - min(cls_means)
    print(f"  cls_mean range across t: {cls_range:.4f} "
          f"(should be > 0 to show t affects predictions)")

    bbox_stds = [v['bbox_std'] for v in results.values()]
    bbox_std_range = max(bbox_stds) - min(bbox_stds)
    print(f"  bbox_std range across t: {bbox_std_range:.4f}")

    print("\n" + "=" * 70)
    print("Diagnostic complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
