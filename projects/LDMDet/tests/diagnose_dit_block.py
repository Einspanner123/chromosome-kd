"""深度诊断 2: 追踪 DiTBlock 内部各组件的输出规模"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from mods.box_tokenizer import (
    bbox_to_reference_points,
    reference_points_with_levels,
)
from mods.deformable_attn import flatten_fpn_features
from mods.dit_block import DiTBlock


def create_mock_data(bs=2, N=100, feat_channels=384):
    strides = [4, 8, 16, 32]
    feats = []
    for s in strides:
        h, w = 800 // s, 1200 // s
        feats.append(torch.randn(bs, feat_channels, h, w))
    return feats


def main():
    print("=" * 70)
    print("DiTBlock Component-Level Diagnostic")
    print("=" * 70)

    bs, N, C = 2, 100, 384
    fpn = create_mock_data(bs, N, C)
    fpn_flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)

    block = DiTBlock(
        feat_channels=C,
        num_heads=3,
        num_fpn_levels=4,
        num_ref_points=8,
        dim_feedforward=2048,
        dropout=0.0,
        adaln_params=9,
        use_adaln_zero=False,
    )

    box_tokens = torch.randn(bs, N, C)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[..., [0,2]] = bbox_coords[..., [0,2]].sort(dim=-1)[0]
    bbox_coords[..., [1,3]] = bbox_coords[..., [1,3]].sort(dim=-1)[0]
    time_emb = torch.randn(bs, C * 4)

    # ================================================================
    # Trace adaln_mlp output magnitude
    # ================================================================
    print("\n--- AdaLN MLP output analysis ---")
    params = block.adaln_mlp(time_emb)
    g1, b1, a1, g2, b2, a2, g3, b3, a3 = params.chunk(9, dim=-1)

    for name, p in [('g1', g1), ('b1', b1), ('a1', a1),
                     ('g2', g2), ('b2', b2), ('a2', a2),
                     ('g3', g3), ('b3', b3), ('a3', a3)]:
        print(f"  {name}: mean={p.mean():.6f}, std={p.std():.6f}, "
              f"range=[{p.min():.4f}, {p.max():.4f}]")

    # Check a1_a2_a3 magnitude (residual scaling)
    for name, p in [('a1', a1), ('a2', a2), ('a3', a3)]:
        a_mean = p.abs().mean(dim=-1).mean().item()
        print(f"  {name} abs mean: {a_mean:.6f}")

    # ================================================================
    # Step 1: Self-Attention pathway
    # ================================================================
    print("\n--- Step 1: Self-Attention ---")
    x = block.norm1(box_tokens)
    x_mod = block._modulate(x, g1, b1)
    print(f"  norm1 output: mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"  modulated:    mean={x_mod.mean():.6f}, std={x_mod.std():.6f}")
    print(f"  modulation effect: {(x_mod - x).abs().mean():.6f}")

    # qkv projection
    qkv_out = block.qkv(x_mod)
    print(f"  qkv out: mean={qkv_out.mean():.6f}, std={qkv_out.std():.6f}")

    qkv = qkv_out.reshape(bs, N, 3, block.num_heads, C // block.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    print(f"  q: mean={q.mean():.6f}, std={q.std():.6f}")

    attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    attn_out = attn_out.transpose(1, 2).reshape(bs, N, C)
    attn_out = block.proj(attn_out)
    print(f"  attn_out (after proj): mean={attn_out.mean():.6f}, std={attn_out.std():.6f}")

    a1_effect = a1.unsqueeze(1).abs().mean(dim=-1).mean().item()
    print(f"  a1 scaling factor: ~{a1_effect:.6f}")
    effective_attn = (a1.unsqueeze(1) * attn_out).abs().mean().item()
    print(f"  effective attn contribution: {effective_attn:.6f}")

    # ================================================================
    # Step 2: Cross-Attention pathway
    # ================================================================
    print("\n--- Step 2: Deformable Cross-Attention ---")
    ref_pts = bbox_to_reference_points(bbox_coords)
    ref_pts = reference_points_with_levels(ref_pts, 4)

    x2 = block.norm2(box_tokens)
    x2_mod = block._modulate(x2, g2, b2)
    cross_out = block.cross_attn(
        query=x2_mod,
        reference_points=ref_pts,
        value=fpn_flattened,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    print(f"  cross_attn output: mean={cross_out.mean():.6f}, std={cross_out.std():.6f}")
    print(f"  cross_attn range: [{cross_out.min():.4f}, {cross_out.max():.4f}]")

    a2_effect = a2.unsqueeze(1).abs().mean(dim=-1).mean().item()
    print(f"  a2 scaling factor: ~{a2_effect:.6f}")
    effective_cross = (a2.unsqueeze(1) * cross_out).abs().mean().item()
    print(f"  effective cross attn contribution: {effective_cross:.6f}")

    # ================================================================
    # Step 3: FFN pathway
    # ================================================================
    print("\n--- Step 3: FFN ---")
    x3 = block.norm3(box_tokens)
    x3_mod = block._modulate(x3, g3, b3)
    ffn_out = block.ffn(x3_mod)
    print(f"  ffn output: mean={ffn_out.mean():.6f}, std={ffn_out.std():.6f}")
    print(f"  ffn range: [{ffn_out.min():.4f}, {ffn_out.max():.4f}]")

    a3_effect = a3.unsqueeze(1).abs().mean(dim=-1).mean().item()
    print(f"  a3 scaling factor: ~{a3_effect:.6f}")
    effective_ffn = (a3.unsqueeze(1) * ffn_out).abs().mean().item()
    print(f"  effective ffn contribution: {effective_ffn:.6f}")

    # ================================================================
    # Full block forward
    # ================================================================
    print("\n--- Full Block Forward ---")
    with torch.no_grad():
        output = block(box_tokens, fpn_flattened, spatial_shapes,
                       level_start_index, time_emb, bbox_coords)
    diff = (output - box_tokens).abs().mean().item()
    print(f"  Input mean: {box_tokens.abs().mean():.6f}")
    print(f"  Output mean: {output.abs().mean():.6f}")
    print(f"  Total change: {diff:.6f}")
    percent_change = diff / box_tokens.abs().mean().item() * 100
    print(f"  % change: {percent_change:.2f}%")

    # ================================================================
    # Compare different bbox inputs produce different block outputs
    # ================================================================
    print("\n--- Cross-Attention Sensitivity to Reference Points ---")
    bboxes_center = torch.full((bs, N, 4), 0.5)
    bboxes_edge = torch.full((bs, N, 4), 0.1)
    bboxes_edge[..., 2:] = 0.3

    tokens_same = torch.randn(bs, N, C)

    with torch.no_grad():
        out_center = block(tokens_same, fpn_flattened, spatial_shapes,
                           level_start_index, time_emb, bboxes_center)
        out_edge = block(tokens_same, fpn_flattened, spatial_shapes,
                         level_start_index, time_emb, bboxes_edge)

    bbox_sensitivity = (out_center - out_edge).abs().mean().item()
    print(f"  Output diff (center vs edge bboxes): {bbox_sensitivity:.6f}")
    if bbox_sensitivity < 0.01:
        print("  WARNING: Cross-attention NOT sensitive to reference point changes!")

    # ================================================================
    # Check weight initialization of adaln_mlp last layer
    # ================================================================
    print("\n--- AdaLN MLP Last Layer Weights ---")
    last_weight = block.adaln_mlp[-1].weight
    last_bias = block.adaln_mlp[-1].bias
    print(f"  Last layer weight: mean={last_weight.mean():.6f}, "
          f"std={last_weight.std():.6f}, norm={last_weight.norm():.6f}")
    print(f"  Last layer bias:   mean={last_bias.mean():.6f}, "
          f"std={last_bias.std():.6f}")
    print("  Weight gain (xavier_uniform_ with gain=0.02): "
          "std should be ~0.02/sqrt(fan_in+fan_out)")
    expected_std = 0.02 / ((last_weight.shape[0] + last_weight.shape[1]) ** 0.5)
    print(f"  Expected std: {expected_std:.6f}, Actual std: {last_weight.std():.6f}")

    print("\n" + "=" * 70)
    print("Diagnostic complete")
    print("=" * 70)

    # ================================================================
    # Summary analysis
    # ================================================================
    print("\n=== ROOT CAUSE ANALYSIS ===")
    print()
    print(f"1. AdaLN a1/a2/a3 scaling factors: {a1_effect:.6f} / {a2_effect:.6f} / {a3_effect:.6f}")
    print("   -> The gain=0.02 initialization makes a1,a2,a3 near-zero")
    print("   -> box_tokens = box_tokens + a1*attn_out ≈ box_tokens (pass-through)")
    print()
    print(f"2. Cross-attn sensitivity to bbox: {bbox_sensitivity:.6f}")
    if bbox_sensitivity < 0.01:
        print("   -> Cross-attention does not differentiate between bbox positions!")
    print()
    print(f"3. Overall block change: {diff:.6f} ({percent_change:.2f}%)")
    print("   -> DiTBlock acts as near-identity, tokens barely change")


if __name__ == '__main__':
    main()
