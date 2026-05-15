"""
Diagnose ConvNeXt NaN: 逐阶段检查 pipeline 中 NaN 首次出现位置。

逐步测试:
  1. TIMMBackbone (ConvNeXt-Base + 预训练权重) → 特征输出
  2. + FPN neck
  3. + Single Head 逐层前向 (RoI → SA → DynConv → FFN → cls/reg)

每步检查 min/max/std/nan/inf。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.LDMDet.model import TIMMBackbone

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def stats(x, name=''):
    xf = x.float()
    return {
        'name': name,
        'shape': tuple(x.shape),
        'min': xf.min().item(),
        'max': xf.max().item(),
        'mean': xf.mean().item(),
        'std': xf.std().item(),
        'has_nan': xf.isnan().any().item(),
        'has_inf': xf.isinf().any().item(),
        'nan_count': xf.isnan().sum().item(),
        'inf_count': xf.isinf().sum().item(),
    }


def print_stats(s):
    flag = ''
    if s['has_nan']:
        flag = ' *** NaN! ***'
    if s['has_inf']:
        flag = ' *** Inf! ***'
    print(
        f"  [{s['name']:30s}] shape={s['shape']} | "
        f"min={s['min']:9.4f} max={s['max']:9.4f} mean={s['mean']:9.4f} std={s['std']:9.4f} | "
        f"nan={s['has_nan']}({s['nan_count']}) inf={s['has_inf']}({s['inf_count']}){flag}"
    )


# ============================================================
# Stage 1: TIMMBackbone
# ============================================================
print('=' * 70)
print('Stage 1: TIMMBackbone (ConvNeXt-Base)')
print('=' * 70)

backbone = TIMMBackbone(
    model_name='convnext_base',
    features_only=True,
    pretrained=False,
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.4,
    frozen_stages=2,
    checkpoint_path='checkpoints/convnext_base_22k_1k_224.pth',
)
backbone.eval()
backbone.to(DEVICE)

# Input matching dataset: [1, 3, 800, 1333]
x = torch.randn(1, 3, 800, 1333, device=DEVICE)
print(f'Input: {tuple(x.shape)}')

with torch.no_grad():
    feats = backbone(x)
    for i, f in enumerate(feats):
        s = stats(f, f'backbone_stage_{i}')
        print_stats(s)
print()

# ============================================================
# Stage 2: + FPN (train mode 检查 dropout/stoch depth 影响)
# ============================================================
print('=' * 70)
print('Stage 2: + FPN Neck (train mode)')
print('=' * 70)

from mmdet.models.necks import FPN

fpn = FPN(
    in_channels=[128, 256, 512, 1024],
    out_channels=256,
    num_outs=4,
)
fpn.train()
fpn.to(DEVICE)

with torch.no_grad():
    feats_bb = backbone(x)
    fpn_feats = fpn(feats_bb)
    for i, f in enumerate(fpn_feats):
        s = stats(f, f'fpn_level_{i}')
        print_stats(s)
print()

# ============================================================
# Stage 3: Single Head 逐层检查
# ============================================================
print('=' * 70)
print('Stage 3: Single Head — 逐层检查')
print('=' * 70)

from mmcv.ops import RoIAlign

from projects.LDMDet.mods.modules import SinusoidalPositionEmbeddings
from projects.LDMDet.mods.single_head import SingleDiffusionDetHead
from projects.LDMDet.mods.utils import bbox2roi

torch.manual_seed(42)
np.random.seed(42)

feat_channels = 256
num_proposals = 500
num_classes = 24
bs = 2  # use batch_size=2 to match training

# Build RoIAlign (use FPN level 0, stride 4, spatial_scale=1/4)
roi_pooler = RoIAlign(output_size=7, spatial_scale=1.0 / 4.0, sampling_ratio=0)
roi_pooler.to(DEVICE)

# Time embedding
time_mlp = nn.Sequential(
    SinusoidalPositionEmbeddings(feat_channels),
    nn.Linear(feat_channels, feat_channels * 4),
    nn.GELU(),
    nn.Linear(feat_channels * 4, feat_channels * 4),
).to(DEVICE)

# Build single_head (train mode)
single_head = SingleDiffusionDetHead(
    num_classes=num_classes,
    feat_channels=feat_channels,
    time_conditioning='adaln_zero',
    use_objectness=False,
    prediction_mode='velocity',
)
single_head.train()
single_head.to(DEVICE)

# Re-generate backbone+FPN features with matching batch_size
x2 = torch.randn(bs, 3, 800, 1333, device=DEVICE)
with torch.no_grad():
    feats_bb2 = backbone(x2)
    fpn_feats2 = fpn(feats_bb2)
print(f'FPN shapes (bs={bs}): {[tuple(f.shape) for f in fpn_feats2]}')

# Random bboxes within image [0, 800] x [0, 1333]
bboxes = torch.zeros(bs, num_proposals, 4, device=DEVICE)
for i in range(bs):
    x1 = torch.rand(num_proposals, device=DEVICE) * 1200
    y1 = torch.rand(num_proposals, device=DEVICE) * 700
    x2 = (x1 + torch.rand(num_proposals, device=DEVICE) * 133 +
          10).clamp(max=1333)
    y2 = (y1 + torch.rand(num_proposals, device=DEVICE) * 100 +
          10).clamp(max=800)
    bboxes[i, :, 0] = x1
    bboxes[i, :, 1] = y1
    bboxes[i, :, 2] = x2
    bboxes[i, :, 3] = y2

t = torch.rand(bs, device=DEVICE) * 0.5 + 0.5
time_emb = time_mlp(t * 1000)

print(f'bboxes range: [{bboxes.min():.1f}, {bboxes.max():.1f}]')
print(f'time_emb range: [{time_emb.min():.4f}, {time_emb.max():.4f}]')

# --- Step 3a: RoI Align (only level 0 features for simplicity) ---
rois = bbox2roi([bboxes[i] for i in range(bs)])
# Use FPN level 0 (P2) only — stride 4, spatial_scale=1/4
single_feat = fpn_feats2[0]  # (bs, 256, H/4, W/4)
roi_features = roi_pooler(single_feat, rois)  # (bs*num_boxes, 256, 7, 7)
s = stats(roi_features, 'roi_features (from fpn[0])')
print_stats(s)

# --- Step 3b: Proposals from RoI mean ---
proposals = roi_features.flatten(2).mean(-1).view(bs, num_proposals,
                                                  feat_channels)
s = stats(proposals, 'proposals_init')
print_stats(s)

# --- Step 3c: AdaLN params ---
adaln_params = single_head.adaln_mlp(time_emb)  # (bs, 6*feat_channels)
adaln_params = torch.repeat_interleave(adaln_params, num_proposals, dim=0)
gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_params.chunk(6, dim=-1)
s_g1 = stats(gamma1, 'gamma1')
print_stats(s_g1)
s_b1 = stats(beta1, 'beta1')
print_stats(s_b1)
s_a1 = stats(alpha1, 'alpha1')
print_stats(s_a1)
s_g2 = stats(gamma2, 'gamma2')
print_stats(s_g2)

# --- Step 3d: Block 1 — Self-Attention with AdaLN ---
proposals_sa = proposals.view(bs, num_proposals,
                              feat_channels).permute(1, 0, 2)
proposals_flat = proposals_sa.reshape(num_proposals * bs, feat_channels)
s = stats(proposals_flat, 'before SA modulation')
print_stats(s)

q_modulated = F.layer_norm(proposals_flat,
                           [feat_channels]) * (1 + gamma1) + beta1
q_modulated = q_modulated.view(num_proposals, bs, feat_channels)
s = stats(q_modulated, 'q_modulated_SA')
print_stats(s)

attn_out, _ = single_head.self_attn(
    q_modulated, q_modulated, value=q_modulated)
s = stats(attn_out, 'attn_out')
print_stats(s)

attn_out_flat = attn_out.reshape(num_proposals * bs, feat_channels)
proposals_flat = proposals_flat + alpha1 * attn_out_flat
proposals_sa = proposals_flat.view(num_proposals, bs, feat_channels)
s = stats(proposals_flat, 'after SA + alpha1 gate')
print_stats(s)

# --- Step 3e: Block 2 — DynamicConv ---
proposals_dc = proposals_sa.permute(1, 0, 2).reshape(1, bs * num_proposals,
                                                     feat_channels)
roi_feat_for_dc = roi_features.view(bs * num_proposals, feat_channels,
                                    -1).permute(2, 0, 1)
s = stats(roi_feat_for_dc, 'roi_feat → DynConv key/value')
print_stats(s)
s = stats(proposals_dc, 'proposals → DynConv query')
print_stats(s)

inst_out = single_head.inst_interact(proposals_dc, roi_feat_for_dc)
s = stats(inst_out, 'inst_interact_out')
print_stats(s)

proposals_dc = proposals_dc + single_head.dropout2(inst_out)
obj_features = single_head.norm2(proposals_dc)
s = stats(obj_features, 'obj_features (after DynConv+LN)')
print_stats(s)

# --- Step 3f: Block 3 — FFN + AdaLN ---
obj_flat = obj_features.squeeze(0)
s = stats(obj_flat, 'before FFN modulation')
print_stats(s)
ffn_input = F.layer_norm(obj_flat, [feat_channels]) * (1 + gamma2) + beta2
ffn_out = single_head.linear2(
    single_head.dropout(single_head.act(single_head.linear1(ffn_input))))
s = stats(ffn_out, 'ffn_out')
print_stats(s)
obj_flat = obj_flat + alpha2 * ffn_out
s = stats(obj_flat, 'fc_feature (after FFN+alpha2)')
print_stats(s)

# --- Step 3g: cls_head ---
class_logits = single_head.cls_head(obj_flat).float()
s = stats(class_logits, 'class_logits')
print_stats(s)

# --- Step 3h: reg_head ---
pred_bboxes = single_head._predict_bboxes(obj_flat, bboxes)
s = stats(pred_bboxes, 'pred_bboxes')
print_stats(s)

# ============================================================
# Diagnose DynamicConv internals
# ============================================================
print()
print('=' * 70)
print('Stage 4: DynamicConv 内部逐层检查')
print('=' * 70)

dyn = single_head.inst_interact
features = roi_feat_for_dc.transpose(0, 1)  # (N, 49, 256)
parameters = dyn.dynamic_layer(proposals_dc.squeeze(0))  # (N, num_params*2)
s = stats(features, 'dyn_features (49 tokens)')
print_stats(s)
s = stats(parameters, 'dyn_parameters')
print_stats(s)

param1, param2 = parameters.chunk(2, dim=1)
param1 = param1.view(-1, feat_channels, dyn.dynamic_dim)  # (N, 256, 64)
s = stats(param1, 'dyn_param1')
print_stats(s)

features_bmm1 = torch.bmm(features, param1)  # (N, 49, 64)
s = stats(features_bmm1, 'after bmm1')
print_stats(s)

features_n1 = dyn.norm1(features_bmm1)  # LayerNorm
s = stats(features_n1, 'after LN1')
print_stats(s)

features_a1 = dyn.act(features_n1)  # ReLU
s = stats(features_a1, 'after ReLU1')
print_stats(s)

param2 = param2.view(-1, dyn.dynamic_dim, feat_channels)  # (N, 64, 256)
s = stats(param2, 'dyn_param2')
print_stats(s)

features_bmm2 = torch.bmm(features_a1, param2)  # (N, 49, 256)
s = stats(features_bmm2, 'after bmm2')
print_stats(s)

features_n2 = dyn.norm2(features_bmm2)  # LayerNorm
s = stats(features_n2, 'after LN2')
print_stats(s)

features_a2 = dyn.act(features_n2)  # ReLU
s = stats(features_a2, 'after ReLU2')
print_stats(s)

features_flat = features_a2.reshape(features_a2.size(0), -1)  # (N, 49*256)
s = stats(features_flat, 'dyn_flattened')
print_stats(s)

features_out = dyn.out_layer(features_flat)
s = stats(features_out, 'dyn_out_layer')
print_stats(s)

features_n3 = dyn.norm3(features_out)
s = stats(features_n3, 'dyn_out_LN3')
print_stats(s)

features_final = dyn.act(features_n3)
s = stats(features_final, 'dyn_out_ReLU3')
print_stats(s)

# ============================================================
# Final summary
# ============================================================
print()
print('=' * 70)
all_results = [class_logits, pred_bboxes]
has_nan = any(x.isnan().any().item() for x in all_results)
has_inf = any(x.isinf().any().item() for x in all_results)
if has_nan:
    print('*** RESULT: NaN FOUND in single-head output ***')
elif has_inf:
    print('*** RESULT: Inf FOUND in single-head output ***')
else:
    print('*** RESULT: ALL CLEAN — single head is fine ***')

# ============================================================
# Stage 5: Full multi-head cascade (6 heads) with backbone in train mode
# ============================================================
print()
print('=' * 70)
print('Stage 5: Full 6-head cascade (backbone + FPN in TRAIN mode)')
print('=' * 70)

backbone.train()
fpn.train()

from mmdet.models.roi_heads import SingleRoIExtractor
from projects.LDMDet.mods.diffusiondet_head import DiffusionDetHead

# Build proper RoI extractor with all FPN levels
roi_extractor = SingleRoIExtractor(
    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
    out_channels=feat_channels,
    featmap_strides=[4, 8, 16, 32],
)
roi_extractor.to(DEVICE)

# Full head (no cross-attn, no criterion)
head = DiffusionDetHead(
    num_classes=num_classes,
    feat_channels=feat_channels,
    num_proposals=num_proposals,
    num_heads=6,
    single_head=copy.deepcopy(single_head),
    roi_extractor=roi_extractor,
    criterion=None,
    diffusion_type='rectified_flow',
    rf_schedule='shifted',
    ot_coupling=False,  # no OT for diagnostic
    prediction_mode='velocity',
)
head.train()
head.to(DEVICE)

# Run backbone + FPN in train mode
x3 = torch.randn(bs, 3, 800, 1333, device=DEVICE)
with torch.no_grad():
    feats_bb3 = backbone(x3)
    fpn_feats3 = fpn(feats_bb3)
    for i, f in enumerate(fpn_feats3):
        s = stats(f, f'fpn3_level_{i}')
        print_stats(s)

# Generate noisy boxes in diffusion space (matching actual training)
noise = torch.randn(bs, num_proposals, 4, device=DEVICE)
t_raw = torch.rand(bs, device=DEVICE) * 0.5 + 0.5

# raw_to_xyxy for display
img_metas = [{'img_shape': (800, 1333)} for _ in range(bs)]
curr_bboxes = head._raw_to_xyxy(noise, img_metas)
t_input = t_raw * head.timesteps

print(
    f'noisy bboxes range: [{curr_bboxes.min():.1f}, {curr_bboxes.max():.1f}]')

# Full forward pass through all 6 heads
print('\nRunning full 6-head forward...')
cls_logits_seq, pred_bboxes_seq, obj_seq, vel_seq = head(
    fpn_feats3, curr_bboxes, t_input)

print('Per-head outputs:')
has_nan = False
for i in range(6):
    cls_s = stats(cls_logits_seq[i], f'head_{i}_cls_logits')
    bbox_s = stats(pred_bboxes_seq[i], f'head_{i}_pred_bboxes')
    vel_s = stats(vel_seq[i],
                  f'head_{i}_velocity') if vel_seq[i] is not None else None
    print_stats(cls_s)
    print_stats(bbox_s)
    if vel_s:
        print_stats(vel_s)
    if cls_s['has_nan'] or bbox_s['has_nan']:
        has_nan = True
        print(f'  *** NaN at head {i}! ***')
    print()

print('=' * 70)
if has_nan:
    print(
        '*** RESULT: NaN FOUND in cascade — NaN propagates from earlier head ***'
    )
else:
    print(
        '*** RESULT: ALL CLEAN in cascade — NaN likely from LOSS computation ***'
    )
    print('    → Next test: actual loss computation')
print('=' * 70)
