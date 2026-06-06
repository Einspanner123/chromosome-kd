#!/usr/bin/env python3
"""直接加载 checkpoint 检查模型参数是否学到了有意义的内容。

用法:
    cd /home/linkst/workplace/chromo/chromosome-kd
    conda run -n chromo python projects/LDMDet/tools/inspect_checkpoint.py
"""

import sys
from pathlib import Path

import torch
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def inspect_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 查找 state_dict
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    # 去除 mmengine 前缀
    clean = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[7:]
        clean[k] = v
    
    print(f'Checkpoint keys: {len(clean)}')
    print(f'Epoch: {ckpt.get("epoch", "N/A")}')
    print(f'Meta: {ckpt.get("meta", {})}')
    
    # 查找关键层: bbox_head 相关的参数
    head_keys = [k for k in clean if 'bbox_head' in k]
    print(f'\nHead-related keys: {len(head_keys)}')
    
    # 分类头 (cls) 参数
    cls_keys = [k for k in head_keys if 'cls' in k.lower()]
    reg_keys = [k for k in head_keys if 'reg' in k.lower() or 'bbox' in k.lower()]
    token_keys = [k for k in head_keys if 'token' in k.lower() or 'pos_embed' in k.lower() or 'query' in k.lower()]
    adaln_keys = [k for k in head_keys if 'adaln' in k.lower() or 'adaln_zero' in k.lower()]
    
    # 检查 key 分布
    print(f'\n  cls_params: {len(cls_keys)}')
    print(f'  reg_params: {len(reg_keys)}')
    print(f'  token_params: {len(token_keys)}')
    print(f'  adaln_params: {len(adaln_keys)}')
    
    # 重点检查: 分类头的 bias（最后一层决定各类得分）
    print('\n=== 分类头输出层 bias ===')
    for k, v in clean.items():
        if 'cls' in k.lower() and v.dim() == 1:  # bias
            print(f'  {k}: shape={v.shape} mean={v.mean():.6f} std={v.std():.6f} '
                  f'min={v.min():.6f} max={v.max():.6f}')
    
    # 检查回归头的权重
    print('\n=== 回归头输出层 ===')
    for k, v in clean.items():
        if 'reg' in k.lower() and ('head' in k.lower() or 'out' in k.lower()):
            if v.dim() <= 3:  # weights or bias
                print(f'  {k}: shape={v.shape} mean={v.mean():.6f} std={v.std():.6f} '
                      f'min={v.min():.6f} max={v.max():.6f}')
    
    # 检查 box_tokenizer 的 query_embed
    print('\n=== Box Tokenizer query_embed ===')
    for k, v in clean.items():
        if 'query' in k.lower() and v.dim() >= 2:
            print(f'  {k}: shape={v.shape} mean={v.mean():.6f} std={v.std():.6f} '
                  f'min={v.min():.6f} max={v.max():.6f}')
    
    # 检查 adaln_zero 的 alpha 是否还在零附近
    print('\n=== AdaLN-Zero alpha 参数 (应为接近 0 的小值) ===')
    for k, v in clean.items():
        if 'alpha' in k.lower() and ('adaln' in k.lower()):
            print(f'  {k}: shape={v.shape} mean={v.mean():.6f} std={v.std():.6f} '
                  f'max_abs={v.abs().max():.6f}')
            if v.abs().max() < 1e-4:
                print(f'    *** WARNING: alpha 仍然接近 0！')
    
    # 查找 conv 层
    print('\n=== DiTBlock 中的关键层 ===')
    for k, v in clean.items():
        if 'block' in k.lower() and v.dim() >= 2:
            if any(x in k.lower() for x in ['conv', 'linear', 'ffn']):
                if v.shape[0] <= 10:  # 只打印小尺寸的
                    print(f'  {k}: shape={v.shape} mean={v.mean():.6f} std={v.std():.6f}')
    
    # 检查是否有 NaN
    all_vals = []
    for k, v in clean.items():
        if v.dtype in (torch.float32, torch.float16, torch.float64):
            nans = torch.isnan(v).sum().item()
            if nans > 0:
                print(f'  NaN in {k}: {nans}/{v.numel()}')
            infs = torch.isinf(v).sum().item()
            if infs > 0:
                print(f'  Inf in {k}: {infs}/{v.numel()}')

    print('\n=== 梯度检查 (如果 checkpoint 有优化器状态) ===')
    if 'optimizer' in ckpt:
        opt = ckpt['optimizer']
        print(f'Optimizer type: {opt.get("type", "unknown")}')
        if 'param_groups' in opt:
            for pg in opt['param_groups']:
                print(f'  lr: {pg.get("lr", "?")}')
    else:
        print('  无优化器状态 (正常)')

    # 检查 backbone 是否冻结
    print('\n=== Backbone 参数是否更新 ===')
    backbone_keys_pretrained = [k for k in clean if 'backbone' in k]
    backbone_updated = 0
    for k in backbone_keys_pretrained:
        v = clean[k]
        if v.dim() >= 1:
            if v.std() > 1e-6:
                backbone_updated += 1
    print(f'  Backbone params: {len(backbone_keys_pretrained)}, updated (>0 std): {backbone_updated}')
    
    # 总结
    print('\n===== 关键诊断 =====')
    # 检查 query_embed 的多样性
    for k, v in clean.items():
        if 'query_embed' in k.lower() and v.dim() == 3:
            # [1, N, C] 
            q = v[0]  # [N, C]
            # 计算 token 之间的 cosine similarity
            q_norm = torch.nn.functional.normalize(q, dim=-1)
            sim = q_norm @ q_norm.T
            off_diag = sim[~torch.eye(sim.shape[0], dtype=bool)].mean()
            print(f'  query_embed inter-token cosine sim mean: {off_diag:.6f}')
            if off_diag > 0.99:
                print(f'  *** WARNING: 所有 query tokens 几乎完全相同! 模型无法区分不同 proposal。')
            elif off_diag > 0.9:
                print(f'  ** 注意: query tokens 高度相似，可能导致预测框重复。')


if __name__ == '__main__':
    ckpt_path = str(REPO / 'work_dirs/ldmdet_dinov3_small_384_rope_shifted3/epoch_15.pth')
    print(f'检查 checkpoint: {ckpt_path}')
    inspect_ckpt(ckpt_path)