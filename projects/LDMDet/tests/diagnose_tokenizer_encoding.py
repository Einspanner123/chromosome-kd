"""诊断: BoxTokenizer 编码区分度分析 — 定位 cos_sim=0.80 的根因"""
import sys
sys.path.insert(0, '/home/linkst/workplace/chromo/chromosome-kd')

import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(42)

import glob, os

from mmdet.apis import init_detector

cfg_path = 'projects/LDMDet/configs/ldmdet_dit.py'
ckpts = sorted(glob.glob('work_dirs/ldmdet_dit_gain05_delta/**/*.pth', recursive=True))
ckpt_path = ckpts[-1] if ckpts else None

print(f"Loading checkpoint: {os.path.basename(ckpt_path)}")
model = init_detector(cfg_path, ckpt_path, device='cpu')
model.eval()
device = next(model.parameters()).device

head = model.bbox_head
tokenizer = head.box_tokenizer
feat_channels = head.feat_channels

print(f"feat_channels: {feat_channels}")
print(f"init_mode: {tokenizer.init_mode}")

# 生成 100 个均匀分布在图上的框 (模拟推理时的初始框)
bs, N = 1, head.num_proposals  # 100
# 10x10 grid of boxes
grid_size = 10
step = 1.0 / grid_size
cx = torch.linspace(step/2, 1-step/2, grid_size)
cy = torch.linspace(step/2, 1-step/2, grid_size)
cx_grid, cy_grid = torch.meshgrid(cx, cy, indexing='ij')
# 去归一化: 归一化坐标 [0,1] → 像素坐标 [0,800]
cx_px = cx_grid.flatten()[:N] * 800
cy_px = cy_grid.flatten()[:N] * 800
box_size = 40  # 近似初始框大小
x1 = (cx_px - box_size/2).clamp(0, 760)
y1 = (cy_px - box_size/2).clamp(0, 760)
x2 = (cx_px + box_size/2).clamp(40, 800)
y2 = (cy_px + box_size/2).clamp(40, 800)
bboxes_img = torch.stack([x1, y1, x2, y2], dim=1).unsqueeze(0).to(device)  # (1, 100, 4) 像素坐标
bboxes_norm = bboxes_img / 800  # (1, 100, 4) 归一化坐标

print(f"\n框坐标范围: x=[{bboxes_img[:,:,0].min():.0f},{bboxes_img[:,:,0].max():.0f}], "
      f"y=[{bboxes_img[:,:,1].min():.0f},{bboxes_img[:,:,1].max():.0f}]")
print(f"不同框之间的 L2 距离 (像素): mean={torch.cdist(bboxes_img[0], bboxes_img[0]).mean():.0f}")

# 生成 FPN 特征
img = torch.randn(bs, 3, 800, 800, device=device)
feats = model.extract_feat(img)

# ==========================================
# 1. 分解 Box Tokenizer 各部分
# ==========================================
print("\n" + "="*60)
print("1. Box Tokenizer 各组件分解")
print("="*60)

with torch.no_grad():
    level_indices = tokenizer._assign_fpn_level(bboxes_norm)
    print(f"FPN 层级分布: {torch.bincount(level_indices[0].cpu())}")
    
    # pos_embed
    pos_embed = tokenizer.bbox_pos_embed(bboxes_norm)  # (1, 100, C)
    # lvl_embed
    lvl_embed = tokenizer.level_embed(level_indices)   # (1, 100, C)
    # content (单点采样)
    content = tokenizer._sample_content_features(bboxes_norm, feats)  # (1, 100, C)
    content_proj = tokenizer.content_proj(content)      # (1, 100, C)
    
    # 初始项 (init_mode='zero' → 全零)
    sampled_feat = torch.zeros(bs, N, feat_channels, device=device)
    
    # 完整 token
    full_tokens = sampled_feat + pos_embed + lvl_embed + content_proj

def analyze_component(tensor, name):
    """分析单个组件的统计特性"""
    t = tensor[0]  # (N, C)
    # 每个通道的方差 (跨 proposal)
    channel_var = t.var(dim=0).mean().item()
    # 每个 proposal 的范数
    token_norms = t.norm(dim=-1)
    # 归一化后的余弦相似度
    t_norm = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = torch.matmul(t_norm, t_norm.T)
    mask = ~torch.eye(N, dtype=torch.bool)
    pw_cos = cos_sim[mask]
    
    print(f"\n  [{name}]")
    print(f"    magnitude: mean={token_norms.mean():.2f}, std={token_norms.std():.2f}")
    print(f"    cross-proposal channel var: {channel_var:.6f}")
    print(f"    cos_sim: mean={pw_cos.mean():.4f}, std={pw_cos.std():.4f}, "
          f"[{pw_cos.min():.4f}, {pw_cos.max():.4f}]")
    return pw_cos

analyze_component(pos_embed, "pos_embed (raw, 无 LayerNorm)")
pos_embed_normed = tokenizer.pos_norm(pos_embed)
analyze_component(pos_embed_normed, "pos_embed (+LayerNorm)")
analyze_component(lvl_embed, "lvl_embed (FPN level)")
analyze_component(content_proj, "content_proj (单点采样)")

# 完整 token (通过 forward 方法)
full_tokens, _ = tokenizer(bboxes_norm, feats)
analyze_component(full_tokens, "完整 token (forward, 含 pos_norm)")

# ==========================================
# 2. 测试 pos_embed 对不同坐标的响应
# ==========================================
print("\n" + "="*60)
print("2. bbox_pos_embed MLP 对不同坐标的区分能力")
print("="*60)

# 构造极端坐标: 四个角和中心
test_coords = torch.tensor([
    [0, 0, 0.1, 0.1],          # 左上角
    [0.9, 0, 1.0, 0.1],        # 右上角
    [0, 0.9, 0.1, 1.0],        # 左下角
    [0.9, 0.9, 1.0, 1.0],      # 右下角
    [0.45, 0.45, 0.55, 0.55],  # 中心
    [0.2, 0.3, 0.5, 0.6],      # 随机
    [0.6, 0.1, 0.9, 0.4],      # 随机
], device=device).unsqueeze(0)  # (1, 7, 4)

with torch.no_grad():
    test_embed = tokenizer.bbox_pos_embed(test_coords)  # (1, 7, C)
    t_norm = test_embed / (test_embed.norm(dim=-1, keepdim=True) + 1e-8)
    cos_mat = torch.matmul(t_norm[0], t_norm[0].T)
    
    labels = ['左上', '右上', '左下', '右下', '中心', '随机A', '随机B']
    print(f"\n  极端坐标间的余弦相似度矩阵:")
    print(f"  {'':>8}", end='')
    for l in labels:
        print(f"{l:>8}", end='')
    print()
    for i, l in enumerate(labels):
        print(f"  {l:>8}", end='')
        for j in range(7):
            print(f"{cos_mat[i,j].item():8.4f}", end='')
        print()
    
    pw_mask = ~torch.eye(7, dtype=torch.bool)
    pw_cos = cos_mat[pw_mask]
    print(f"\n  pairwise cos_sim: mean={pw_cos.mean():.4f}, "
          f"[{pw_cos.min():.4f}, {pw_cos.max():.4f}]")

# 测试: 坐标距离 vs 嵌入相似度
print(f"\n  坐标欧氏距离 vs 嵌入余弦相似度 (100个框):")
with torch.no_grad():
    coord_dist = torch.cdist(bboxes_norm[0], bboxes_norm[0], p=2)  # (100, 100)
    pe = tokenizer.bbox_pos_embed(bboxes_norm)
    pe_norm = pe / (pe.norm(dim=-1, keepdim=True) + 1e-8)
    embed_sim = torch.matmul(pe_norm[0], pe_norm[0].T)  # (100, 100)
    
    mask = ~torch.eye(N, dtype=torch.bool)
    cd = coord_dist[mask].cpu().numpy()
    es = embed_sim[mask].cpu().numpy()
    
    # 按坐标距离分组
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    for i in range(len(bins)-1):
        in_bin = (cd >= bins[i]) & (cd < bins[i+1])
        if in_bin.sum() > 0:
            print(f"    coord_dist [{bins[i]:.1f},{bins[i+1]:.1f}): "
                  f"n={in_bin.sum():5d}, cos_sim mean={es[in_bin].mean():.4f}")

# ==========================================
# 3. 分析 MLP 权重潘化
# ==========================================
print("\n" + "="*60)
print("3. bbox_pos_embed MLP 权重分析")
print("="*60)

mlp = tokenizer.bbox_pos_embed
for i, layer in enumerate(mlp):
    if isinstance(layer, nn.Linear):
        w = layer.weight.data
        print(f"  Linear[{i}]: in={w.shape[1]}, out={w.shape[0]}, "
              f"weight_norm={w.norm().item():.4f}, "
              f"weight_std={w.std().item():.4f}, "
              f"bias_std={layer.bias.std().item():.4f}")

# 检查第一层权重: 4个输入通道 (x1,y1,x2,y2) 的权重分布
w0 = mlp[0].weight.data  # (384, 4)
print(f"\n  第一层对各坐标的敏感度 (weight L2 norm):")
print(f"    x1: {w0[:,0].norm().item():.4f}")
print(f"    y1: {w0[:,1].norm().item():.4f}")
print(f"    x2: {w0[:,2].norm().item():.4f}")
print(f"    y2: {w0[:,3].norm().item():.4f}")

# ==========================================
# 4. 对比: 正弦位置编码方案
# ==========================================
print("\n" + "="*60)
print("4. 对比: 正弦位置编码 (sin/cos Fourier features)")
print("="*60)

def sinusoidal_position_encoding(coords, num_freqs=64, temperature=10000):
    """
    coords: (bs, N, 4) 归一化坐标 [0,1]
    返回: (bs, N, C) 其中 C = 4 * num_freqs * 2 = 4*64*2=512
    对每个坐标维度 (x1,y1,x2,y2) 应用 sin/cos 编码
    """
    bs, N, D = coords.shape
    device = coords.device
    
    # 频率: (num_freqs,)
    freqs = 1.0 / (temperature ** (torch.arange(0, num_freqs, device=device).float() / num_freqs))
    
    # coords: (bs, N, 4, 1) * freqs: (1, 1, 1, num_freqs) → (bs, N, 4, num_freqs)
    angles = coords.unsqueeze(-1) * freqs.view(1, 1, 1, -1) * (2 * torch.pi)
    
    sin_part = torch.sin(angles)  # (bs, N, 4, num_freqs)
    cos_part = torch.cos(angles)  # (bs, N, 4, num_freqs)
    
    encoding = torch.cat([sin_part, cos_part], dim=-1)  # (bs, N, 4, 2*num_freqs)
    encoding = encoding.reshape(bs, N, D * 2 * num_freqs)  # (bs, N, 512)
    
    return encoding

with torch.no_grad():
    sin_enc = sinusoidal_position_encoding(bboxes_norm, num_freqs=48)
    # 投影到 384 维
    sin_enc = sin_enc[:, :, :feat_channels]  # 取前384维
    
    analyze_component(sin_enc, "正弦编码 (前384维)")
    
    # 坐标距离 vs 正弦编码相似度
    print(f"\n  坐标距离 vs 正弦编码余弦相似度:")
    se_norm = sin_enc / (sin_enc.norm(dim=-1, keepdim=True) + 1e-8)
    se_sim = torch.matmul(se_norm[0], se_norm[0].T)
    se_sim_flat = se_sim[mask].cpu().numpy()
    
    for i in range(len(bins)-1):
        in_bin = (cd >= bins[i]) & (cd < bins[i+1])
        if in_bin.sum() > 0:
            print(f"    coord_dist [{bins[i]:.1f},{bins[i+1]:.1f}): "
                  f"n={in_bin.sum():5d}, "
                  f"MLP cos_sim={es[in_bin].mean():.4f}, "
                  f"正弦 cos_sim={se_sim_flat[in_bin].mean():.4f}")

print("\n" + "="*60)
print("诊断完成")
print("="*60)