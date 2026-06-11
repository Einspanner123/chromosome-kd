"""诊断: 验证 Self-Attention 是否导致预测框趋同 (homogenization)"""
import sys
sys.path.insert(0, '/home/linkst/workplace/chromo/chromosome-kd')

import torch
import numpy as np
torch.manual_seed(42)

import glob, os

from mmdet.apis import init_detector

cfg_path = 'projects/LDMDet/configs/ldmdet_dit.py'
ckpts = sorted(glob.glob('work_dirs/ldmdet_dit_gain05_delta/**/*.pth', recursive=True))
ckpt_path = ckpts[-1] if ckpts else None

print(f"Loading: {os.path.basename(ckpt_path) if ckpt_path else 'no ckpt'}")
model = init_detector(cfg_path, ckpt_path, device='cpu')
model.eval()
device = next(model.parameters()).device
print(f"Model ready, device={device}")

head = model.bbox_head
print(f"head type: {type(head).__name__}")
print(f"num_proposals: {head.num_proposals}")
single = head.head_series[0]  # DiTSingleHead
print(f"dit_blocks: {len(single.dit_blocks)}")

print("\n=== 测试 1: 预测框多样性 (随机图像) ===")
with torch.no_grad():
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        img = torch.randn(1, 3, 800, 800, device=device)
        feats = model.extract_feat(img)
        
        img_metas = [dict(img_shape=(800, 800), ori_shape=(800, 800), scale_factor=(1.0, 1.0))]
        results = head.predict(feats, img_metas, rescale=False)
        res = results[0]
        
        if hasattr(res, 'bboxes') and len(res.bboxes) > 0:
            bboxes = res.bboxes.cpu().numpy()
            scores = res.scores.cpu().numpy()
            labels = res.labels.cpu().numpy()
            n = len(bboxes)
            
            x1, y1, x2, y2 = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
            areas = (x2-x1)*(y2-y1)
            xx1 = np.maximum(x1[:,None], x1[None,:]); yy1 = np.maximum(y1[:,None], y1[None,:])
            xx2 = np.minimum(x2[:,None], x2[None,:]); yy2 = np.minimum(y2[:,None], y2[None,:])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou = inter / (areas[:,None] + areas[None,:] - inter + 1e-8)
            mask = ~np.eye(n, dtype=bool)
            pw_iou = iou[mask]
            
            centers = np.stack([(x1+x2)/2, (y1+y2)/2], axis=1)
            center_dist = np.linalg.norm(centers[:,None] - centers[None,:], axis=2)
            pw_dist = center_dist[mask]
            
            print(f"\n  Seed {seed}: {n} preds")
            print(f"    IoU: mean={pw_iou.mean():.4f}, med={np.median(pw_iou):.4f}, "
                  f"max={pw_iou.max():.4f}, min={pw_iou.min():.4f}")
            print(f"    中心距离: mean={pw_dist.mean():.1f}, med={np.median(pw_dist):.1f}")
            print(f"    分数: [{scores.min():.4f}, {scores.max():.4f}], std={scores.std():.4f}")
            print(f"    标签种类: {len(np.unique(labels))}")
            w, h = x2-x1, y2-y1
            print(f"    框尺寸: w=[{w.min():.1f},{w.max():.1f}], h=[{h.min():.1f},{h.max():.1f}]")
            print(f"    IoU>0.5 比例: {(pw_iou>0.5).mean():.4f}")
        else:
            print(f"\n  Seed {seed}: NO PREDICTIONS!")

print("\n=== 测试 2: Box Token 在 Self-Attention 前后的多样性 ===")
with torch.no_grad():
    torch.manual_seed(42)
    img = torch.randn(1, 3, 800, 800, device=device)
    feats = model.extract_feat(img)
    
    bs, N = 1, head.num_proposals
    
    # 推理时的初始噪声框 (与推理流程一致)
    noisy_boxes = head._init_inference_boxes(bs, device)  # raw格式: x_start
    img_metas = [dict(img_shape=(800, 800), ori_shape=(800, 800), scale_factor=(1.0, 1.0))]
    noisy_boxes_xyxy = head._raw_to_xyxy(noisy_boxes, img_metas)
    print(f"\n  初始噪声框: x=[{noisy_boxes_xyxy[:,:,0].min():.3f},{noisy_boxes_xyxy[:,:,0].max():.3f}], "
          f"y=[{noisy_boxes_xyxy[:,:,1].min():.3f},{noisy_boxes_xyxy[:,:,1].max():.3f}]")
    
    from projects.LDMDet.mods.deformable_attn import flatten_fpn_features
    fpn_f, ss, lsi = flatten_fpn_features(feats)
    
    box_tokens, _ = head.box_tokenizer(noisy_boxes_xyxy, feats)
    
    def tok_div(tokens, name):
        tn = tokens.float() / (tokens.float().norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(tn, tn.transpose(1, 2))  # (bs, N, N)
        bs, N, _ = sim.shape
        mask = ~torch.eye(N, dtype=torch.bool, device=tokens.device)
        # 提取 pairwise (非对角线) 值
        pw = sim[:, mask]  # (bs, N*(N-1))
        print(f"  {name}: cos_sim mean={pw.mean():.4f}, std={pw.std():.4f}, "
              f"[{pw.min():.4f}, {pw.max():.4f}]")
        return pw
    
    print("\n  Token 多样性变化:")
    tok_div(box_tokens, "初始")
    
    time_emb = head.time_mlp(torch.zeros(bs, device=device))
    for i, block in enumerate(single.dit_blocks):
        box_tokens = block(box_tokens, fpn_f, ss, lsi, time_emb, noisy_boxes_xyxy)
        tok_div(box_tokens, f"Block {i+1}")
    
    # 最终预测
    cls_logits = single.cls_head(box_tokens)
    bbox_preds = single.reg_head(box_tokens)
    scores = torch.sigmoid(cls_logits)
    
    print(f"\n  最终预测:")
    print(f"    分类分数: mean={scores.mean():.4f}, std={scores.std():.4f}, max={scores.max():.4f}")
    print(f"    bbox_preds: x=[{bbox_preds[:,:,0].min():.4f},{bbox_preds[:,:,0].max():.4f}], "
          f"y=[{bbox_preds[:,:,1].min():.4f},{bbox_preds[:,:,1].max():.4f}], "
          f"w=[{bbox_preds[:,:,2].min():.4f},{bbox_preds[:,:,2].max():.4f}], "
          f"h=[{bbox_preds[:,:,3].min():.4f},{bbox_preds[:,:,3].max():.4f}]")

print("\n=== 完成 ===")