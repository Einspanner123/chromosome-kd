"""Diagnose: check what the model actually predicts during inference"""
import sys
sys.path.insert(0, '/home/linkst/workplace/chromo/chromosome-kd')

import torch
torch.manual_seed(42)

from mmengine.config import Config
from mmengine.registry import MODELS

# Import and register model
print("Importing model module...")
from projects.LDMDet.model import PurePyTorchDiffusionDet
from projects.LDMDet.mods import box_tokenizer  # register box_tokenizer components
MODELS.register_module(name='LDMDet', module=PurePyTorchDiffusionDet, force=True)
MODELS.register_module(module=PurePyTorchDiffusionDet, force=True)
print("Model registered OK")

cfg = Config.fromfile('projects/LDMDet/configs/ldmdet_dit.py')
cfg.model.bbox_head.num_proposals = 100

print("Building model via MODELS.build...")
model = MODELS.build(cfg.model)
model.eval()
device = next(model.parameters()).device
print(f"Model built, device={device}")

# Try loading checkpoint
ckpt_path = 'work_dirs/ldmdet_dit_gain05_delta/epoch_2.pth'
import os
if os.path.exists(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print("Checkpoint loaded")
else:
    import glob
    ckpts = glob.glob('work_dirs/ldmdet_dit_gain05_delta/**/*.pth', recursive=True)
    print(f"Available: {[os.path.basename(c) for c in ckpts]}")
    if ckpts:
        ckpt_path = ckpts[0]
        print(f"Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print("Checkpoint loaded")

print("\n=== predict() with tensor ===")
with torch.no_grad():
    img_tensor = torch.rand(1, 3, 800, 800, device=device)
    feats = model.extract_feat(img_tensor)
    for i, f in enumerate(feats):
        print(f"  feat {i}: shape={f.shape}, mean={f.mean():.4f}")
    
    from projects.LDMDet.mods.dit_head import ImageMeta
    metas = [ImageMeta(img_shape=(800, 800), ori_shape=(800, 800), scale_factor=(1.0, 1.0))]
    
    results = model.bbox_head.predict(feats, metas, rescale=False)
    res = results[0]
    if hasattr(res, 'bboxes'):
        n = len(res.bboxes)
        print(f"\nPredictions: {n} boxes")
        if n > 0:
            scores = res.scores
            bboxes = res.bboxes
            print(f"  Scores: min={scores.min():.4f}, max={scores.max():.4f}, median={scores.median():.4f}")
            print(f"  Box coords: x=[{bboxes[:,0].min():.1f},{bboxes[:,2].max():.1f}], y=[{bboxes[:,1].min():.1f},{bboxes[:,3].max():.1f}]")
            w = bboxes[:,2] - bboxes[:,0]
            h = bboxes[:,3] - bboxes[:,1]
            print(f"  Sizes: w=[{w.min():.1f},{w.max():.1f}], h=[{h.min():.1f},{h.max():.1f}]")
            print(f"  Valid (w>0,h>0): {(w>0).logical_and(h>0).sum().item()}/{n}")
            # Show first 5 preds
            print(f"  Top-5 scores: {sorted(scores.tolist(), reverse=True)[:5]}")
        else:
            print("  NO PREDICTIONS!")
    else:
        print(f"  Unexpected type: {type(res)}")

print("\n=== simple_predict (full pipeline) ===")
with torch.no_grad():
    import numpy as np
    img = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    results = model.simple_predict(img)
    for i, res in enumerate(results):
        pred = res.pred_instances
        n = len(pred.bboxes)
        print(f"  Result {i}: {n} boxes, scores=[{pred.scores.min():.4f},{pred.scores.max():.4f}]" if n > 0 else f"  Result {i}: EMPTY")

print("\n=== Done ===")