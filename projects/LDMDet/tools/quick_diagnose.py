#!/usr/bin/env python
"""快速诊断模型预测输出"""
import sys

sys.path.insert(0, 'projects/LDMDet')

import mmengine
import torch
from mmengine.config import Config

cfg = Config.fromfile('projects/LDMDet/configs/ldmdet_dit.py')
cfg.work_dir = 'work_dirs/ldmdet_dinov3_small_384_rope_shifted3'
cfg.load_from = 'work_dirs/ldmdet_dinov3_small_384_rope_shifted3/epoch_12.pth'

runner = mmengine.runner.Runner.from_cfg(cfg)
model = runner.model
model.eval()

val_loader = runner.val_dataloader
batch = next(iter(val_loader))
# 通过 data_preprocessor 处理
data = model.data_preprocessor(batch, False)
inputs = data['inputs']
data_samples = data['data_samples']

device = next(model.parameters()).device
inputs = inputs.to(device) if isinstance(inputs, torch.Tensor) else inputs

with torch.no_grad():
    results = model.predict(inputs, data_samples)

for i, (ds, res) in enumerate(zip(data_samples, results)):
    pred = res.pred_instances
    gt = ds.gt_instances
    print(f'Image {i}: ori={ds.ori_shape}, img={ds.img_shape}')
    print(f'  GT count: {len(gt.bboxes)}, pred count: {len(pred.bboxes)}')
    if len(pred.bboxes) > 0:
        print(f'  Pred scores: [{pred.scores.min():.4f}, {pred.scores.max():.4f}]')
        print(f'  Pred bbox range: x=[{pred.bboxes[:,0].min():.0f},{pred.bboxes[:,2].max():.0f}] y=[{pred.bboxes[:,1].min():.0f},{pred.bboxes[:,3].max():.0f}]')
        if len(gt.bboxes) > 0:
            # GT bbox 在 resized image 坐标下，需转换到原始图像坐标
            sf = gt.bboxes.new_tensor(ds.scale_factor).repeat((1, 2))
            gt_orig = gt.bboxes / sf
            from torchvision.ops import box_iou
            ious = box_iou(pred.bboxes, gt_orig)
            max_iou = ious.max(dim=1)[0]
            print(f'  Max IoU: [{max_iou.min():.4f}, {max_iou.max():.4f}]')
            print(f'  IoU>0.3: {(max_iou>0.3).sum()}, IoU>0.5: {(max_iou>0.5).sum()}')
            print(f'  GT bbox range: x=[{gt_orig[:,0].min():.0f},{gt_orig[:,2].max():.0f}] y=[{gt_orig[:,1].min():.0f},{gt_orig[:,3].max():.0f}]')
    if i >= 1:
        break
