"""
LDMDet-DiT + DINOv3 全链路逐层追踪测试

测试流程（从数据加载到 Loss）:
  1. 数据加载 & pipeline (真实图像 + 增强)
  2. DataPreprocessor (归一化 + pad)
  3. Backbone (DINOv3) 输出形状
  4. Neck (SimpleFeatureFusion) 多尺度输出形状
  5. DiTDiffusionDetHead.forward (含 BoxTokenizer, DiTBlock ×6)
  6. Criterion (Matcher + Loss)
  7. 反向传播验证梯度流

运行方式:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python projects/LDMDet/tests/test_full_pipeline_trace.py
"""

import os
import sys

import torch

# ── 检查 GPU 可用性 ──
assert torch.cuda.is_available(), '需要 GPU'
device = torch.device('cuda:0')

print(f'[SETUP] device={device}  cuda_capable={torch.cuda.is_available()}')
print(f'[SETUP] torch={torch.__version__}')

# ── 加载 MMEngine / MMDetection 配置 ──
from mmengine.config import Config

cfg_path = 'projects/LDMDet/configs/ldmdet_dit.py'
cfg = Config.fromfile(cfg_path)
print(f'[CONFIG] loaded from {cfg_path}')

# ── 手动构建 dataset 与 dataloader ──
from mmdet.datasets import CocoDataset
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import (
    LoadAnnotations,
    LoadImageFromFile,
)
from mmdet.datasets.transforms.transforms import RandomFlip, Resize

data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'

CLASSES = (
    'A1',
    'A2',
    'A3',
    'B4',
    'B5',
    'C10',
    'C11',
    'C12',
    'C6',
    'C7',
    'C8',
    'C9',
    'D13',
    'D14',
    'D15',
    'E16',
    'E17',
    'E18',
    'F19',
    'F20',
    'G21',
    'G22',
    'X',
    'Y',
)
METAINFO = dict(classes=CLASSES)

train_pipeline = [
    LoadImageFromFile(backend_args=None),
    LoadAnnotations(with_bbox=True),
    Resize(scale=(1333, 800), keep_ratio=True),
    RandomFlip(prob=0.5),
    PackDetInputs(),
]

dataset = CocoDataset(
    data_root=data_root,
    ann_file='train/_annotations.coco.json',
    data_prefix=dict(img='train/'),
    pipeline=train_pipeline,
    metainfo=METAINFO,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    test_mode=False,
    serialize_data=False,
)

from mmengine.dataset import pseudo_collate

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    collate_fn=pseudo_collate,
    drop_last=True,
)

print(
    f'[DATA] dataset size={len(dataset)}  dataloader batches={len(dataloader)}'
)

# =====================================================
# STEP 1: 取一个 batch 真实数据
# =====================================================
print('\n' + '=' * 60)
print('STEP 1: 数据加载 & pipeline')
print('=' * 60)

batch = next(iter(dataloader))
data_samples = batch['data_samples']
inputs_list = batch['inputs']  # List[Tensor(H,W,3)], uint8, 不同尺寸

# 检查 inputs 格式: pseudo_collate 返回 list
if isinstance(inputs_list, list):
    print(f'  inputs type=list  len={len(inputs_list)}')
    for i, img in enumerate(inputs_list):
        print(f'  inputs[{i}].shape = {img.shape}  dtype={img.dtype}')
else:
    print(f'  inputs.shape = {inputs_list.shape}')

print(f'  data_samples count = {len(data_samples)}')

for i, ds in enumerate(data_samples):
    h, w = ds.metainfo['img_shape']
    oh, ow = ds.metainfo['ori_shape']
    n_gt = len(ds.gt_instances.bboxes)
    print(
        f'  sample[{i}]: img_shape=({h},{w})  ori_shape=({oh},{ow})  gt_count={n_gt}'
    )
    if n_gt > 0:
        print(f'    gt_bboxes[:2]: {ds.gt_instances.bboxes[:2]}')
        print(f'    gt_labels[:2]: {ds.gt_instances.labels[:2]}')

# =====================================================
# STEP 2: DataPreprocessor (归一化 + Pad)
# =====================================================
print('\n' + '=' * 60)
print('STEP 2: DataPreprocessor')
print('=' * 60)

from mmdet.models.data_preprocessors import DetDataPreprocessor

preprocessor = DetDataPreprocessor(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)
preprocessor = preprocessor.to(device)

# 注意: data_preprocessor 接收的是 list of Tensor (uint8)
data_dict = preprocessor(
    {'inputs': inputs_list, 'data_samples': data_samples}, training=True
)
processed_inputs = data_dict['inputs']  # (B, 3, H', W') after padding
processed_samples = data_dict['data_samples']

print(
    f'  processed_inputs.shape = {processed_inputs.shape}  dtype={processed_inputs.dtype}'
)
print(f'  processed_inputs.min = {processed_inputs.min().item():.3f}')
print(f'  processed_inputs.max = {processed_inputs.max().item():.3f}')
print(f'  processed_inputs.mean = {processed_inputs.mean().item():.3f}')

for i, ds in enumerate(processed_samples):
    h, w = ds.metainfo['img_shape']
    ph, pw = ds.metainfo.get('pad_shape', (h, w))
    print(f'  sample[{i}]: img_shape=({h},{w})  pad_shape=({ph},{pw})')

# =====================================================
# STEP 3: Backbone (DINOv3-Small)
# =====================================================
print('\n' + '=' * 60)
print('STEP 3: Backbone (DINOv3-Small)')
print('=' * 60)

import timm

backbone = timm.create_model(
    'vit_small_patch16_dinov3.lvd1689m',
    pretrained=True,
    features_only=True,
    out_indices=(0, 1, 2, 3),
)
backbone = backbone.to(device).eval()

with torch.no_grad():
    bb_features = backbone(processed_inputs)

print(f'  backbone output count = {len(bb_features)}')
for i, f in enumerate(bb_features):
    print(
        f'  bb_feat[{i}].shape = {f.shape}  min={f.min().item():.3f}  max={f.max().item():.3f}'
    )

# =====================================================
# STEP 4: Neck (SimpleFeatureFusion)
# =====================================================
print('\n' + '=' * 60)
print('STEP 4: Neck (PurePyTorchSimpleFeatureFusion)')
print('=' * 60)

sys.path.insert(0, os.getcwd())
from projects.LDMDet.mods.modules import PurePyTorchSimpleFeatureFusion

neck = PurePyTorchSimpleFeatureFusion(
    in_channels=[384, 384, 384, 384],
    out_channels=384,
    num_outs=4,
)
neck = neck.to(device).eval()

with torch.no_grad():
    neck_features = neck(bb_features)

print(f'  neck output count = {len(neck_features)}')
for i, f in enumerate(neck_features):
    print(
        f'  neck_feat[{i}].shape = {f.shape}  min={f.min().item():.3f}  max={f.max().item():.3f}'
    )

# =====================================================
# STEP 5: Build DiTDiffusionDetHead
# =====================================================
print('\n' + '=' * 60)
print('STEP 5: Build DiTDiffusionDetHead')
print('=' * 60)

from mmengine.registry import init_default_scope

from mmdet.registry import MODELS

init_default_scope('mmdet')

# 注册自定义模块
import projects.LDMDet.model
import projects.LDMDet.mods.dit_head
import projects.LDMDet.mods.dit_single_head
import projects.LDMDet.mods.loss  # noqa

# 从配置构建 head
head_cfg = cfg.model.bbox_head
bbox_head = MODELS.build(head_cfg)

# 手动构建 criterion（用 PurePyTorchDiffusionDet._build_criterion）
from projects.LDMDet.model import PurePyTorchDiffusionDet

detector = PurePyTorchDiffusionDet(
    backbone=dict(type='ResNet', depth=50),
    neck=None,
    bbox_head=head_cfg,
)
criterion = detector._build_criterion(dict(cfg.model.bbox_head.criterion))
bbox_head.criterion = criterion

bbox_head = bbox_head.to(device)
bbox_head.train()

print(f'  bbox_head type = {type(bbox_head).__name__}')
print(f'  num_heads = {len(bbox_head.head_series)}')
print(f'  feat_channels = {bbox_head.feat_channels}')
print(f'  num_proposals = {bbox_head.num_proposals}')

# 检查权重共享
head_params = [id(p) for p in bbox_head.head_series[0].parameters()]
all_same = all(
    id(p1) == id(p2)
    for head in bbox_head.head_series[1:]
    for p1, p2 in zip(bbox_head.head_series[0].parameters(), head.parameters())
)
print(f'  weight_sharing_active = {all_same}')

# =====================================================
# STEP 6: Prepare targets for loss
# =====================================================
print('\n' + '=' * 60)
print('STEP 6: Prepare targets')
print('=' * 60)

img_metas = []
gt_bboxes = []
gt_labels = []

for ds in processed_samples:
    meta = dict(
        img_shape=ds.metainfo['img_shape'],
        pad_shape=ds.metainfo.get('pad_shape', ds.metainfo['img_shape']),
        ori_shape=ds.metainfo.get('ori_shape', ds.metainfo['img_shape']),
        scale_factor=ds.metainfo.get('scale_factor', [1.0, 1.0]),
    )
    img_metas.append(meta)
    gt_bboxes.append(ds.gt_instances.bboxes.to(device))
    gt_labels.append(ds.gt_instances.labels.to(device))

print(f'  batch_size = {len(img_metas)}')
for i, (m, b, l) in enumerate(zip(img_metas, gt_bboxes, gt_labels)):
    print(
        f'  img[{i}]: {m["img_shape"]}  gt_bboxes={b.shape}  gt_labels={l.shape}'
    )

# =====================================================
# STEP 7: DiTDiffusionDetHead.forward (train mode)
# =====================================================
print('\n' + '=' * 60)
print('STEP 7: DiTDiffusionDetHead.loss (full forward + loss)')
print('=' * 60)

# 清理 CUDA 缓存，避免 OOM
torch.cuda.empty_cache()

# 注意：需要把 neck_features 转为 list of tensors
features_for_head = [f.to(device) for f in neck_features]

try:
    losses = bbox_head.loss(features_for_head, img_metas, gt_bboxes, gt_labels)
    print(f'  loss keys = {list(losses.keys())}')
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f'    {k} = {v.item():.4f}')
        else:
            print(f'    {k} = {v}')
except Exception as e:
    print(f'  [ERROR] loss() failed: {type(e).__name__}: {e}')
    import traceback

    traceback.print_exc()
    print('  Trying forward() directly for diagnosis...')

    # Fall back to manual forward
    with torch.no_grad():
        try:
            outputs = bbox_head.forward(
                features_for_head, None, torch.zeros(2, device=device)
            )
            print('  forward() outputs:')
            for i, o in enumerate(outputs):
                if isinstance(o, torch.Tensor):
                    print(f'    output[{i}].shape = {o.shape}')
        except Exception as e2:
            print(f'  forward() also failed: {type(e2).__name__}: {e2}')

# =====================================================
# STEP 8: 反向传播验证梯度流
# =====================================================
print('\n' + '=' * 60)
print('STEP 8: 反向传播 & 梯度流检查')
print('=' * 60)

# 只在 loss 成功计算的情况下做 BP
if 'losses' in locals() and isinstance(losses, dict):
    total_loss = sum(losses.values())
    total_loss.backward()

    # 检查各组件梯度
    named_params = {
        'backbone.blocks.0': backbone.blocks[0]
        if hasattr(backbone, 'blocks')
        else None,
    }

    grad_stats = {}
    for name, param in bbox_head.named_parameters():
        if param.grad is not None:
            g = param.grad
            grad_stats[name] = {
                'norm': g.norm().item(),
                'min': g.min().item(),
                'max': g.max().item(),
                'mean': g.mean().item(),
                'finite': torch.isfinite(g).all().item(),
            }

    # 按梯度范数排序，取前 10
    sorted_grads = sorted(
        grad_stats.items(), key=lambda x: x[1]['norm'], reverse=True
    )
    print(f'  Parameters with gradients: {len(grad_stats)}')
    print('  Top-10 gradients (by norm):')
    for name, stats in sorted_grads[:10]:
        print(
            f'    {name}: norm={stats["norm"]:.4f}  max={stats["max"]:.4f}  finite={stats["finite"]}'
        )

    nan_params = [n for n, s in sorted_grads if not s['finite']]
    zero_params = [
        n for n, s in sorted_grads if s['norm'] < 1e-8 and s['finite']
    ]
    if nan_params:
        print(f'  [WARN] NaN gradients in: {nan_params[:5]}')
    if zero_params:
        print(f'  [WARN] Zero gradients in: {zero_params[:5]}')

    # 检查 backbone 梯度
    print('  Backbone grad check:')
    for name, param in backbone.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            print(f'    backbone.{name}: grad_norm={g_norm:.4f}')
            break  # just check one

    print('\n  [OK] Backward pass completed successfully')
else:
    print('  [SKIP] backward pass (loss computation failed)')

# =====================================================
# STEP 9: 稳定性检查
# =====================================================
print('\n' + '=' * 60)
print('STEP 9: 数值稳定性检查')
print('=' * 60)

# 检查模型输出是否包含 NaN
with torch.no_grad():
    try:
        bbox_head.eval()
        dummy_bboxes = torch.rand(2, 300, 4, device=device).clamp(0, 1)
        dummy_t = torch.rand(2, device=device) * 4

        outputs = bbox_head.forward(features_for_head, dummy_bboxes, dummy_t)

        print('  forward outputs:')
        for i, o in enumerate(outputs):
            if isinstance(o, torch.Tensor):
                has_nan = torch.isnan(o).any().item()
                has_inf = torch.isinf(o).any().item()
                print(
                    f'    output[{i}].shape={o.shape}  has_nan={has_nan}  has_inf={has_inf}'
                )

        # 检查 cls logits 和 pred bboxes
        cls_logits = outputs[0]
        pred_bboxes = outputs[1]
        print(
            f'\n  cls_logits: min={cls_logits.min().item():.3f}  max={cls_logits.max().item():.3f}  mean={cls_logits.mean().item():.3f}'
        )
        print(
            f'  pred_bboxes: min={pred_bboxes.min().item():.3f}  max={pred_bboxes.max().item():.3f}  mean={pred_bboxes.mean().item():.3f}'
        )

        bbox_head.train()
    except Exception as e:
        print(
            f'  [WARN] forward stability check failed: {type(e).__name__}: {e}'
        )

print('\n' + '=' * 60)
print('TEST COMPLETE')
print('=' * 60)
