import sys
import time
from pathlib import Path

import torch
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

register_all_modules()

# 导入我们的模型包装类，触发注册
try:
    import projects.LDMDet.model as model_mod

    LDMDet = model_mod.LDMDet
except Exception as e:
    print(f'Failed to import LDMDet: {e}')
    sys.exit(1)


def get_model():
    num_classes = 24
    model_cfg = dict(
        type='LDMDet',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
        ),
        backbone=dict(
            type='ResNet',
            depth=50,  # 使用 ResNet50 进行更真实的基准测试
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
        ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4),
        bbox_head=dict(
            type='PurePyTorchDiffusionDetHead',
            num_classes=num_classes,
            feat_channels=256,
            num_proposals=300,  # 典型的提议数量
            num_heads=6,  # 典型的头部数量
            single_head=dict(
                type='PurePyTorchSingleDiffusionDetHead',
                num_classes=num_classes,
                num_cls_convs=3,
                num_reg_convs=3,
            ),
            roi_extractor=dict(
                type='PurePyTorchSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
            ),
            criterion=dict(
                type='PurePyTorchDiffusionDetCriterion',
                num_classes=num_classes,
                assigner=dict(
                    type='PurePyTorchDiffusionDetMatcher',
                    match_costs=[
                        dict(type='PurePyTorchFocalLossCost', weight=2.0),
                        dict(
                            type='PurePyTorchBBoxL1Cost',
                            weight=5.0,
                            box_format='xyxy'),
                        dict(
                            type='PurePyTorchIoUCost',
                            iou_mode='giou',
                            weight=2.0),
                    ],
                ),
                loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0),
                loss_bbox=dict(type='PurePyTorchL1Loss', loss_weight=5.0),
                loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=2.0),
            ),
        ),
    )
    model = MODELS.build(model_cfg)
    return model


def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Benchmarking on {device}...')

    model = get_model().to(device)
    model.eval()

    # 准备输入数据
    batch_size = 1  # 减小 batch size 以避免 OOM，但仍可测量单图速度
    img = torch.randn(batch_size, 3, 800, 800).to(device)

    data_samples = []
    for i in range(batch_size):
        data_sample = DetDataSample()
        data_sample.set_metainfo(
            dict(
                img_shape=(800, 800),
                ori_shape=(800, 800),
                pad_shape=(800, 800),
                batch_input_shape=(800, 800),
            ))
        # 训练模式需要的 GT
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.tensor([[10, 10, 100, 100]],
                                           dtype=torch.float32).to(device)
        gt_instances.labels = torch.tensor([0], dtype=torch.int64).to(device)
        data_sample.gt_instances = gt_instances
        data_samples.append(data_sample)

    # 预热
    print('Warming up...')
    for _ in range(10):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                model.test_step(dict(inputs=img, data_samples=data_samples))

    # 测试推理延迟
    print('Measuring inference latency (Mixed Precision)...')
    torch.cuda.synchronize()
    start_time = time.time()
    num_iters = 50

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    latencies = []

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in range(num_iters):
                starter.record()
                model.test_step(dict(inputs=img, data_samples=data_samples))
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))

    avg_latency = sum(latencies) / num_iters
    fps = (batch_size * 1000) / avg_latency
    print(f'Inference Latency: {avg_latency:.2f} ms')
    print(f'FPS: {fps:.2f}')

    # 测试训练延迟 (Forward + Backward)
    print(
        'Measuring training latency (Forward + Backward, Mixed Precision)...')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    train_latencies = []
    for _ in range(num_iters):
        optimizer.zero_grad()
        starter.record()

        with torch.cuda.amp.autocast():
            loss_dict = model.loss(img, data_samples)
            losses = sum(loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        ender.record()
        torch.cuda.synchronize()
        train_latencies.append(starter.elapsed_time(ender))

    avg_train_latency = sum(train_latencies) / num_iters
    print(f'Training Latency: {avg_train_latency:.2f} ms')

    # 显存占用
    print(
        f'Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB'
    )
    print(
        f'Max Memory Reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024:.2f} MB'
    )


if __name__ == '__main__':
    benchmark()
